import os

import sys
from functools import partial
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

# Add parent directory to path for easy import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import apply_map, filter_files_by_ext, get_video_params


# Function to extract frames
def capture_frames(src: str, dest: str) -> None:
    """
    Captures video frames from root src and extracts them to dest folder.

    Args:
        src (str): path to video file
        dest (str): path to destination folder

    """
    print(f"Capturing frames from {src} to {dest}")
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        cur_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(dest, "frame%d.jpg" % cur_frame), frame)
            cur_frame += 1
        cap.release()
    else:
        print("Cannot open video file")
    cv2.destroyAllWindows()


@hydra.main(version_base="1.2", config_path="../../", config_name="config")
def main(cfg: DictConfig):
    # quick check for casia
    casia_src, casia_dest, casia_ext = get_video_params(cfg)

    # get all videos in the dataset
    all_videos = filter_files_by_ext(casia_src, casia_ext)

    # create partial function with extract location for casia
    casia_capturer = partial(capture_frames, dest=casia_dest)

    # apply capturer to all videos
    apply_map(casia_capturer, all_videos)


if __name__ == "__main__":
    main()
