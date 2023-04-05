import os
from functools import partial

import cv2
import hydra
from helpers import apply_map, filter_files_by_ext, get_video_params
from omegaconf import DictConfig


# Function to extract frames
def capture_frames(src: str, dest: str) -> None:
    """Captures video frames from root src and saves them to dest folder."""
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
    src, dest, ext = get_video_params(cfg)

    # videos and metadata are generators
    videos = filter_files_by_ext(src, ext)
    casia_capturer = partial(capture_frames, dest=dest)

    # apply capturer to all videos
    apply_map(casia_capturer, videos)


if __name__ == "__main__":
    main()
