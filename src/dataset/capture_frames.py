import os
import sys
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

# Add parent directory to path for easy import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import filter_files_by_ext


# Function to extract frames
def capture_frames(src: str, dest: str) -> None:
    """
    Extracts frames from a video and saves them to a directory.
    """
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

        # print("Finished extracting frames from {}".format(src))
        print("Frames saved to {}".format(dest))
        print("Total frames: {}".format(cur_frame))

    else:
        print("Cannot open video file")
    cv2.destroyAllWindows()


# Function to extract frames from videos
def extract_frames(src: str, dest: str, ext: str) -> None:
    """
    Extracts frames from all videos in src and saves them to dest
    with the following structure:

    dest/client_name/live/video_name/frame{frame number}.jpg
    dest/client_name/spoof/video_name/frame{frame number}.jpg


    Args:
        src (str): path to video files
        dest (str): path to destination folder
        ext (str): video file extension

    """
    # Get list of video files
    video_files = filter_files_by_ext(src, ext)
    # Extract frames
    for video_file in video_files:
        input_filename = Path(video_file).stem
        client = Path(video_file).parent.stem
        label = input_filename.split("_")[-1]
        if label == "1":
            # live video
            output_directory = Path(dest) / client / "live" / input_filename
        elif label == "0":
            # spoof video
            output_directory = Path(dest) / client / "spoof" / input_filename
        else:
            print("Invalid label: {}".format(label))
            continue

        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        capture_frames(video_file, str(output_directory))


@hydra.main(version_base="1.2", config_path="../..", config_name="config")
def main(cfg: DictConfig) -> None:
    # Extract train data
    extract_frames(
        cfg.dataset.train_src, cfg.dataset.train_dest, cfg.dataset.ext
    )

    # Extract test data
    extract_frames(
        cfg.dataset.test_src, cfg.dataset.test_dest, cfg.dataset.ext
    )


if __name__ == "__main__":
    main()
