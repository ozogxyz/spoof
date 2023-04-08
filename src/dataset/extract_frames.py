import csv
import os
from pathlib import Path

import cv2


def filter_files_by_ext(path: str, ext: str):
    for path in Path(path).rglob(f"*{ext}"):  # type: ignore
        yield str(path)


# Function to extract frames
def capture_frames(src: str, dest: str, label: str) -> None:
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
            # get src filename without extension
            frame_name = create_filename(src, label, cur_frame)
            filename = os.path.join(dest, frame_name)
            cv2.imwrite(filename, frame)
            cur_frame += 1
        cap.release()

        print("Finished extracting frames from {}".format(src))
        print("Frames saved to {}".format(dest))
        print("Total frames: {}".format(cur_frame))

    else:
        print("Cannot open video file")
    cv2.destroyAllWindows()


def create_filename(src, label, cur_frame):
    video_filename = Path(src).stem
    frame_name = (
        video_filename + "_frame_" + str(cur_frame) + "_" + label + ".jpg"
    )

    return frame_name


def create_labels_csv(src: str, dest: str, ext: str = ".jpg") -> None:
    """
    Creates a csv file with the following structure:

    frame_name,label

    Args:
        src (str): path to video files
        dest (str): path to destination folder
        ext (str): video file extension

    """
    # Get list of video files
    frame_files = filter_files_by_ext(src, ext)

    with open(os.path.join(dest, "labels.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "Label"])

        for file in frame_files:
            filename = Path(file).name
            label = Path(file).stem.split("_")[-1]
            writer.writerow([filename, label])


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

        capture_frames(video_file, str(output_directory), label)
