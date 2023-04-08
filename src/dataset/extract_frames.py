import csv
import json
import os
from pathlib import Path

import cv2


def filter_files_by_ext(path: str, ext: str):
    for path in Path(path).rglob(f"*{ext}"):  # type: ignore
        yield str(path)


# Function to extract frames
def capture_frames(
    video_src: str, metadata_src: str, dest: str, label: str
) -> None:
    """Extracts frames from a video and saves them to a directory. Simultaneously saves the
    corresponding metadata to a json file. The filenames are determined as follows:

    {video_filename}_frame_{cur_frame}_{label}.jpg
    {video_filename}_frame_{cur_frame}_{label}.json

    Args:
        video_src: path to the video file
        metadata_src: path to the metadata file
        dest: path to the directory where frames will be saved
        label: label of the video
    """
    # Open and start to read the video
    cap = cv2.VideoCapture(video_src)
    print("Started extracting frames from {}".format(video_src))

    # Load the corresponding metadata
    print("Loading metadata from {}".format(metadata_src))
    if os.path.exists(metadata_src):
        video_metadata = json.loads(metadata_src)
    else:
        raise FileNotFoundError(
            "Metadata file {} does not exist".format(metadata_src)
        )

    if cap.isOpened():
        cur_frame = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame
            frame_name = create_filename(video_src, label, cur_frame)
            frame_filename = frame_name + ".jpg"
            cv2.imwrite(os.path.join(dest, frame_filename), frame)

            # Save the metadata (only face_rect and face_landmarks)
            frame_meta = video_metadata.get(str(cur_frame))
            frame_meta = {k: frame_meta[k] for k in list(frame_meta.keys())[:2]}

            with open(frame_name + ".json", "w") as f:
                json.dump(frame_meta, f)

            cur_frame += 1
        cap.release()

        print("Finished extracting frames from {}".format(video_src))
        print("Frames saved to {}".format(dest))
        print("Total frames: {}".format(cur_frame))

        print()

        print("Finished extracting metadata from {}".format(video_src))
        print("Metadata saved to {}".format(dest))
        print("Total metadata: {}".format(cur_frame))

    else:
        print("Cannot open video file")
    cv2.destroyAllWindows()


def create_filename(src: str, label: str, cur_frame: int):
    video_filename = Path(src).stem
    frame_name = video_filename + "_frame_" + str(cur_frame) + "_" + label

    return frame_name


def create_labels_csv(
    frame_src: str,
    dest: str,
    frame_ext: str = ".jpg",
    metadata_ext: str = ".json",
) -> None:
    """Creates a csv file with the following structure:

    Path to frame, path to its metadata, label

    Args:
        frame_src (str): path to the root of the extracted frame files
        dest (str): path to destination folder
        frame_ext (str): frame file extension
        metadata_ext (str): metadata file extension
    """
    # Since the metadata and frame files have the same name and only
    # differ in extension, we can use the same name for both

    frame_files = filter_files_by_ext(frame_src, ext=frame_ext)

    with open(os.path.join(dest, "labels.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame_Name", "Metadata_Name", "Label"])

        for frame_file in frame_files:
            frame_name = Path(frame_file)
            frame_filename = frame_name.with_suffix(frame_ext)
            metadata_filename = frame_name.with_suffix(metadata_ext)
            label = Path(frame_file).stem.split("_")[-1]
            writer.writerow([frame_filename, metadata_filename, label])


# Function to extract frames from videos
def extract_frames(
    video_src: str,
    metadata_src: str,
    dest: str,
    video_ext: str = ".avi",
    metadata_ext: str = ".json",
) -> None:
    """Extracts frames and metadata from all videos in src and saves them to dest with the
    following structure:

    dest/client_name/live/video_name/video_name_frame_{frame number}.jpg
    dest/client_name/live/video_name/video_name_frame_{frame number}.json

    dest/client_name/spoof/video_name/video_name_frame_{frame number}.jpg
    dest/client_name/spoof/video_name/video_name_frame_{frame number}.json


    Args:
        video_src (str): path to the directory containing the videos
        metadata_src (str): path to the directory containing the metadata
        dest (str): path to the directory where frames will be saved
        video_ext (str): video file extension
        metadata_ext (str): metadata file extension
    """
    # Get list of video files
    video_files = filter_files_by_ext(video_src, ext=video_ext)

    # Extract frames
    for video_file in video_files:
        input_filename = Path(video_file).stem

        # Get the client name
        client = Path(video_file).parent.stem

        # Get the corresponding metadata filename
        metadata_src = os.path.join(
            metadata_src, client, input_filename + metadata_ext
        )

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

        capture_frames(video_file, metadata_src, str(output_directory), label)
