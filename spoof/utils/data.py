import json
import os
from pathlib import Path
import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset


def capture_frames(src: str, dest: str) -> int:
    """Captures frames of a video."""
    # Open and start to read the video
    cap = cv2.VideoCapture(src)

    if cap.isOpened():
        cur_frame = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"{dest}_frame_{cur_frame}_.jpg"
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping...")
                continue
            cv2.imwrite(filename=filename, img=frame)
            cur_frame += 1

        cap.release()
        cv2.destroyAllWindows()
        return cur_frame

    else:
        print("Cannot open video file")
        return 0


def create_image_folder(
    video_src: str,
    img_folder_root: str,
    video_ext: str = ".avi",
) -> int:
    """Creates an image folder from all the videos in the directory provided. Resulting folder is
    in torchvision ImageFolder format.

    In CASIA, if the video filename ends with 0 it's a spoof, if it ends with
    1 it's real.

    img_folder_root/real/xxx.png
    img_folder_root/real/xxy.jpeg
    img_folder_root/real/xxz.png
    .
    .
    .
    img_folder_root/spoof/123.jpg
    img_folder_root/spoof/nsdf3.png
    img_folder_root/spoof/asd932_.png
    """

    videos = [str(p) for p in Path(video_src).rglob(f"*{video_ext}")]

    # Extract frames
    total_frame_count = 0
    print(f"Found {len(videos)} videos in {video_src}")
    for video in videos:
        print(f"Processing {video}")
        if Path(video).stem[-1] == "0":
            label = "spoof"
        else:
            label = "real"

        # Create the destination folder
        Path(img_folder_root).joinpath(label).mkdir(parents=True, exist_ok=True)

        # Extract frames
        frame_count = capture_frames(
            src=video,
            dest=str(Path(img_folder_root).joinpath(label).joinpath(Path(video).stem)),
        )

        total_frame_count += frame_count
    print(f"Extracted {total_frame_count} frames")
    print(f"Extracted {total_frame_count / len(videos)} frames per video")
    print(f"Extracted {total_frame_count / len(videos) / 30} seconds per video")
    print("Finished extracting frames.")
    return total_frame_count


def create_annotations(
    metadata_root: str, extracted_frames_root: str, annotations_path: str
):
    """Creates the annotations file for the CASIA dataset. Only the face rectangle and landmarks
    are extracted from the metadata. If a key is missing, meaning that the extracted frames are
    more than the metadata, the frame is skipped.

    Args:
        metadata_root: Root folder of the metadata.
        extracted_frames_root: Root folder of the extracted frames.
        annotations_path: Path to save the annotations file.
    Format:
        ``[image_path, face_rectangle, face_landmarks]``

    The face_rectangle and face_landmarks are in the following format:

    `[x_1, y_1, x_2, y_2]`

    where `x1, y1, x2, y2` are the coordinates of the top left and bottom right
    corners of the rectangle.

    The face_landmarks are in the following format:
        `[x1, y1, x2, y2, ..., x5, y5, x6, y6, x7, y7]`

    where `x1, y1, x2, y2, ..., x7, y7` are the coordinates of the 7 landmarks.

    The landmarks are in the following order:

        1. left eye center
        2. right eye center
        3. nose tip
        4. left mouth corner
        5. right mouth corner
        6. left ear
        7. right ear

    Returns:
        annotations: List of tuples containing the image path, face rectangle
            and face landmarks.
    """
    # Skip if annotations file already exist
    if os.path.exists(annotations_path):
        print(f"File {annotations_path} already exists. Skipping...")
        return

    # Get all the metadata files
    metadata_files = [str(p) for p in Path(metadata_root).rglob("*json")]
    print(f"Found {len(metadata_files)} metadata files")

    # Get all the extracted frames
    extracted_frames = [str(p) for p in Path(extracted_frames_root).rglob("*.jpg")]
    print(f"Found {len(extracted_frames)} extracted frames")

    # Create the annotations file
    annotations = []
    print("Creating annotations...")
    for metadata_file in metadata_files:
        # Get the corresponding extracted frames
        extracted_frames = [
            str(p)
            for p in Path(extracted_frames_root).rglob(f"{Path(metadata_file).stem}*.jpg")
        ]

        # Read the metadata file
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Get the face rectangle and landmarks using the frame number as key
        face_rectangles = {int(k): v["face_rect"] for k, v in metadata.items()}
        face_landmarks = {int(k): v["lm7pt"] for k, v in metadata.items()}

        # Rename face_rect to face_rectangle
        for k, v in metadata.items():
            v["face_rectangle"] = v.pop("face_rect")

        # Rename lm7pt to landmarks
        for k, v in metadata.items():
            v["landmarks"] = v.pop("lm7pt")

        # Create the annotations
        for frame in extracted_frames:
            frame_num = int(Path(frame).stem.split("_")[-2])
            rel_frame_path = str(Path(frame).relative_to(extracted_frames_root))
            label = 1 if rel_frame_path.split("/")[0] == "real" else 0
            if frame_num in face_rectangles and frame_num in face_landmarks:
                annotations.append(
                    (
                        rel_frame_path,
                        face_rectangles[frame_num],
                        face_landmarks[frame_num],
                        label,
                    )
                )

    # Save the annotations as a json file
    print(f"Saving annotations to {annotations_path}")
    with open(annotations_path, "w") as f:
        json.dump(annotations, f)

    print(f"Created {len(annotations)} annotations")
