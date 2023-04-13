import json
import logging
import os
from pathlib import Path

from torchvision.transforms import Compose

from transforms import FaceRegionRCXT, MetaAddLMSquare

logger = logging.getLogger(__name__)


def get_transforms(args):
    """Returns the transforms for the training and testing datasets."""
    if args.dataset == "casia":
        align = FaceRegionRCXT(size=(224, 224))
        sq = MetaAddLMSquare()
        transform = Compose([sq, align])
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    return transform


# This function is common for all datasets so it's placed in utils
def create_annotations(
    metadata_root: str,
    extracted_frames_root: str,
    annotations_path: str,
):
    """Creates the annotations file for the dataset. Only the face rectangle and landmarks are
    extracted from the metadata. If a key is missing, meaning that the extracted frames are more
    than the metadata, the frame is skipped.

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
    # Skip if annotations file already exist and not overwriting
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
        print(extracted_frames)

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
            frame_num = int(Path(frame).stem.split("_")[-1])
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
