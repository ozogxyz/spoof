from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import csv


# Helper functions
def get_output_dir(video_path, dest):
    """Returns the output directory for the frames to be extracted from the video. If the video
    filename ends with _1, then label is real, otherwise spoof. Output directory structure:

    {dest}/{client}/real/{video_filename}
    {dest}/{client}/spoof/{video_filename}

    Args:
        video_path (str): path to the video file
        dest (str): path to the directory where frames will be saved
    Returns:
        output_directory (str): path to the output directory
    """
    input_filename = Path(video_path).stem

    # Get the client name
    # get the parent directory of the video
    client = Path(video_path).parent.name
    label = get_label(input_filename)
    if label == "1":
        # real video
        output_directory = Path(dest) / client / "real" / input_filename
    elif label == "0":
        # spoof video
        output_directory = Path(dest) / client / "spoof" / input_filename
    else:
        print("Invalid label: {}".format(label))
        return None

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_directory


def filter_files_by_ext(path: str, ext: str):
    for path in Path(path).rglob(f"*{ext}"):
        yield str(path)


def create_filename(save_dest: str, label, frame_number, ext):
    """Creates the filename for the frame with the following structure:

    {video_filename}_frame_{cur_frame}_{label}.jpg
    """
    filename = f"{save_dest}_frame_{frame_number}_{label}"
    filename = Path(filename).with_suffix(ext)

    return filename


def get_label(video_path: str) -> str:
    return "1" if Path(video_path).stem.endswith("_1") else "0"


def capture_frames(video_path: str, save_dest: str) -> int:
    """Extracts frames from a video. If video filename ends with _1, then label is real, otherwise
    spoof.

    {video_filename}_frame_{cur_frame}_{label}.jpg
    Args:
        video_path: path to the video file
        save_dest: path to the directory where frames will be saved
    Returns:
        cur_frame: number of frames extracted
    """
    # Open and start to read the video
    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem
    save_dest = str(Path(save_dest) / video_name)
    if cap.isOpened():
        cur_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame
            label = get_label(video_path)
            filename = create_filename(save_dest, label, cur_frame, ".jpg")
            cv2.imwrite(filename=str(filename), img=frame)
            cur_frame += 1

        cap.release()

        print("Finished extracting frames: {}".format(Path(video_path).stem))
        print("Total frames: {}".format(cur_frame))
        cv2.destroyAllWindows()
        return cur_frame

    else:
        print("Cannot open video file")
        return 0


def extract_frames(
    video_src: str,
    save_dest: str,
    video_ext: str = ".avi",
) -> int:
    """Extracts frames and metadata from all videos in src and saves them to dest with the
    following structure:

    dest/client_name/live/video_name/video_name_frame_{frame number}_{label}.jpg
    dest/client_name/spoof/video_name/video_name_frame_{frame number}_{label}.jpg
    Args:
        video_src (str): path to the directory containing the videos
        dest (str): path to the directory where frames will be saved
        video_ext (str): video file extension
    Returns:
        total_frame_count (int): total number of frames extracted
    """
    # Get list of video files
    video_files = filter_files_by_ext(video_src, ext=video_ext)

    # Extract frames
    total_frame_count = 0
    for video_file in video_files:
        output_directory = get_output_dir(video_file, save_dest)
        frame_count = capture_frames(video_file, str(output_directory))
        total_frame_count += frame_count

    print("Total frames extracted: {}".format(total_frame_count))

    return total_frame_count


def extract_meta_per_frame(meta_path, save_dest) -> int:
    meta_name = Path(meta_path).stem
    save_dest = str(Path(save_dest) / meta_name)
    with open(meta_path, "r") as f:
        cur_frame = 0
        metadata = json.load(f)
        for frame, meta in metadata.items():
            # Only get the first 2 keys namely face_rect and lm7pts
            face_rect_lm7pt = dict(itertools.islice(meta.items(), 2))
            label = get_label(meta_path)
            # frame 0 is pitch black
            filename = create_filename(save_dest, label, cur_frame + 1, ".json")
            with open(filename, "w") as f:
                json.dump(face_rect_lm7pt, f)

            cur_frame += 1
        return cur_frame


def extract_metadata(metadata_root: str, save_dest: str) -> int:
    """Extracts metadata from a video. Saves the metadata in a json file. Save location is the
    directory of the frames extracted from the video file.

    {video_filename}_frame_{cur_frame}_{label}.json
    Args:
        metadata_root: path to the metadata directory
        save_dest: path to the directory where frames will be saved
    Returns:
        cur_frame: number of frames extracted (extract_frames.py)
    """
    # Get list of the metadata files
    metadata_files = filter_files_by_ext(metadata_root, ".json")

    # Iterate through the metadata files
    total_frame_count = 0
    for metadata_file in metadata_files:
        output_directory = get_output_dir(metadata_file, save_dest)
        frame_count = extract_meta_per_frame(
            metadata_file, str(output_directory)
        )
        print("Extracting metadata from {}".format(Path(metadata_file)))
        total_frame_count += frame_count
        print("Total metadata extracted for frames: {}".format(frame_count))

    return total_frame_count


class CreateSampleDict:
    def __call__(self, sample: Dict) -> Dict:
        return self.create_sample(sample["meta"], sample["image"])

    def _validate_face_rect(self, meta: Dict, key: str) -> Dict:
        """Validate if face_rect is in correct format."""
        face_rect = meta.get(key)
        assert face_rect is not None, "Face rect not found in meta"
        if isinstance(face_rect, list):
            face_rect = np.array(face_rect).ravel()
        assert face_rect.shape == (4,), "Face rect shape is wrong"
        meta[key] = face_rect

        return meta

    def _validate_landmarks(self, meta: Dict, key: str) -> Dict:
        """Validate if landmark is in correct format."""
        face_landmarks = meta.get(key)
        assert face_landmarks is not None, "Landmark not found in meta"
        if isinstance(face_landmarks, list):
            face_landmarks = np.array(face_landmarks).ravel()
        if len(face_landmarks) == 14:
            face_landmarks = face_landmarks.reshape(7, -1)
        assert face_landmarks.shape == (
            7,
            2,
        ), "Landmark shape is wrong: {}".format(face_landmarks.shape)

        meta[key] = face_landmarks

        return meta

    def _rename_keys(self, meta: Dict, new_key: str, old_key: str) -> Dict:
        if old_key in meta.keys():
            meta = deepcopy(meta)
            meta[new_key] = meta.pop(old_key)

        return meta

    def create_sample(self, meta: Dict, frame) -> Dict:
        """Add metadata in our format to a frame np array.

        For now only adheres to CASIA format, ideally should convert
        any metadata to our format. In CASIA keys are face_rect
        and lm7pt.
        Args:
            meta: metadata of a frame obtained from data
            frame: np array of a frame
        Returns:
            sample: a dict with keys "image" and "meta"
        """
        meta = self._rename_keys(meta, "face_landmark", "lm7pt")
        meta = self._validate_face_rect(meta, "face_rect")
        meta = self._validate_landmarks(meta, "face_landmark")

        sample = {
            "image": frame,
            "meta": meta,
        }

        return sample


class CASIA(Dataset):
    """CASIA dataset.

    Args:
        dataset_dir: path to the dataset directory
        annotations: path to the annotation file with relative path to a frame and its labels
        transform: transform to apply to the sample
    """

    def __init__(
        self,
        dataset_dir: str,
        annotations_file: str,
        transform: Callable,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.train_list = []
        self.transform = transform
        train_file_buf = open(annotations_file, "r")
        line = train_file_buf.readline().strip()
        while line:
            image_path, label = line.split(",")
            self.train_list.append((image_path, int(label)))
            line = train_file_buf.readline().strip()

    def __len__(self) -> int:
        return len(self.train_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.train_list[index]
        image_path = os.path.join(self.dataset_dir, image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.transpose(2, 0, 1).astype(np.float32)
        image = torch.from_numpy(image)

        return image, label
