import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CASIA(Dataset):
    """CASIA dataset."""

    def __init__(
        self,
        annotations_path: str,
        video_root: str,
        img_root: str,
        extract=False,
        transform=None,
    ):
        """
        Args:
            annotations_path (string): Path to the json file with annotations.
            video_root (string): Directory with all the videos.
            img_root (string): Directory with all the images.
            extract (bool): If True, extract frames from videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if os.path.exists(annotations_path):
            with open(annotations_path, "r") as f:
                self.annotations = json.load(f)
        else:
            raise ValueError(f"Annotations file {annotations_path} not found")
        if extract and os.path.exists(video_root):
            self._extract(video_root, img_root)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_root, self.annotations[idx][0])

        face_rect = self.annotations[idx][1]
        face_landmarks = self.annotations[idx][2]
        face_landmarks = np.array(face_landmarks).reshape(-1, 2)
        label = self.annotations[idx][3]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        meta = {"face_rect": face_rect, "face_landmark": face_landmarks}
        sample = {"image": image, "meta": meta}

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def _capture_frames(self, src: str, dest: str) -> int:
        """Captures frames of a video."""
        # Open and start to read the video
        cap = cv2.VideoCapture(src)

        if cap.isOpened():
            cur_frame = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                filename = f"{dest}_frame_{cur_frame}.jpg"
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

    def _extract(
        self,
        root_dir: str,
        extract_to: str,
        video_ext: str = ".avi",
    ) -> int:
        """Creates an image folder from all the videos in the directory provided. Resulting folder is
        in torchvision ImageFolder format.
        In CASIA, if the video filename ends with 0 it's a spoof, if it ends with
        1 it's real.
        extract_to/real/xxx.png
        extract_to/real/xxy.jpeg
        extract_to/real/xxz.png
        .
        .
        .
        extract_to/spoof/123.jpg
        extract_to/spoof/nsdf3.png
        extract_to/spoof/asd932_.png
        """

        videos = [str(p) for p in Path(root_dir).rglob(f"*{video_ext}")]

        # Extract frames
        total_frame_count = 0
        print(f"Found {len(videos)} videos in {root_dir}")
        for video in videos:
            print(f"Processing {video}")
            if Path(video).stem[-1] == "0":
                label = "spoof"
            else:
                label = "real"

            # Create the destination folder
            Path(extract_to).joinpath(label).mkdir(parents=True, exist_ok=True)

            # Extract frames
            frame_count = self._capture_frames(
                src=video,
                dest=str(Path(extract_to).joinpath(label).joinpath(Path(video).stem)),
            )

            total_frame_count += frame_count
        print(f"Extracted {total_frame_count} frames")
        print("Finished extracting frames.")
        return total_frame_count


class LCC_FASD:
    pass


class ReplayAttack:
    pass
