import argparse
import csv
import json
import os
from logging import getLogger
from pathlib import Path

import cv2
from tqdm import tqdm

logger = getLogger(__name__)
logger.setLevel("DEBUG")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--data_root", type=str, default="data")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="casia",
        choices=["casia", "replay", "oulu"],
        help="Dataset name",
    )

    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Preparing {args.dataset} dataset")

    dataset_splits = {
        "casia": ["train", "val", "test"],
        "replay": ["train", "devel", "test"],
        "oulu": ["train", "dev", "test"],
    }

    splits = dataset_splits.get(args.dataset)
    if not splits:
        logger.error(f"Unknown dataset {args.dataset}")
        return

    for split in splits:
        print(f"Processing {split} split")
        video_dir = Path(args.data_root) / args.dataset / "video" / split
        if not video_dir.exists():
            logger.error(f"{video_dir} does not exist")
            return

        video_ext = get_video_extension(args.dataset)
        videos = find_videos(video_dir, video_ext)

        # for video in tqdm(
        #     videos,
        #     desc="Extracting frames",
        #     unit="video",
        #     leave=False,
        #     ncols=80,
        #     ascii=True,
        # ):
        #     extract_frames(video)

        metadata_files = find_metadata_files(args, split)

        annotations = []
        for metadata_file in tqdm(
            metadata_files,
            desc="Processing metadata",
            unit="metadata",
            leave=False,
            ncols=80,
            ascii=True,
        ):
            process_metadata(annotations, metadata_file)

        save_annotations(video_dir, annotations)


def get_video_extension(dataset):
    if dataset == "casia" or dataset == "oulu":
        return ".avi"
    elif dataset == "replay":
        return ".mov"
    else:
        return ""


def find_videos(video_dir, video_ext):
    videos = [str(p) for p in Path(video_dir).rglob(f"*{video_ext}")]
    print(f"Found {len(videos)} videos in {video_dir}")
    return videos


def find_metadata_files(args, split):
    meta_dir = Path(args.data_root) / args.dataset / "meta" / split
    metadata_files = [str(p) for p in Path(meta_dir).rglob("*json")]
    print(f"Found {len(metadata_files)} metadata files in {meta_dir}")
    return metadata_files


def save_annotations(video_dir, annotations):
    dest = video_dir / "annotations.csv"
    logger.info(f"Saving annotations to {dest}")
    print(f"Saving annotations to {dest}")
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(annotations)


def process_metadata(annotations, metadata_file):
    with open(metadata_file) as f:
        metadata = json.load(f)
    for key, value in metadata.items():
        label = 1 if real(metadata_file) else 0
        # frame = f"{Path(metadata_file).with_suffix('')}_{key}.jpg"
        frame = metadata_file.replace(".json", "_") + key + ".jpg"
        frame = str(Path(frame).relative_to(Path(metadata_file).parent.parent))
        print(frame)
        return
        annotations.append(
            [
                frame,
                *value["face_rect"],
                *value["face_landmark"],
                label,
            ]
        )


def extract_frames(video, ext=".jpg"):
    cap = cv2.VideoCapture(str(video))
    vname = str(Path(video).parent / Path(video).stem)  # video name
    if cap.isOpened():
        fr = 0  # metadata keys start from 0
        while True:
            ret, frame = cap.read()
            if ret:
                fn = f"{vname}_{fr}{ext}"
                if not Path(fn).exists():
                    cv2.imwrite(fn, frame)
                fr += 1
            else:
                break

        cap.release()


def real(filename: str) -> bool:
    if "casia" in filename or "oulu" in filename:
        return Path(filename).stem[-1] == "1"
    elif "replay" in filename:
        return "real" in filename
    else:
        logger.error(f"Unknown dataset {filename}")
        return False


if __name__ == "__main__":
    args = parse_args()
    prepare(args)
