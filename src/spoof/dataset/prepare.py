import csv
import json
from pathlib import Path

import cv2
from tqdm import tqdm
from logging import getLogger
import argparse

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
        choices=["casia", "replay"],
        help="Dataset name",
    )

    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Preparing {args.dataset} dataset")
    if args.dataset == "casia":
        splits = ["train", "val", "test"]
    elif args.dataset == "replay":
        splits = ["train", "devel", "test"]
    for split in splits:
        logger.info(f"Processing {split} split")
        print(f"Processing {split} split")
        video_dir = Path(args.data_root) / args.dataset / split
        if not video_dir.exists():
            logger.error(f"{video_dir} does not exist")
            return
        if args.dataset == "casia":
            video_ext = ".avi"
        elif args.dataset == "replay":
            video_ext = ".mov"
        else:
            logger.error(f"Unknown dataset {args.dataset}")
            return

        videos = [str(p) for p in Path(video_dir).rglob(f"*{video_ext}")]

        logger.info(f"Found {len(videos)} videos in {video_dir}")
        print(f"Found {len(videos)} videos in {video_dir}")

        for video in tqdm(
            videos,
            desc="Extracting frames",
            unit="video",
            leave=False,
            ncols=80,
            ascii=True,
        ):
            vname = video.split(".")[0]
            cap = cv2.VideoCapture(video)
            if cap.isOpened():
                fr = 0  # metadata keys start from 0
                while True:
                    ret, frame = cap.read()
                    if ret == True:
                        fn = vname + "_" + str(fr) + ".jpg"
                        if not Path(fn).exists():
                            cv2.imwrite(fn, frame)
                        fr += 1
                    else:
                        break

                cap.release()

        metadata_files = [str(p) for p in Path(video_dir).rglob("*json")]

        logger.info(
            f"Found {len(metadata_files)} metadata files in {video_dir}"
        )
        print(f"Found {len(metadata_files)} metadata files in {video_dir}")

        annotations = []
        for metadata_file in tqdm(
            metadata_files,
            desc="Processing metadata",
            unit="metadata",
            leave=False,
            ncols=80,
            ascii=True,
        ):
            with open(metadata_file) as f:
                metadata = json.load(f)
            for key in metadata.keys():
                label = 1 if real(metadata_file) else 0
                frame = metadata_file.replace(".json", "_") + key + ".jpg"
                annotations.append(
                    [
                        frame,
                        *metadata[key]["face_rect"],
                        *metadata[key]["face_landmark"],
                        label,
                    ]
                )

        dest = video_dir / "annotations.csv"
        logger.info(f"Saving annotations to {dest}")
        print(f"Saving annotations to {dest}")
        with open(dest, "w") as f:
            writer = csv.writer(f)
            writer.writerows(annotations)


def real(filename: str) -> bool:
    if "casia" in filename:
        return Path(filename).stem[-1] == "1"
    elif "replay" in filename:
        return "real" in filename
    else:
        logger.error(f"Unknown dataset {filename}")
        return False


if __name__ == "__main__":
    args = parse_args()
    prepare(args)
