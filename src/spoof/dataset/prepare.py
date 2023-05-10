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
    parser.add_argument("-v", "--video_dir", type=str, default="data")
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
    for split in ["train", "val", "test"]:
        logger.info(f"Processing {split} split")
        video_dir = Path(args.video_dir) / args.dataset / split
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

        metadata_files = [str(p) for p in Path(args.video_dir).rglob("*json")]

        logger.info(
            f"Found {len(metadata_files)} files in {metadata_files[0]}"
        )

        annotations = []
        for metadata_file in tqdm(metadata_files):
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

        dest = Path(args.video_dir) / args.dataset / split / "annotations.csv"
        logger.info(f"Saving annotations to {dest}")
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
