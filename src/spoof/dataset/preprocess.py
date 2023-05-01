import csv
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm


# This function is common for all datasets so it's placed in utils
def create_annotations(
    data_root: str,
    save_dir: str,
):
    # Get all the extracted frames and metadata files
    metadata_files = [str(p) for p in Path(data_root).rglob("*json")]

    # get the keys from the metadata files and find its corresponding frame
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

    with open(save_dir, "w") as f:
        writer = csv.writer(f)
        writer.writerows(annotations)


def extract(
    video_dir: str,
    video_ext: str = ".avi",
) -> int:
    videos = [str(p) for p in Path(video_dir).rglob(f"*{video_ext}")]

    print(f"Found {len(videos)} videos in {video_dir}")

    for video in tqdm(videos):
        label = "real" if real(video) else "spoof"
        vname = video.split(".")[0]

        cap = cv2.VideoCapture(video)
        if cap.isOpened():
            fr = 1
            while True:
                ret, frame = cap.read()
                if ret == True:
                    fn = vname + "_" + str(fr) + ".jpg"
                    if os.path.exists(fn):
                        continue
                    cv2.imwrite(fn, frame)
                    fr += 1
                else:
                    break

            cap.release()


def real(filename: str) -> bool:
    if "casia" in filename:
        return Path(filename).stem[-1] == "1"
    elif "replay" in filename:
        return "real" in filename


if __name__ == "__main__":
    video_dir = "data/replay/train"
    extract(video_dir, video_ext=".mov")
    create_annotations(video_dir, video_dir + "/annotations.csv")

    video_dir = "data/replay/val"
    extract(video_dir, video_ext=".mov")
    create_annotations(video_dir, video_dir + "/annotations.csv")

    video_dir = "data/replay/test"
    extract(video_dir, video_ext=".mov")
    create_annotations(video_dir, video_dir + "/annotations.csv")

    video_dir = "data/casia/train"
    extract(video_dir, video_ext=".avi")
    create_annotations(video_dir, video_dir + "/annotations.csv")

    video_dir = "data/casia/val"
    extract(video_dir, video_ext=".avi")
    create_annotations(video_dir, video_dir + "/annotations.csv")

    video_dir = "data/casia/test"
    extract(video_dir, video_ext=".avi")
    create_annotations(video_dir, video_dir + "/annotations.csv")
