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
    extracted_frames = [str(p) for p in Path(data_root).rglob("*.jpg")]
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
                    metadata[key]["face_rect"],
                    metadata[key]["face_landmark"],
                    label,
                ]
            )

    with open(save_dir, "w") as f:
        writer = csv.writer(f)
        writer.writerows(annotations)


def extract(
    video_dir: str,
    save_dir: str,
    video_ext: str = ".avi",
) -> int:
    videos = [str(p) for p in Path(video_dir).rglob(f"*{video_ext}")]
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    print(f"Found {len(videos)} videos in {video_dir}")

    for video in tqdm(videos):
        label = "real" if real(video) else "spoof"
        vname = video.split("/")[-1].split(".")[0]
        folder = os.path.join(save_dir, label)
        if not os.path.exists(os.path.join(save_dir, label)):
            os.makedirs(os.path.join(save_dir, label))
        cap = cv2.VideoCapture(video)
        if cap.isOpened():
            fr = 1
            while True:
                ret, frame = cap.read()
                if ret == True:
                    fn = folder + "/" + vname + "_" + str(fr) + ".jpg"
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
    video_dir = "data/casia/val"
    save_dir = os.path.join(video_dir, "images")
    # extract(video_dir, save_dir, video_ext=".mov")

    create_annotations(video_dir, video_dir + "/annotations.csv")
