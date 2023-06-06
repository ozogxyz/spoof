import csv
import json
import os
from pathlib import Path


def create_annotations(image_dir, metadata_dir):
    annotations = []
    for image_file in image_dir.glob("*.png"):
        metadata_file = metadata_dir / (image_file.stem + ".json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            label = int(image_dir.stem == "real")
            annotation = [
                image_file,
                *metadata["0"]["face_rect"],
                *metadata["0"]["face_landmark"],
                label,
            ]
            annotations.append(annotation)
    return annotations


def prepare_dataset(data_root, splits, classes):
    for split in splits:
        split_dir = Path(data_root) / split
        annotations = []
        for class_ in classes:
            image_dir = split_dir / class_
            metadata_dir = Path(data_root) / "meta" / split / class_
            annotations.extend(create_annotations(image_dir, metadata_dir))

        dest = split_dir / "annotations.csv"
        with open(dest, "w", newline="") as f:
            writer = csv.writer(f)
            headers = [
                "image_file",
                "face_rect_x",
                "face_rect_y",
                "face_rect_width",
                "face_rect_height",
            ]
            headers += [
                f"landmark_{i}" for i in range(1, 15)
            ]  # Add headers for the 14-point face landmark
            headers.append("label")
            headers.append("spoof_type")
            writer.writerow(headers)
            writer.writerows(annotations)


if __name__ == "__main__":
    os.chdir(os.path.join(os.getenv("HOME"), "spoof"))
    data_root = "data/lcc"
    splits = ["train", "test"]
    classes = ["real", "spoof"]
    prepare_dataset(data_root, splits, classes)
