import csv
import json
import os
from pathlib import Path


def prepare_lcc(data_root):
    splits = ["train", "test"]
    classes = ["real", "spoof"]

    for split in splits:
        split_dir = Path(data_root) / split
        image_dirs = [split_dir / class_ for class_ in classes]
        metadata_dir = Path(data_root) / "meta" / split

        annotations = []

        for image_dir, class_ in zip(image_dirs, classes):
            for image_file in image_dir.glob("*.png"):
                metadata_file = (
                    metadata_dir / class_ / (image_file.stem + ".json")
                )
                if not metadata_file.exists():
                    continue

                with open(metadata_file) as f:
                    metadata = json.load(f)

                label = 1 if class_ == "real" else 0

                annotations.append(
                    [
                        str(image_file.relative_to(split_dir)),
                        *metadata["0"]["face_rect"],
                        *metadata["0"]["face_landmark"],
                        label,
                    ]
                )

        dest = split_dir / "annotations.csv"
        with open(dest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(annotations)


if __name__ == "__main__":
    os.chdir(os.path.join(os.getenv("HOME"), "spoof"))
    data_root = Path("data/lcc")
    prepare_lcc(data_root)
