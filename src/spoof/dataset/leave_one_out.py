import os
import csv
import glob
import json

root_dir = "data/siwm"
live_train_dir = os.path.join(root_dir, "every5-live_train")
live_test_dir = os.path.join(root_dir, "every5-live_test")
spoof_dir = os.path.join(root_dir, "every5-spoof")


def create_annotations_file(output_file, annotations):
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_path",
                "x",
                "y",
                "width",
                "height",
                "landmark_x1",
                "landmark_y1",
                "landmark_x2",
                "landmark_y2",
                "landmark_x3",
                "landmark_y3",
                "landmark_x4",
                "landmark_y4",
                "landmark_x5",
                "landmark_y5",
                "landmark_x6",
                "landmark_y6",
                "landmark_x7",
                "landmark_y7",
                "class_label",
                "spoof_type",
            ]
        )

        for annotation in annotations:
            writer.writerow(annotation)


def process_annotations(image_paths, annotations, label, spoof_type):
    for image_path in image_paths:
        metadata_file = image_path.replace(".png", ".json")

        with open(metadata_file) as f:
            metadata = json.load(f)
        face_rect = metadata["face_rectangle"]
        face_landmark = metadata["face_landmark"]
        annotation = [
            image_path,
            *face_rect,
            *face_landmark,
            label,
            spoof_type,
        ]
        annotations.append(annotation)


def create_train_annotations():
    train_annotations_file = "data/siwm/train_annotations.csv"
    train_image_paths = glob.glob(
        os.path.join(live_train_dir, "**/*.png"), recursive=True
    )
    train_annotations = []

    process_annotations(train_image_paths, train_annotations, 1, "live")

    spoof_annotations = []
    spoof_subfolders = glob.glob(os.path.join(spoof_dir, "*"))

    for spoof_subfolder in spoof_subfolders:
        spoof_type = os.path.basename(spoof_subfolder)
        spoof_image_paths = glob.glob(
            os.path.join(spoof_subfolder, "**/*.png"), recursive=True
        )
        process_annotations(
            spoof_image_paths, spoof_annotations, 0, spoof_type
        )

    train_annotations += spoof_annotations

    create_annotations_file(train_annotations_file, train_annotations)


def create_test_annotations():
    test_annotations_file = "data/siwm/test_annotations.csv"
    test_image_paths = glob.glob(
        os.path.join(live_test_dir, "**/*.png"), recursive=True
    )
    test_annotations = []

    process_annotations(test_image_paths, test_annotations, 1, "live")

    spoof_annotations = []
    spoof_subfolders = glob.glob(os.path.join(spoof_dir, "*"))

    for spoof_subfolder in spoof_subfolders:
        spoof_type = os.path.basename(spoof_subfolder)
        spoof_image_paths = glob.glob(
            os.path.join(spoof_subfolder, "**/*.png"), recursive=True
        )
        process_annotations(
            spoof_image_paths, spoof_annotations, 0, spoof_type
        )

    test_annotations += spoof_annotations

    create_annotations_file(test_annotations_file, test_annotations)


create_train_annotations()
create_test_annotations()
