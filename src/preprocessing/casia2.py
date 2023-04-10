from pathlib import Path
import cv2
import hydra
from omegaconf import DictConfig
import torch
from torchvision import transforms, datasets


def capture_frames(src: str, dest: str) -> int:
    """Captures frames of a video."""
    # Open and start to read the video
    cap = cv2.VideoCapture(src)

    if cap.isOpened():
        cur_frame = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"{dest}_frame_{cur_frame}_.jpg"
            cv2.imwrite(filename=filename, img=frame)
            cur_frame += 1

        cap.release()
        cv2.destroyAllWindows()
        return cur_frame

    else:
        print("Cannot open video file")
        return 0


def create_image_folder(
    video_src: str,
    img_folder_root: str,
    video_ext: str = ".avi",
) -> int:
    """Creates an image folder from all the videos in the directory provided. Resulting folder is
    in torchvision ImageFolder format.

    In CASIA, if the video filename ends with 0 it's a spoof, if it ends with
    1 it's real.

    img_folder_root/real/xxx.png
    img_folder_root/real/xxy.jpeg
    img_folder_root/real/xxz.png
    .
    .
    .
    img_folder_root/spoof/123.jpg
    img_folder_root/spoof/nsdf3.png
    img_folder_root/spoof/asd932_.png
    """

    videos = [str(p) for p in Path(video_src).rglob(f"*{video_ext}")]

    # Extract frames
    total_frame_count = 0
    print(f"Found {len(videos)} videos in {video_src}")
    for video in videos:
        print(f"Processing {video}")
        if Path(video).stem[-1] == "0":
            label = "spoof"
        else:
            label = "real"

        # Create the destination folder
        Path(img_folder_root).joinpath(label).mkdir(parents=True, exist_ok=True)

        # Extract frames
        frame_count = capture_frames(
            src=video,
            dest=str(Path(img_folder_root).joinpath(label).joinpath(Path(video).stem)),
        )

        total_frame_count += frame_count
    print(f"Extracted {total_frame_count} frames")
    print(f"Extracted {total_frame_count / len(videos)} frames per video")
    print(f"Extracted {total_frame_count / len(videos) / 30} seconds per video")
    print("Finished extracting frames.")
    return total_frame_count


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    video_src = cfg.data.train_videos
    img_folder_root = cfg.data.train_images
    # create_image_folder(video_src, img_folder_root)

    data_transform = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    data = datasets.ImageFolder(root=img_folder_root, transform=data_transform)

    print(data.classes)


if __name__ == "__main__":
    main()
