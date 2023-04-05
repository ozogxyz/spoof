import functools
import os
import cv2
import hydra

from pathlib import Path

from omegaconf import DictConfig


# Function to extract frames
def capture_frames(src, dest):
    """Captures video frames from root src and saves them to dest folder."""
    print(f"Capturing frames from {src} to {dest}")
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        cur_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(dest, "frame%d.jpg" % cur_frame), frame)
            cur_frame += 1
        cap.release()
    else:
        print("Cannot open video file")
    cv2.destroyAllWindows()


def filter_files_by_ext(path, ext):
    for path in Path(path).rglob(f"*{ext}"):
        yield str(path)


@hydra.main(version_base="1.2", config_path="../../", config_name="config")
def main(cfg: DictConfig):
    src = cfg.dataset.src
    dest = cfg.dataset.dest
    ext = cfg.dataset.ext
    meta = cfg.dataset.meta

    videos = filter_files_by_ext(src, ext)
    meta = filter_files_by_ext(src, meta)

    capture_casia = functools.partial(capture_frames, dest=dest)
    casia_frames = list(map(capture_casia, videos))


if __name__ == "__main__":
    main()
