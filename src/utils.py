from pathlib import Path
from typing import Callable, Iterable

from omegaconf import DictConfig

# Helper functions
def get_video_params(cfg: DictConfig):
    src = cfg.dataset.src
    dest = cfg.dataset.dest
    ext = cfg.dataset.ext

    return src, dest, ext


def get_metadata_params(cfg: DictConfig):
    src = cfg.dataset.src
    dest = cfg.dataset.dest
    ext = cfg.dataset.ext
    meta = cfg.dataset.meta

    return src, dest, ext, meta


def filter_files_by_ext(path: str, ext: str):
    for path in Path(path).rglob(f"*{ext}"):  # type: ignore
        yield str(path)


def apply_map(func: Callable, iterable: Iterable):
    return list(map(func, iterable))
