from pathlib import Path


# Helper functions
def get_video_params(cfg):
    src = cfg.dataset.src
    dest = cfg.dataset.dest
    ext = cfg.dataset.ext

    return src, dest, ext


def get_metadata_params(cfg):
    src = cfg.dataset.src
    dest = cfg.dataset.dest
    ext = cfg.dataset.ext
    meta = cfg.dataset.meta

    return src, dest, ext, meta


def filter_files_by_ext(path, ext):
    for path in Path(path).rglob(f"*{ext}"):
        yield str(path)


def apply_map(func, iterable):
    return list(map(func, iterable))
