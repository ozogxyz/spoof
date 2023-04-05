import json
from functools import partial

import hydra
from omegaconf import DictConfig

from utils import filter_files_by_ext, get_metadata_params


def extract_metadata(src):
    print(f"Extracting metadata from {src}")
    with open(src) as f:
        data = json.load(f)

    return data


@hydra.main(version_base="1.2", config_path="../../", config_name="config")
def main(cfg: DictConfig):
    src, dest, ext, meta = get_metadata_params(cfg)

    # videos and metadata are generators
    metadata = filter_files_by_ext(src, meta)

    casia_metadata = partial(extract_metadata, dest=dest)

    # apply_map(extract_metadata, metadata)

    next(metadata)
    next(metadata)
    next(metadata)
    next(metadata)
    next(metadata)

    d = extract_metadata(next(metadata))
    print(d)


if __name__ == "__main__":
    main()
