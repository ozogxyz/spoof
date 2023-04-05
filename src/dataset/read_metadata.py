import json
import os
import sys

import hydra
from omegaconf import DictConfig

# Add parent directory to path for easy import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import filter_files_by_ext, get_metadata_params


def extract_metadata(src):
    print(f"Extracting metadata from {src}")
    # TODO ????????
    with open(src) as f:
        data = json.load(f)

    return data


# This part is for debugging purposes
@hydra.main(version_base="1.2", config_path="../../", config_name="config")
def main(cfg: DictConfig):
    src, dest, ext, meta = get_metadata_params(cfg)

    # videos and metadata are generators
    metadata = filter_files_by_ext(src, meta)

    # apply_map(extract_metadata, metadata)

    next(metadata)
    next(metadata)
    next(metadata)
    next(metadata)
    next(metadata)

    print(extract_metadata(next(metadata)))


if __name__ == "__main__":
    main()
