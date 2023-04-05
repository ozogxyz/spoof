import json
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for easy import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import filter_files_by_ext, get_metadata_params


def read_metadata(src):
    print(f"Extracting metadata from {src}")
    # TODO visualize face rectangles and landmark points from the meta
    with open(src) as f:
        yield json.load(f)


# This part is for debugging purposes
@hydra.main(version_base="1.2", config_path="../../", config_name="config")
def main(cfg: DictConfig):
    src, dest, ext, meta = get_metadata_params(cfg)

    # metadata is a generator yielding jsons
    metadata = filter_files_by_ext(src, meta)
    next(metadata)
    next(metadata)
    next(metadata)
    jsons = read_metadata(next(metadata))

    print(jsons.send(None))


if __name__ == "__main__":
    main()
