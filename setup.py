#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="spoof",
    version="0.1.0",
    author="Orkun Ozoglu",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.15",
        "pandas",
        "scikit-learn",
        "hydra-core>=1.3",
        "tensorboard",
        "pytorch_lightning>=1.5.0, <=1.9.0",
        "opencv-python>=4.5.5.62",
        "pytorch_pretrained_vit",
    ]
)