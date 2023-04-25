#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="spoof",
    version="0.1.0",
    description="Face Spoofing Detection",
    author="Orkun Ozoglu",
    author_email="ozoglu.o@python.edu",
    packages=find_packages(where="spoof"),
    package_dir={"": "spoof"},
    python_requires=">=3.8",
    install_requires=[
        "hydra-core>=1.3",
        "opencv-python>=4.7",
        "omegaconf>=2.3",
        "torch>=2.0",
        "torchvision>=0.15",
        "pytorch_lightning>=1.5.0, <=1.9.0",
        "scikit-learn",
        "timm",
        "pandas",
        "tensorboard",
    ],
    extras_require={
        "dev": [
            "black>=21.0",
            "pre-commit>=2.0",
            "docformatter",
        ],
    },
)
