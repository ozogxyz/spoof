#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="SpoofingDetection",
    version="1.0",
    description="Face Spoofing Detection",
    author="Orkun Ozoglu",
    author_email="ozoglu.o@python.edu",
    packages=find_packages(where="spoof"),
    package_dir={"": "spoof"},
    python_requires=">=3.10",
    install_requires=[
        "hydra-core==1.3.2",
        "opencv-python==4.7.0.72,>=4.7.0",
        "omegaconf==2.3.0",
        "torch==2.0.0",
        "torchvision==0.15.1",
    ],
    extras_require={
        "dev": [
            "black==21.5b2",
            "pre-commit==2.13.0",
            "docformatter",
            "pre-commit",
        ],
    },
)
