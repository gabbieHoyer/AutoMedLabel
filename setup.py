# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything
# Adapted from https://github.com/facebookresearch/sam2

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Adapted from https://github.com/ultralytics/ultralytics


from setuptools import setup, find_packages

setup(
    name="AutoMedLabel",
    version="0.1.0",
    description="A framework for evaluating foundation models in musculoskeletal MRI",
    author="Gabrielle Hoyer",
    author_email="youremail@example.com",
    url="https://github.com/gabbieHoyer/AutoMedLabel",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your dependencies here, or rely on requirements.txt if you prefer.
        "pyrootutils",
        "numpy",
        "torch",
        # etc.
    ],
    entry_points={
        "console_scripts": [
            "automedlabel=src.main:main",  # This creates an executable named 'automedlabel'
        ],
    },
)
