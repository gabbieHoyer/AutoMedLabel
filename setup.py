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
    author_email="gabbie.hoyer@ucsf.edu",
    url="https://github.com/gabbieHoyer/AutoMedLabel",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies used throughout project
        "numpy>=1.23.0,<=2.1.1",
        "pandas>=1.1.4",
        "torch>=1.8.0",
        "opencv-python>=4.6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.3.0",
        "scikit-image",
        "scikit-learn",
        "scipy>=1.4.1",
    ],
    extras_require={
        "segmentation": [
            "monai",
            "albumentations",
            "pycocotools",
            "torchvision>=0.9.0",
        ],
        "preprocessing": [
            "cc3d",
            "pyarrow",
            "pydicom",
            "pyrootutils",
        ],
        "notebooks": [
            "IPython",
            "jupyter",
        ],
        "statistics": [
            "pingouin",
            "rich",
            "seaborn>=0.11.0",
            "statsmodels",
        ],
        "object_detection": [
            "clearml",
            "comet_ml",
            "coremltools",
            "py-cpuinfo",
            "dataset",
            "dill",
            "duckdb",
            "dvc",
            "dvclive",
            "mlflow",
            "mss",
            "neptune",
            "nncf",
            "onnx",
            "onnxruntime",
            "onnxsim",
            "openai",
            "openvino",
            "paddle",
            "psutil",
            "ray[tune]",
            "requests>=2.23.0",
            "rtdetr",
            "sentry_sdk",
            "shapely",
            "super_gradients",
            "tensorboard",
            "tensorflow",
            "tensorflowjs",
            "tensorrt",
            "tflite_runtime",
            "tflite_support",
            "tritonclient",
            "ultralytics-thop>=2.0.0",
            "x2paddle",
            "yt_dlp",
        ],
        "sam": [
            "Pillow>=7.1.2",
            "hydra-core",
            "omegaconf",
        ],
    },
    entry_points={
        "console_scripts": [
            "automedlabel=src.main:main",  # creates an executable named 'automedlabel'
        ],
    },
)
