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
