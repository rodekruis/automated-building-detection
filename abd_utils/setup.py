"""
Setup.py file.
Install once-off with:  "pip install ."
For development:        "pip install -e .[dev]"
"""
import setuptools


with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

PROJECT_NAME = "abd_utils"

setuptools.setup(
    name=PROJECT_NAME,
    version="0.1",
    description="Satellite image preprocessing utilities",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require={
        "dev": [  # Place NON-production dependencies in this list - so for DEVELOPMENT ONLY!
            "black",
            "flake8"
        ],
    },
    entry_points={
        'console_scripts': [
            f"download-images = {PROJECT_NAME}.download_images:main",
            f"filter-buildings = {PROJECT_NAME}.filter_buildings:main",
            f"images-to-abd = {PROJECT_NAME}.images_to_abd:main",
            f"add-osm-data = {PROJECT_NAME}.add_osm_data:main"
        ]
    }
)
