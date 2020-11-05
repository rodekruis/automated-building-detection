from setuptools import setup, find_packages


dev_packages = ["jupyterlab", "mypy", "flake8"]


setup(
    name="abd_model",
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").readlines(),
    description="",
    extras_require={
        'dev': dev_packages
    },
    entry_points={"console_scripts": ["abd = abd_model.tools.__main__:main"]},
    long_description_content_type="text/markdown",
)