# ADA data processing tools

Scripts to download/transform pre- and post-disaster images, adapted from 
https://github.com/jmargutt/ADA_tools.

### Major differences

- Data processing scripts moved one level up, directly into `ada_tools` folder
- Added new entrypoints in `setup.py`:
    
    - `load-images`: get images from Maxar
    - `filter-images`: filter images
    - `filter-buildings`: filter buildings
    - `final-layer`: final layer
    - `prepare-data`: transform for damage classification (after building detection)

    Run `<command> --help` to see available arguments.


### Notes on installation
- `GDAL` dependency often causes issues and has to be installed separately;
system requirements need to be installed first and `GDAL` python library version
has to match that of local installation.
See snippet of Dockerfile:

```
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN deps='build-essential cmake gdal-bin python-gdal libgdal-dev kmod wget apache2' && \
	apt-get update && \
	apt-get install -y $deps && \
	pip install --upgrade pip && \
	pip install GDAL==$(gdal-config --version)
```
