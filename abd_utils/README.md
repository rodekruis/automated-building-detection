# automated-building-detection utilities

Scripts to download/transform satellite images.

### Usage
* `download-images`
```
Usage: download-images [OPTIONS]

  download tiled images from Bing Maps API in a given AOI

Options:
  --aoi TEXT      area of interest (vector format)
  --output TEXT   output directory
  --zoom INTEGER  zoom level [default: 17]
  --help          Show this message and exit.
  ```
* `images-to-abd`
```
Usage: images-to-abd [OPTIONS]

  convert tiled images to abd format for building detection

Options:
  --images TEXT  input directory
  --output TEXT  output directory
  --help         Show this message and exit.
  ```
* `filter-buildings`
```
Usage: filter-buildings [OPTIONS]

  merge touching buildings, filter small ones, simplify geometry

Options:
  --data TEXT       input (vector format)
  --dest TEXT       output (vector format)
  --crsmeters TEXT  CRS in unit meters, to filter small buildings [default: EPSG:4087]
  --area INTEGER    minimum building area, in m2 [default: 10]
  --help            Show this message and exit.
  ```


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
