[![version: 0.0.0](https://img.shields.io/badge/stable-0.0.0-blue?style=flat-square)](https://github.com/rodekruis/automated-building-detection)
[![style: PEP 8](https://img.shields.io/badge/style-PEP8-red?style=flat-square)](https://www.python.org/dev/peps/pep-0008/)

# automated-building-detection
Automated Building Detection using Deep Learning: a NLRC/510 tool

**Scope**: quickly map a large area to support disaster response operations

**Input**: very-high-resolution (<= 0.5 m/pixel) RGB satellite images. Currently supported:
* [Bing Maps](https://www.bing.com/maps/aerial)
* Any custom image in raster format

**Output**: buildings in vector format (geojson), to be used in digital map products

## Credits
Built on top of [robosat](https://github.com/mapbox/robosat) and [robosat.pink](https://github.com/acannistra/robosat.pink).

Development: [Ondrej Zacha](https://github.com/ondrejzacha), [Wessel de Jong](https://github.com/Wessel93), [Jacopo Margutti](https://github.com/jmargutt)

Contact: [Jacopo Margutti](mailto:jmargutti@redcross.nl).

## Structure
* `abd_utils` utility functions to download/process satellite images
* `abd_model` framework to train and run building detection models on images
* `input` input/configuration files needed to run the rest

## Requirements:
To download satellite images:
* [Bing Maps account](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/creating-a-bing-maps-account)
* [Bing Maps Key](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key)

To run the building detection models:
* GPU with VRAM >= 8 GB
* [NVIDIA GPU Drivers](https://www.nvidia.com/Download/index.aspx) and [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

If using Docker
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), to expose the GPUs to the docker container

## Getting started
### Using Docker
1. Install [Docker](https://www.docker.com/get-started).
2. Download the [latest Docker Image](https://hub.docker.com/r/rodekruis/automated-building-detection)
```
docker pull rodekruis/automated-building-detection
```
3. Create a docker container and connect it to a local directory (`<path-to-your-workspace>`)
```
docker run --name automated-building-detection -dit -v <path-to-your-workspace>:/workdir --ipc=host --gpus all -p 5000:5000 rodekruis/automated-building-detection
```
4. Access the container
```
docker exec -it automated-building-detection bash
```

### Manual Setup
1. Install Python 3.7 and [pip](https://pypi.org/project/pip/)
2. Install [Anaconda](https://www.anaconda.com/products/individual)
3. Create and activate a new Anaconda environment
```
conda create --name abdenv python=3.7 
conda activate abdenv
```
4. From root directory, move to `abd_utils` and install
```
cd abd_utils
pip install .
```
5. Move to `abd_model` and install
```
cd ../abd_model
pip install .
```
N.B. Remember to activate `abdenv` next time

## End-to-end example
How to use these tools? We take as example [a small Dutch town](https://en.wikipedia.org/wiki/Giethoorn); to predict the buildings in another area, simply change the input AOI (you can create your own using e.g. [geojson.io](http://geojson.io/)).

Detailed explanation on usage and parameters of the different commands is given in the subdirectories `abd_utils` and `abd_model`.

1. Add you Bing Maps Key in `abd_utils/src/abd_utils/.env` (the Docker container has [vim](https://www.vim.org/) pre-installed)
2. Download the images of the AOI, divided in [tiles](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
```
download-images --aoi input/AOI.geojson --output bing-images
```
3. Convert the images into the format needed to run the building detection model
```
images-to-abd --images bing-images/images --output abd-input
```
4. [Download a pre-trained model](https://drive.google.com/file/d/1pMkrBjdpmOgT_MzqZSLHvmQDsZNM_Lwo/view?usp=sharing) (more details below) and add it to the `input` directory
5. Run the building detection model 
```
abd predict --config input/config.toml --dataset abd-input --cover abd-input/cover.csv --checkpoint input/neat-fullxview-epoch75.pth --out abd-predictions --metatiles --keep_borders
```
6. Vectorize model output (from pixels to polygons)
```
abd vectorize --config input/config.toml --type Building --masks abd-predictions --out abd-predictions/buildings.geojson
```
7. Merge touching polygons, remove small artifacts, simplify geometry
```
filter-buildings --data abd-predictions/buildings.geojson --dest abd-predictions/buildings-clean.geojson
```

## Model collection
* [neat-fullxview-epoch75](https://rodekruis.sharepoint.com/sites/510-Team/_layouts/15/guestaccess.aspx?docid=048f1927be4af4bc09805be0cfc376b22&authkey=AZSnVN8hrbj9CYSV8K-wg9o&expiration=2021-08-08T22%3A00%3A00.000Z&e=VIywGA): 
  * architecture: AlbuNet ([U-Net-like](https://arxiv.org/abs/1505.04597) encoder-decoder with a ResNet, ResNext or WideResNet encoder)
  * training: [xBD dataset](https://arxiv.org/pdf/1911.09296.pdf), 75 epochs
  * performance: [IoU](https://en.wikipedia.org/wiki/Jaccard_index) 0.79, [MCC](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) 0.75

