# automated-building-detection
Automated building detection tools.

## Structure
1. `abd_utils` utility functions to download/process satellite images
2. `neat_eo` framework to train and run building detection models on images
3. `input` input/configuration files needed to run the rest
## Requirements:
1. [Bing Maps account](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/creating-a-bing-maps-account)
2. [Bing Maps Key](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key)
3. GPU with VRAM >= 8 GB
4. [NVIDIA GPU Drivers]

## Getting started
### Using Docker
1. Install [Docker](https://www.docker.com/get-started).
2. Download the [latest Docker Image](https://hub.docker.com/r/jmargutti/automated-building-detection)
```
docker pull jmargutti/automated-building-detection
```
3. Create a docker container and connect it to a local directory (`<path-to-your-workspace>`)
```
docker run --name automated-building-detection -dit -v <path-to-your-workspace>:/ -p 5000:5000 jmargutti/automated-building-detection
```
4. Access the container
```
docker exec -it automated-building-detection bash
```

### Manual Setup
TBI

## End-to-end example
This section explains how to predict buildings in a given area of interest (AOI). It uses as example [a small Dutch town](https://en.wikipedia.org/wiki/Giethoorn); to predict the buildings in another area, simply change the input AOI (you can create your own using e.g. [geojson.io](http://geojson.io/).
Detailed explanation on the usage of the different commands is given in the subdirectories `abd_utils` and `neat_eo`.

2. Add you Bing Maps Key in `abd_utils/src/abd_utils/.env` (the Docker container comes with [vim](https://www.vim.org/) pre-installed)
3. Download the images of the AOI, divided in [tiles](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
```
download-images --aoi input/AOI.geojson --output images
```
3. Convert the images into the format needed to run the building detection model
```
images-to-neo --images images --output neo-images
```
3. Run the building detection model 
```
neo predict --config input/config.toml --dataset neo-images --cover neo-images/cover.csv --checkpoint input/neat-fullxview-epoch75.pth --out neo-predictions --metatiles --keep_borders
```
3. Vectorize model output (from pixels to polygons)
```
neo vectorize --config input/config.toml --type Building --masks neo-predictions --out neo-predictions/buildings.geojson
```
3. Merge touching polygons, remove small artifacts, simplify geometry
```
filter-buildings --data neo-predictions/buildings.geojson --dest neo-predictions/buildings-clean.geojson
```

