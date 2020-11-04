# automated-building-detection
Automated building detection tools.

**Scope**: quickly map a large area to support disaster response operations

**Input**: very-high-resolution (<= 0.5 m/pixel) RGB satellite images from [Bing Maps](https://www.bing.com/maps/aerial)

**Output**: buildings in vector format (geojson), to be used in digital map products

Repository is work-in-progress, contact [jmargutt](mailto:jmargutti@redcross.nl) in case of issues. 

Soon to be added: 
* detect buildings on "any" raster image 
* re-train the model on known buildings

## Structure
* `abd_utils` utility functions to download/process satellite images
* `neat_eo` framework to train and run building detection models on images
* `input` input/configuration files needed to run the rest

## Requirements:
To download satellite images:
* [Bing Maps account](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/creating-a-bing-maps-account)
* [Bing Maps Key](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key)

To run the building detection models:
* GPU with VRAM >= 8 GB
* [NVIDIA GPU Drivers](https://www.nvidia.com/Download/index.aspx)

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
How to use these tools? We take as example [a small Dutch town](https://en.wikipedia.org/wiki/Giethoorn); to predict the buildings in another area, simply change the input AOI (you can create your own using e.g. [geojson.io](http://geojson.io/)).

Detailed explanation on usage and parameters of the different commands is given in the subdirectories `abd_utils` and `neat_eo`.

2. Add you Bing Maps Key in `abd_utils/src/abd_utils/.env` (the Docker container has [vim](https://www.vim.org/) pre-installed)
3. Download the images of the AOI, divided in [tiles](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
```
download-images --aoi input/AOI.geojson --output images
```
3. Convert the images into the format needed to run the building detection model
```
images-to-neo --images images --output neo-images
```
3. [Download a pre-trained model](https://rodekruis.sharepoint.com/sites/510-Team/_layouts/15/guestaccess.aspx?docid=048f1927be4af4bc09805be0cfc376b22&authkey=AZSnVN8hrbj9CYSV8K-wg9o&expiration=2021-08-08T22%3A00%3A00.000Z&e=VIywGA) and add it to the `input` directory
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

