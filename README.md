# automated-building-detection
Automated building detection tools

## Requirements:
1. [Get a Bing Maps account](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/creating-a-bing-maps-account)
2. [Create a Bing Maps Key](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key)

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

### End-to-end example
1. The area of interest (AOI) is defined in `input/AOI.geojson`; you can create your own using [geojson.io](http://geojson.io/)
2. Add you Bing Maps Key in `abd_utils/src/abd_utils/.env` (the Docker container comes with [vim](https://www.vim.org/) pre-installed)
3. Download the images of the AOI, divided in [tiles](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
```
download-images --aoi input/AOI.geojson --output images
```
3. Convert the images in the format needed to run the building detection model
```
images-to-neo --images images --output neo-images
```
3. Convert the images in the format needed to run the building detection model
```
neo predict --config input/config.toml --dataset neo-images --cover neo-images/cover.csv --checkpoint input/neat-fullxview-epoch75.pth --out neo-predictions --metatiles --keep_borders
```


