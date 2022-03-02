This documents shows an end to end example of using the abd model to identify buildings from satellite images, improved with the ability to separate adjacent buildings in crowded areas.

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
4. [Download a pre-trained model](https://rodekruis.sharepoint.com/sites/510-Team/_layouts/15/guestaccess.aspx?docid=0f686f33162d34f6a8e810b9d8f43e3fa&authkey=ATiMPAT4k1TbJcMOBY-IbhM&expiration=2023-01-25T00%3A00%3A00.000Z&e=U3AZ9C) (more details below) and add it to the `input` directory
5. Run the building detection model 
```
abd predict --config input/config.toml --dataset abd-input --cover abd-input/cover.csv --checkpoint input/neat-fullxview-epoch75.pth --out abd-predictions --metatiles --keep_borders
```
6. Separate touching buildings instances in crowded areas
```
abd separate --masks abd_predictions --config config.toml --out abd-predictions-separated
```
7. Vectorize model output (from pixels to polygons)
```
abd vectorize --config input/config.toml --type Building --masks abd-predictions-separated --out abd-predictions/buildings-separated.geojson
```
8. represent separated buildings with squares/circles around the building centroids, with areas proportional to the polygon area. The building centers will be overlapped with the original model output in the final visualization.
```
filter-buildings --data abd-predictions/buildings-separated.geojson --dest abd-predictions/buildings-separated-markers.geojson --building_markers y --marker square```

## Model collection
* [neat-fullxview-epoch100](https://rodekruis.sharepoint.com/sites/510-Team/_layouts/15/guestaccess.aspx?docid=0f686f33162d34f6a8e810b9d8f43e3fa&authkey=ATiMPAT4k1TbJcMOBY-IbhM&expiration=2023-01-25T00%3A00%3A00.000Z&e=U3AZ9C): 
  * architecture: AlbuNet ([U-Net-like](https://arxiv.org/abs/1505.04597) encoder-decoder with a ResNet, ResNext or WideResNet encoder)
  * training: WMS Lyon dataset, 100 epochs, see training docs for how to download it
  * performance: Building/Border - [IoU](https://en.wikipedia.org/wiki/Jaccard_index) 0.74/0.38, [MCC](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) 0.81/0.18

