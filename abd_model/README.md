# neat-EO

Computer Vision framework for GeoSpatial imagery.
Unofficial fork of [neat-eo](https://neat-eo.pink/), built on top of [robosat.pink](https://github.com/dselivanov/robosat.pink).

To make the package installable, `setup.py` has been adjusted, along with a rearranged file structure.

*Note*: this package is referenced as both "neat-eo" and "neo" inside this project.

## Readme from the original project:

## Documentation:
Tools:
1. `neo cover` Generate a tiles covering, in csv format: X,Y,Z
1. `neo download` Downloads tiles from a Web Server (XYZ or WMS)
1. `neo extract` Extracts GeoJSON features from OpenStreetMap .pbf
1. `neo rasterize` Rasterize vector features (GeoJSON or PostGIS), to raster tiles
1. `neo subset` Filter images in a slippy map dir using a csv tiles cover
1. `neo tile` Tile a raster coverage
1. `neo train` Train a model on a dataset
1. `neo eval` Evaluate a model on a dataset
1. `neo export` Export a model to ONNX or Torch JIT
1. `neo predict` Predict masks, from a dataset, with an already trained model
1. `neo compare` Compute composite images and/or metrics to compare several slippy map dirs
1. `neo vectorize` Vectorize output: extract GeoJSON features from predicted masks
1. `neo info` Print neat-EO version informations

## Requirements:
### NVIDIA GPU Drivers [mandatory for train and predict]
```bash
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/435.21/NVIDIA-Linux-x86_64-435.21.run
sudo sh NVIDIA-Linux-x86_64-435.21.run -a -q --ui=none
```

### HTTP Server [for WebUI rendering]
```bash
sudo apt install -y apache2 && sudo ln -s ~ /var/www/html/neo
```

## NOTES:
1. Requires: Python 3.6 or 3.7
1. GPU with VRAM >= 8 GB is mandatory
1. To test neat-EO install, launch in a new terminal: `neo info`
1. If needed, to remove pre-existing Nouveau driver: `sudo sh -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf && update-initramfs -u && reboot"`
