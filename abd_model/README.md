# automated-building-detection model

Scripts to train and run building detection models. Built on top of [robosat.pink](https://github.com/acannistra/robosat.pink).

To make the package installable, `setup.py` has been adjusted, along with a rearranged file structure.

## Requirements:
### NVIDIA GPU Drivers [mandatory for train and predict]
```bash
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/435.21/NVIDIA-Linux-x86_64-435.21.run
sudo sh NVIDIA-Linux-x86_64-435.21.run -a -q --ui=none
```

## Usage:
Tools:
1. `abd cover` Generate a tiles covering, in csv format: X,Y,Z
1. `abd download` Downloads tiles from a Web Server (XYZ or WMS)
1. `abd extract` Extracts GeoJSON features from OpenStreetMap .pbf
1. `abd rasterize` Rasterize vector features (GeoJSON or PostGIS), to raster tiles
1. `abd subset` Filter images in a slippy map dir using a csv tiles cover
1. `abd tile` Tile a raster coverage
1. `abd train` Train a model on a dataset
1. `abd eval` Evaluate a model on a dataset
1. `abd export` Export a model to ONNX or Torch JIT
1. `abd predict` Predict masks, from a dataset, with an already trained model
1. `abd compare` Compute composite images and/or metrics to compare several slippy map dirs
1. `abd vectorize` Vectorize output: extract GeoJSON features from predicted masks
1. `abd info` Print abd-model version informations

## NOTES:
1. Requires: Python 3.6 or 3.7
1. GPU with VRAM >= 8 GB is mandatory
1. To test abd-model install, launch in a new terminal: `abd info`
1. If needed, to remove pre-existing Nouveau driver: `sudo sh -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf && update-initramfs -u && reboot"`
