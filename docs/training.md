This document is an adaptation of the [RoboSat.pink documentation](https://github.com/acannistra/robosat.pink): [From OpenData to OpenDataSet](https://github.com/acannistra/robosat.pink/blob/master/docs/from_opendata_to_opendataset.md).


Context:
-------
In this document we detail the commands to use in order to train a model from scratch, starting from the download of an open dataset and ending with a fully functional predictive model.

Retrieve OpenData:
------------------

We choose to use OpenData from <a href="https://rdata-grandlyon.readthedocs.io/en/latest/">Grand Lyon metropole</a> because they provide recent imagery and several vector layers throught standardized Web Services.

Let's first create a working directory where to do all the following instructions.
```
mkdir ~/abd_dataset
cd ~/abd_dataset
```

First step is to define the coverage geospatial extent and a <a href="https://wiki.openstreetmap.org/wiki/Zoom_levels">zoom level</a>:

```
abd cover --zoom 18 --bbox 4.795,45.628,4.935,45.853 --out  cover
```


Then to download imagery, throught <a href="https://www.opengeospatial.org/standards/wms">WMS</a>:

```
abd download --type WMS --url 'https://download.data.grandlyon.com/wms/grandlyon?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=Ortho2015_vue_ensemble_16cm_CC46&WIDTH=512&HEIGHT=512&CRS=EPSG:3857&BBOX={xmin},{ymin},{xmax},{ymax}&FORMAT=image/jpeg' --format jpeg --cover cover --out images
```

NOTE:
- Retina resolution of 512px is prefered to a regular 256px, because it will improve the training accuracy result. 
- Launch this command again, if any tile download error, till the whole coverage is fully downloaded.
- Jpeg is prefered over default webp only because some browser still not handle webp format

[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/images/"><img src="img/from_opendata_to_opendataset/images.png" /></a>)

Then to download buildings vector roof print, throught <a href="https://www.opengeospatial.org/standards/wfs">WFS</a>, 

```
wget -O ~/rsp_dataset/lyon_roofprint.json 'https://download.data.grandlyon.com/wfs/grandlyon?SERVICE=WFS&REQUEST=GetFeature&TYPENAME=ms:fpc_fond_plan_communaut.fpctoit&VERSION=1.1.0&srsName=EPSG:4326&outputFormat=application/json; subtype=geojson'
```

Roofprint choice is meaningful here, as we use aerial imagery to retrieve patterns. If we used building's footprints instead, our later training accuracy performances would be poorer.


Prepare DataSet
----------------

Now to transform the vector roofprints, to raster labels:

```
abd rasterize --cover cover --config config.toml --type Building --geojson lyon_roofprint.json  --out labels
```

[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/labels/"><img src="img/from_opendata_to_opendataset/labels.png" /></a>)

To also add the touching borders between buildings to the rastered mask:
```
abd rasterize --cover cover --config config.toml --type Border --geojson lyon_roofprint.json  --out labels --borders --append
```

NOTE:
- the touching borders are needed if you want to identify such features with your model. In that case, remember to add the Borders class to the input config.toml file


Then to create a training / validation dataset, with imagery and related roofprint labels:

```
mkdir training validation   
                                                                                   
cat cover | sort -R >  cover.shuffled
head -n 16384 cover.shuffled > training/cover
tail -n 7924  cover.shuffled > validation/cover

abd subset --dir images --cover training/cover --out training/images
abd subset --dir labels --cover training/cover --out training/labels
abd subset --dir images --cover validation/cover --out validation/images
abd subset --dir labels --cover validation/cover --out validation/labels
```

Two points to emphasize there:
 - It's a good idea to take enough data for the validation part (here we took a 70/30 ratio).
 - The shuffle step help to reduce spatial bias in train/validation sets.


Train
-----

Now to launch a first model train:

```
abd train --config config.toml --dataset training/ --out pth --epochs 75
```

NOTE:
- If using the model with the touching borders added class, it could be a good idea to add the option `--classes_weights auto`, which computes weighting coefficient for the classes based on the class distribution in the training set.
 
[//]: # (After only 10 epochs, the building IoU metric on validation dataset, is about **0.82**. )
It's already a good result, at the state of art, with real world data, but we will see how to increase it.


Predict masks
-------------

To create predict masks from our first model, on the whole coverage:

```
abd predict --config config.toml --checkpoint pth/checkpoint-00075.pth --dataset ./ --out predicted_masks/ --cover cover --metatiles --keep_borders
```

[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/masks/"><img src="img/from_opendata_to_opendataset/masks.png" /></a>)

[//]: # ()
[//]: # (Compare)

[//]: # (-------)

[//]: # ()
[//]: # (Then to compare how our first model reacts with this raw data, we compute a composite stack image, with imagery, label and predicted mask.)

[//]: # ()
[//]: # (Color representation meaning is:)

[//]: # ( - pink: predicted by the model &#40;but not present in the initial labels&#41;)

[//]: # ( - green: present in the labels &#40;but not predicted by the model&#41;)

[//]: # ( - grey: both model prediction and labels are synchronized.)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (```)

[//]: # (rsp compare --images ~/rsp_dataset/images ~/rsp_dataset/labels ~/rsp_dataset/masks --mode stack --labels ~/rsp_dataset/labels --masks ~/rsp_dataset/masks --config config.toml --ext jpeg --web_ui ~/rsp_dataset/compare)

[//]: # ()
[//]: # (rsp compare --mode list --labels ~/rsp_dataset/labels --maximum_qod 80 --minimum_fg 5 --masks ~/rsp_dataset/masks --config config.toml --geojson ~/rsp_dataset/compare/tiles.json)

[//]: # (```)

[//]: # ()
[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare/"><img src="img/from_opendata_to_opendataset/compare.png" /></a>)

[//]: # ()
[//]: # (We launch also a csv list diff, to only keep tiles with a low Quality of Data metric &#40;here below 80% on QoD metric as a threshold&#41;, and with at least few buildings pixels supposed to be present in the tile &#40;5% of foreground building as a threshold&#41;.)

[//]: # ()
[//]: # (And if we zoom back on the map, we could see tiles matching the previous filters:)

[//]: # ()
[//]: # ()
[//]: # (<img src="img/from_opendata_to_opendataset/compare_zoom_out.png" />)

[//]: # ()
[//]: # ()
[//]: # (And it becomes clear that some area are not well labelled in the original OpenData.)

[//]: # (So we would have to remove them from the training and validation dataset.)

[//]: # ()
[//]: # (To do so, first step is to select the wrong labelled ones, and the compare tool again is helpfull,)

[//]: # (as it allow to check side by side several tiles directory, and to manual select thoses we want.)

[//]: # ()
[//]: # (```)

[//]: # (rsp compare --mode side --images ~/rsp_dataset/images ~/rsp_dataset/compare --labels ~/rsp_dataset/labels --maximum_qod 80 --minimum_fg 5 --masks ~/rsp_dataset/masks --config config.toml --ext jpeg --web_ui ~/rsp_dataset/compare_side)

[//]: # (```)

[//]: # ()
[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_side/"><img src="img/from_opendata_to_opendataset/compare_side.png" /></a>)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (Filter)

[//]: # (------)

[//]: # ()
[//]: # (The result from the compare selection produce a csv cover list, in the clipboard.)

[//]: # (We put the result in `~rsp_dataset/cover.to_remove`)

[//]: # ()
[//]: # (Then we just remove all theses tiles from the dataset:)

[//]: # (```)

[//]: # (rsp subset --mode delete --dir ~/rsp_dataset/training/images --cover ~/rsp_dataset/cover.to_remove > /dev/null)

[//]: # (rsp subset --mode delete --dir ~/rsp_dataset/training/labels --cover ~/rsp_dataset/cover.to_remove > /dev/null)

[//]: # (rsp subset --mode delete --dir ~/rsp_dataset/validation/images --cover ~/rsp_dataset/cover.to_remove > /dev/null)

[//]: # (rsp subset --mode delete --dir ~/rsp_dataset/validation/labels --cover ~/rsp_dataset/cover.to_remove > /dev/null)

[//]: # (```)

[//]: # ()
[//]: # (For information, we remove about 500 tiles from this raw dataset, in order to clean it up, from obvious inconsistency labelling.)

[//]: # ()
[//]: # ()
[//]: # (Train )

[//]: # (-----)

[//]: # ()
[//]: # (Then with a cleanest training and validation dataset, we can launch a new, and longer, training:)

[//]: # ()
[//]: # (```)

[//]: # (rsp train --config config.toml --epochs 100 ~/rsp_dataset/pth_clean)

[//]: # (```)

[//]: # ()
[//]: # (Building IoU metrics on validation dataset:)

[//]: # ( - After 10  epochs: **0.84** )

[//]: # ( - After 100 epochs: **0.87**)

[//]: # ( )
[//]: # ( )
[//]: # ()
[//]: # (Predict and compare)

[//]: # (-------------------)

[//]: # ()
[//]: # (And now to generate masks prediction, and compare composite images, as previously:)

[//]: # ()
[//]: # (```)

[//]: # (rsp predict --config config.toml --checkpoint ~/rsp_dataset/pth_clean/checkpoint-00100-of-00100.pth ~/rsp_dataset/images ~/rsp_dataset/masks_clean)

[//]: # ()
[//]: # (rsp compare --images ~/rsp_dataset/images ~/rsp_dataset/labels ~/rsp_dataset/masks_clean --mode stack --labels ~/rsp_dataset/labels --masks ~/rsp_dataset/masks_clean --config config.toml --web_ui --ext jpeg ~/rsp_dataset/compare_clean)

[//]: # ()
[//]: # (rsp compare --mode list --labels ~/rsp_dataset/labels --maximum_qod 80 --minimum_fg 5 --masks ~/rsp_dataset/masks_clean --config config.toml --geojson ~/rsp_dataset/compare_clean/tiles.json)

[//]: # (```)

[//]: # ()
[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_clean/"><img src="img/from_opendata_to_opendataset/compare_clean.png" /></a>)

[//]: # ()
[//]: # ()
[//]: # (And to compare only with filtered validation tiles, in side by side mode: )

[//]: # ()
[//]: # (```)

[//]: # (rsp cover --type dir ~/rsp_dataset/validation/images  ~/rsp_dataset/validation/cover.clean)

[//]: # ()
[//]: # (rsp subset --dir ~/rsp_dataset/compare_clean --cover ~/rsp_dataset/validation/cover.clean --out ~/rsp_dataset/validation/compare_clean)

[//]: # ()
[//]: # (rsp subset --dir ~/rsp_dataset/masks_clean --cover ~/rsp_dataset/validation/cover.clean --out ~/rsp_dataset/validation/masks_clean)

[//]: # ()
[//]: # (rsp compare --mode side --images ~/rsp_dataset/validation/images ~/rsp_dataset/validation/compare_clean --labels ~/rsp_dataset/validation/labels --masks ~/rsp_dataset/validation/masks_clean --config config.toml --web_ui --ext jpeg ~/rsp_dataset/validation/compare_side_clean)

[//]: # (```)

[//]: # ()
[//]: # (<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_side_clean/"><img src="img/from_opendata_to_opendataset/compare_side_clean.png" /></a>)