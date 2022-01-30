import os
import sys
from tqdm import tqdm

import numpy as np
from PIL import Image

from skimage.segmentation import watershed
import scipy.ndimage as ndimage

from abd_model.core import load_config, make_palette
from abd_model.tiles import tiles_from_dir, tile_label_to_file
from abd_model.borders import augment_borders


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("separate", help="Process predicted masks to separate bordering building instances",
                                  formatter_class=formatter_class)

    inp = parser.add_argument_group("Inputs")
    inp.add_argument("--masks", type=str, required=True, help="input masks directory path [required]")
    inp.add_argument("--config", type=str, help="path to config file [required, if no global config setting]")

    out = parser.add_argument_group("Outputs")
    out.add_argument("--out", type=str, required=True, help="path to output file to store features in [required]")

    parser.set_defaults(func=main)

def label_watershed(before, after, component_size=1):
    markers = ndimage.label(after)[0]

    labels = watershed(-before, markers, mask=before, connectivity=8,watershed_line=True)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0
    return labels

def main(args):
    config = load_config(args.config)

    palette, transparency = make_palette([classe["color"] for classe in config["classes"]], complementary=True)

    masks = list(tiles_from_dir(args.masks, xyz_path=True))
    assert len(masks), "empty masks directory: {}".format(args.masks)

    print("abd separate_instances from {}".format(args.masks), file=sys.stderr, flush=True)

    if os.path.dirname(os.path.expanduser(args.out)):
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)

    for tile, path in tqdm(masks, ascii=True, unit="mask"):
        predicted_mask = np.array(Image.open(path).convert("P"), dtype=np.uint8)

        mask = (predicted_mask == 1).astype(np.uint8)
        contour = (predicted_mask > 1).astype(np.uint8)
        # contour = augment_borders(contour)

        seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)
        labels = label_watershed(mask, seed, component_size=1).astype(bool).astype(np.uint8)

        tile_label_to_file(args.out, tile, palette, transparency, labels)
