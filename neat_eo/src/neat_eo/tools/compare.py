import os
import sys
import json
import torch
import concurrent.futures as futures

from PIL import Image
from tqdm import tqdm
import numpy as np

from mercantile import feature

from neat_eo.core import web_ui, Logs, load_module, load_config
from neat_eo.tiles import tiles_from_dir, tile_from_xyz, tiles_from_csv, tile_image_from_file, tile_image_to_file


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "compare",
        help="Compute composite images and/or metrics to compare several XYZ dirs",
        formatter_class=formatter_class,
    )

    inp = parser.add_argument_group("Inputs")
    choices = ["side", "stack", "list"]
    inp.add_argument("--mode", type=str, default="side", choices=choices, help="compare mode [default: side]")
    inp.add_argument("--labels", type=str, help="path to tiles labels directory [required for metrics filtering]")
    inp.add_argument("--masks", type=str, help="path to tiles masks directory [required for metrics filtering)")
    help = "path to config file [required for metrics filtering, if no global config setting]"
    inp.add_argument("--config", type=str, help=help)
    inp.add_argument("--images", type=str, nargs="+", help="path to images directories [required for stack or side modes]")
    inp.add_argument("--cover", type=str, help="path to csv tiles cover file, to filter tiles to tile [optional]")
    inp.add_argument("--workers", type=int, help="number of workers [default: CPU]")

    metric = parser.add_argument_group("Metrics Filtering")
    metric.add_argument(
        "--min", action="append", nargs=3, help="skip tile if class below metric value [0-1] (e.g --min Building QoD 0.7)"
    )
    metric.add_argument(
        "--max", action="append", nargs=3, help="skip tile if class above metric value [0-1] (e.g --max Building IoU 0.85)"
    )

    out = parser.add_argument_group("Outputs")
    out.add_argument("--vertical", action="store_true", help="output vertical image aggregate [optionnal for side mode]")
    out.add_argument("--geojson", action="store_true", help="output results as GeoJSON [optionnal for list mode]")
    out.add_argument("--format", type=str, default="webp", help="output images file format [default: webp]")
    out.add_argument("--out", type=str, help="output path")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def main(args):

    if not args.masks or not args.labels:
        assert args.mode != "list", "Parameters masks and labels are mandatories in list mode."
        assert not (args.min or args.max), "Both --masks and --labels mandatory, for metric filtering."

    if args.min or args.max:
        config = load_config(args.config)

    args.out = os.path.expanduser(args.out)
    cover = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))] if args.cover else None

    args_minmax = set()
    args.min = {(m[0], m[1]): m[2] for m in args.min} if args.min else dict()
    args.max = {(m[0], m[1]): m[2] for m in args.max} if args.max else dict()
    args_minmax.update(args.min.keys())
    args_minmax.update(args.max.keys())
    minmax = dict()
    for mm in args_minmax:
        mm_min = float(args.min[mm]) if mm in args.min else 0.0
        mm_max = float(args.max[mm]) if mm in args.max else 1.0
        assert mm_min < mm_max, "--min must be lower than --max, on {}".format(mm)
        minmax[mm] = {
            "min": mm_min,
            "max": mm_max,
            "class_id": [c for c, classe in enumerate(config["classes"]) if classe["title"] == mm[0]][0],
            "module": load_module("neat_eo.metrics." + mm[1]),
        }

    if not args.workers:
        args.workers = os.cpu_count()

    print("neo compare {} on CPU, with {} workers".format(args.mode, args.workers), file=sys.stderr, flush=True)

    if args.images:
        tiles = [tile for tile in tiles_from_dir(args.images[0], cover=cover)]
        assert len(tiles), "Empty images dir: {}".format(args.images[0])

        for image in args.images[1:]:
            assert sorted(tiles) == sorted([tile for tile in tiles_from_dir(image, cover=cover)]), "Unconsistent images dirs"

    if args.labels and args.masks:
        tiles_masks = [tile for tile in tiles_from_dir(args.masks, cover=cover)]
        tiles_labels = [tile for tile in tiles_from_dir(args.labels, cover=cover)]
        if args.images:
            assert sorted(tiles) == sorted(tiles_masks) == sorted(tiles_labels), "Unconsistent images/label/mask directories"
        else:
            assert len(tiles_masks), "Empty masks dir: {}".format(args.masks)
            assert len(tiles_labels), "Empty labels dir: {}".format(args.labels)
            assert sorted(tiles_masks) == sorted(tiles_labels), "Label and Mask directories are not consistent"
            tiles = tiles_masks

    tiles_list = []
    tiles_compare = []
    progress = tqdm(total=len(tiles), ascii=True, unit="tile")
    log = False if args.mode == "list" else Logs(os.path.join(args.out, "log"))

    with futures.ThreadPoolExecutor(args.workers) as executor:

        def worker(tile):
            x, y, z = list(map(str, tile))

            if args.masks and args.labels:

                label = np.array(Image.open(os.path.join(args.labels, z, x, "{}.png".format(y))))
                mask = np.array(Image.open(os.path.join(args.masks, z, x, "{}.png".format(y))))

                assert label.shape == mask.shape, "Inconsistent tiles (size or dimensions)"

                metrics = dict()
                for mm in minmax:
                    try:
                        metrics[mm] = getattr(minmax[mm]["module"], "get")(
                            torch.as_tensor(label, device="cpu"),
                            torch.as_tensor(mask, device="cpu"),
                            minmax[mm]["class_id"],
                        )
                    except:
                        progress.update()
                        return False, tile

                    if not (minmax[mm]["min"] <= metrics[mm] <= minmax[mm]["max"]):
                        progress.update()
                        return True, tile

            tiles_compare.append(tile)

            if args.mode == "side":
                for i, root in enumerate(args.images):
                    img = tile_image_from_file(tile_from_xyz(root, x, y, z)[1], force_rgb=True)

                    if i == 0:
                        side = np.zeros((img.shape[0], img.shape[1] * len(args.images), 3))
                        side = np.swapaxes(side, 0, 1) if args.vertical else side
                        image_shape = img.shape
                    else:
                        assert image_shape[0:2] == img.shape[0:2], "Unconsistent image size to compare"

                    if args.vertical:
                        side[i * image_shape[0] : (i + 1) * image_shape[0], :, :] = img
                    else:
                        side[:, i * image_shape[0] : (i + 1) * image_shape[0], :] = img

                tile_image_to_file(args.out, tile, np.uint8(side))

            elif args.mode == "stack":
                for i, root in enumerate(args.images):
                    tile_image = tile_image_from_file(tile_from_xyz(root, x, y, z)[1], force_rgb=True)

                    if i == 0:
                        image_shape = tile_image.shape[0:2]
                        stack = tile_image / len(args.images)
                    else:
                        assert image_shape == tile_image.shape[0:2], "Unconsistent image size to compare"
                        stack = stack + (tile_image / len(args.images))

                tile_image_to_file(args.out, tile, np.uint8(stack))

            elif args.mode == "list":
                tiles_list.append([tile, metrics])

            progress.update()
            return True, tile

        for ok, tile in executor.map(worker, tiles):
            if not ok and log:
                log.log("Warning: skipping. {}".format(str(tile)))

    if args.mode == "list":
        with open(args.out, mode="w") as out:

            if args.geojson:
                out.write('{"type":"FeatureCollection","features":[')
                first = True

            for tile_list in tiles_list:
                tile, metrics = tile_list
                x, y, z = list(map(str, tile))

                if args.geojson:
                    prop = '"properties":{{"x":{},"y":{},"z":{}'.format(x, y, z)
                    for metric in metrics:
                        prop += ',"{}":{:.3f}'.format(metric, metrics[metric])
                    geom = '"geometry":{}'.format(json.dumps(feature(tile, precision=6)["geometry"]))
                    out.write('{}{{"type":"Feature",{},{}}}}}'.format("," if not first else "", geom, prop))
                    first = False

                if not args.geojson:
                    out.write("{},{},{}".format(x, y, z))
                    for metric in metrics:
                        out.write("\t{:.3f}".format(metrics[metric]))
                    out.write(os.linesep)

            if args.geojson:
                out.write("]}")

            out.close()

    base_url = args.web_ui_base_url if args.web_ui_base_url else "."

    if args.mode == "side" and not args.no_web_ui:
        template = "compare.html" if not args.web_ui_template else args.web_ui_template
        web_ui(args.out, base_url, tiles, tiles_compare, args.format, template, union_tiles=False)

    if args.mode == "stack" and not args.no_web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        tiles = [tile for tile in tiles_from_dir(args.images[0])]
        web_ui(args.out, base_url, tiles, tiles_compare, args.format, template)
