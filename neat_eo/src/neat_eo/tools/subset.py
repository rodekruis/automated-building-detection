import os
import sys
import shutil

import mercantile
from tqdm import tqdm

from neat_eo.tiles import tiles_from_csv, tile_from_xyz
from neat_eo.core import web_ui


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "subset", help="Filter images in a slippy map dir using a csv tiles cover", formatter_class=formatter_class
    )
    inp = parser.add_argument_group("Inputs")
    inp.add_argument("--dir", type=str, required=True, help="to XYZ tiles input dir path [required]")
    inp.add_argument("--cover", type=str, required=True, help="path to csv cover file to filter dir by [required]")

    mode = parser.add_argument_group("Alternate modes, as default is to create relative symlinks")
    mode.add_argument("--copy", action="store_true", help="copy tiles from input to output")
    mode.add_argument("--delete", action="store_true", help="delete tiles listed in cover")

    out = parser.add_argument_group("Output")
    out.add_argument("--quiet", action="store_true", help="if set, suppress warning output")
    out.add_argument("--out", type=str, nargs="?", default=os.getcwd(), help="output dir path [required for copy]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def main(args):
    assert args.out or args.delete, "out parameter is required"
    args.out = os.path.expanduser(args.out)

    print("neo subset {} with cover {}, on CPU".format(args.dir, args.cover), file=sys.stderr, flush=True)

    ext = set()
    tiles = set(tiles_from_csv(os.path.expanduser(args.cover)))
    assert len(tiles), "Empty Cover: {}".format(args.cover)

    for tile in tqdm(tiles, ascii=True, unit="tiles"):

        if isinstance(tile, mercantile.Tile):
            src_tile = tile_from_xyz(args.dir, tile.x, tile.y, tile.z)
            if not src_tile:
                if not args.quiet:
                    print("WARNING: skipping tile {}".format(tile), file=sys.stderr, flush=True)
                continue
            _, src = src_tile
            dst_dir = os.path.join(args.out, str(tile.z), str(tile.x))
        else:
            src = tile
            dst_dir = os.path.join(args.out, os.path.dirname(tile))

        assert os.path.isfile(src)
        dst = os.path.join(dst_dir, os.path.basename(src))
        ext.add(os.path.splitext(src)[1][1:])

        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        if args.delete:
            os.remove(src)
            assert not os.path.lexists(src)
        elif args.copy:
            shutil.copyfile(src, dst)
            assert os.path.exists(dst)
        else:
            if os.path.islink(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
            assert os.path.islink(dst)

    if tiles and not args.no_web_ui and not args.delete:
        assert len(ext) == 1, "ERROR: Mixed extensions, can't generate Web UI"
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "."
        web_ui(args.out, base_url, tiles, tiles, list(ext)[0], template)
