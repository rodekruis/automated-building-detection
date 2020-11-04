import os
import sys
import time
import concurrent.futures as futures

import requests
from tqdm import tqdm
from mercantile import xy_bounds

from neat_eo.core import web_ui, Logs
from neat_eo.tiles import tiles_from_csv, tile_image_from_url, tile_image_to_file


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "download", help="Downloads tiles from a remote server (XYZ or WMS)", formatter_class=formatter_class
    )

    ws = parser.add_argument_group("Web Server")
    help = "URL server endpoint, with: {z}/{x}/{y} or {xmin},{ymin},{xmax},{ymax} [required]"
    ws.add_argument("--url", type=str, required=True, help=help)
    ws.add_argument("--type", type=str, default="XYZ", choices=["XYZ", "WMS"], help="service type [default: XYZ]")
    ws.add_argument("--rate", type=int, default=10, help="download rate limit in max requests/seconds [default: 10]")
    ws.add_argument("--timeout", type=int, default=10, help="download request timeout (in seconds) [default: 10]")
    ws.add_argument("--workers", type=int, help="number of workers [default: same as --rate value]")

    cover = parser.add_argument_group("Coverage to download")
    cover.add_argument("--cover", type=str, required=True, help="path to .csv tiles list [required]")

    out = parser.add_argument_group("Output")
    out.add_argument("--format", type=str, default="webp", help="file format to save images in [default: webp]")
    out.add_argument("--out", type=str, required=True, help="output directory path [required]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def main(args):

    tiles = list(tiles_from_csv(args.cover))
    assert len(tiles), "Empty cover: {}".format(args.cover)

    args.workers = min(os.cpu_count(), args.rate) if not args.workers else args.workers

    if os.path.dirname(os.path.expanduser(args.out)):
        os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    log = Logs(os.path.join(args.out, "log"), out=sys.stderr)
    log.log("neo download with {} workers, at max {} req/s, from: {}".format(args.workers, args.rate, args.url))

    already_dl = 0
    dl = 0

    with requests.Session() as session:

        progress = tqdm(total=len(tiles), ascii=True, unit="image")
        with futures.ThreadPoolExecutor(args.workers) as executor:

            def worker(tile):
                tick = time.monotonic()
                progress.update()

                try:
                    x, y, z = map(str, [tile.x, tile.y, tile.z])
                    os.makedirs(os.path.join(args.out, z, x), exist_ok=True)
                except:
                    return tile, None, False

                path = os.path.join(args.out, z, x, "{}.{}".format(y, args.format))
                if os.path.isfile(path):  # already downloaded
                    return tile, None, True

                if args.type == "XYZ":
                    url = args.url.format(x=tile.x, y=tile.y, z=tile.z)
                elif args.type == "WMS":
                    xmin, ymin, xmax, ymax = xy_bounds(tile)
                    url = args.url.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                res = tile_image_from_url(session, url, args.timeout)
                if res is None:  # let's retry once
                    res = tile_image_from_url(session, url, args.timeout)
                    if res is None:
                        return tile, url, False

                try:
                    tile_image_to_file(args.out, tile, res)
                except OSError:
                    return tile, url, False

                tock = time.monotonic()

                time_for_req = tock - tick
                time_per_worker = args.workers / args.rate

                if time_for_req < time_per_worker:
                    time.sleep(time_per_worker - time_for_req)

                return tile, url, True

            for tile, url, ok in executor.map(worker, tiles):
                if url and ok:
                    dl += 1
                elif not url and ok:
                    already_dl += 1
                else:
                    log.log("Warning:\n {} failed, skipping.\n {}\n".format(tile, url))

    if already_dl:
        log.log("Notice: {} tiles were already downloaded previously, and so skipped now.".format(already_dl))
    if already_dl + dl == len(tiles):
        log.log("Notice: Coverage is fully downloaded.")

    if not args.no_web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "."
        web_ui(args.out, base_url, tiles, tiles, args.format, template)
