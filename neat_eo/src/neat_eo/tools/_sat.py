import os
import re
import sys
import json
import glob

import hashlib
import requests
import concurrent.futures as futures

from tqdm import tqdm
from zipfile import ZipFile
from datetime import datetime

from neat_eo.core import load_config, Logs
from neat_eo.tiles import tiles_from_csv, tiles_to_granules


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("sat", help="Retrieve Satellite Scenes", formatter_class=formatter_class)
    parser.add_argument("--config", type=str, help="path to config file [required]")
    parser.add_argument("--pg", type=str, help="If set, override config PostgreSQL dsn.")

    extent = parser.add_argument_group("Spatial extent [one among the following is required]")
    extent.add_argument("--cover", type=str, help="path to csv tiles cover file")
    extent.add_argument("--granules", type=str, nargs="+", help="Military Grid Granules, (e.g 31TFL)")
    extent.add_argument("--scenes", type=str, help="Path to a Scenes UUID file")

    filters = parser.add_argument_group("Filters")
    filters.add_argument("--level", type=str, choices=["2A", "3A"], help="Processing Level")
    filters.add_argument("--start", type=str, help="YYYY-MM-DD starting date")
    filters.add_argument("--end", type=str, help="YYYY-MM-DD end date")
    filters.add_argument("--clouds", type=int, help="max threshold for cloud coverage [0-100]")
    filters.add_argument("--limit", type=int, default=500, help="max number of results per granule")

    dl = parser.add_argument_group("Download")
    dl.add_argument("--download", action="store_true", help="if set, perform also download operation.")
    dl.add_argument("--workers", type=int, default=4, help="number of workers [default: 4]")
    dl.add_argument("--timeout", type=int, default=180, help="download request timeout (in seconds) [default: 180]")

    parser.add_argument("--out", type=str, nargs="?", help="output directory path [required if download is set]")
    parser.set_defaults(func=main)


THEIA_URL = "https://theia.cnes.fr/atdistrib"


def get_token(login, password):
    resp = requests.post(THEIA_URL + "/services/authenticate/", data={"ident": login, "pass": password})
    assert resp, "Unable to join Theia Server, check your connection"
    token = resp.text
    assert re.match("[a-zA-Z0-9]+", token), "Invalid authentification, check you login/pass"
    return token


def md5(path):

    assert os.path.isfile(path), "Unable to perform md5 on {}".format(path)

    with open(path, "rb") as fp:
        md5 = hashlib.md5()
        while True:
            chunk = fp.read(16384)
            if not chunk:
                break
            md5.update(chunk)
        return md5.hexdigest()


def search_scenes(args, log):

    scenes = []
    for granule in args.granules:
        data = {"location": "T" + granule, "maxRecords": 500}
        if args.level:
            data["processingLevel"] = "LEVEL" + args.level
        if args.start:
            data["startDate"] = args.start
        if args.end:
            data["completionDate"] = args.end

        url = THEIA_URL + "/resto2/api/collections/SENTINEL2/search.json?"
        data = json.loads(requests.get(url, params=data).text)
        assert data, "Unable to perform query: {}".format(url)

        if not len(data["features"]):
            log.log("======================================================")
            log.log("WARNING: No scene found for {} granule".format(granule))
            log.log("======================================================")
            continue

        log.log("=============================================================================")
        log.log("Selected scenes for {} granule".format(granule))
        log.log("-----------------------------------------------------------------------------")
        log.log("   Date        Clouds              Scene UUID                 Processing     ")
        log.log("-----------------------------------------------------------------------------")

        features = (
            [feature for feature in data["features"] if int(feature["properties"]["cloudCover"]) <= args.clouds]
            if args.clouds is not None
            else data["features"]
        )

        features = [feature for i, feature in enumerate(features) if i < args.limit]
        for feature in features:
            date = datetime.strptime(feature["properties"]["startDate"], "%Y-%m-%dT%H:%M:%SZ").date()
            cover = str(feature["properties"]["cloudCover"]).rjust(3, " ")
            scenes.append(
                {
                    "uuid": feature["id"],
                    "checksum": feature["properties"]["services"]["download"]["checksum"],
                    "dir": feature["properties"]["title"],
                }
            )
            log.log("{}\t{}\t{}\t{}".format(date, cover, feature["id"], feature["properties"]["processingLevel"]))
    log.log("=============================================================================")

    return scenes


def main(args):
    assert args.cover or args.granules or args.scenes, "Either --cover OR --granules OR --scenes is mandatory"
    assert not (args.download and not args.out), "--download implies out parameter"
    assert args.limit, "What about increasing --limit value ?"
    config = load_config(args.config)

    if args.cover:
        args.pg = args.pg if args.pg else config["auth"]["pg"]
        assert args.pg, "PostgreSQL connection settting is mandatory with --cover"
        args.granules = tiles_to_granules(tiles_from_csv(os.path.expanduser(args.cover)), args.pg)

    if args.out:
        args.out = os.path.expanduser(args.out)
        os.makedirs(args.out, exist_ok=True)
        log = Logs(os.path.join(args.out, "log"), out=sys.stderr)
    else:
        log = Logs(None, out=sys.stderr)

    log.log("neo sat on granules: {}".format(" ".join(args.granules)))
    scenes = search_scenes(args, log)

    if args.download:

        log.log("")
        log.log("=============================================================================")
        log.log("Downloading selected scenes")
        log.log("=============================================================================")

        report = []
        login, password = dict([auth.split("=") for auth in config["auth"]["theia"].split(" ")]).values()

        with futures.ThreadPoolExecutor(args.workers) as executor:

            def worker(scene):

                scene_dir = os.path.join(args.out, scene["dir"][:42])  # 42 related to Theia MD issue, dirty workaround
                if not os.path.isabs(scene_dir):
                    scene_dir = "./" + scene_dir

                if glob.glob(scene_dir + "*"):
                    scene["dir"] = glob.glob(scene_dir + "*")[0]
                    return scene, None, True  # Already Downloaded

                token = get_token(login, password)
                url = THEIA_URL + "/resto2/collections/SENTINEL2/{}/download/?issuerId=theia".format(scene["uuid"])
                resp = requests.get(url, headers={"Authorization": "Bearer {}".format(token)}, stream=True)
                if resp is None:
                    return scene, None, False  # Auth issue

                zip_path = os.path.join(args.out, scene["uuid"] + ".zip")
                with open(zip_path, "wb") as fp:
                    progress = tqdm(unit="B", desc=scene["uuid"], total=int(resp.headers["Content-Length"]))
                    for chunk in resp.iter_content(chunk_size=16384):
                        progress.update(16384)
                        fp.write(chunk)

                    return scene, zip_path, True

                return scene, None, False  # Write issue

            for scene, zip_path, ok in executor.map(worker, scenes):
                if zip_path and md5(zip_path) == scene["checksum"]:
                    scene["dir"] = os.path.dirname(ZipFile(zip_path).namelist()[0])
                    ZipFile(zip_path).extractall(args.out)
                    os.remove(zip_path)
                    report.append("Scene {} available in {}".format(scene["uuid"], scene["dir"]))
                elif ok:
                    report.append("SKIPPING downloading {}, as already in {}".format(scene["uuid"], scene["dir"]))
                else:
                    report.append("ERROR: Unable to retrieve Scene {}".format(scene["uuid"]))

        log.log("")
        log.log("=============================================================================")
        for line in report:
            log.log(line)
        log.log("=============================================================================")
