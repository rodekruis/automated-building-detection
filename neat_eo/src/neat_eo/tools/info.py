import os
import re
import sys
import cv2
import torch
import requests
import rasterio
import neat_eo as neo


def add_parser(subparser, formatter_class):

    epilog = "Usages:\nTo kill GPU processes: neo info --processes | xargs sudo kill -9"
    help = "Retrieve Neat-EO.pink informations and MetaData"
    parser = subparser.add_parser("info", help=help, formatter_class=formatter_class, epilog=epilog)
    parser.add_argument("--version", action="store_true", help="if set, output Neat-EO.pink version only")
    parser.add_argument("--processes", action="store_true", help="if set, output GPU processes list")
    parser.add_argument("--checkpoint", type=str, help="if set with a .pth path, output related model metadata")
    parser.set_defaults(func=main)


def main(args):

    if args.version:
        print(neo.__version__)
        sys.exit()

    if args.checkpoint:
        try:
            chkpt = torch.load(os.path.expanduser(args.checkpoint), map_location=torch.device("cpu"))
        except:
            sys.exit("Unable to open checkpoint: {}".format(args.checkpoint))

        print(chkpt["doc_string"])
        for key in chkpt.keys():
            if key in ["state_dict", "optimizer", "doc_string"]:
                continue
            print(key.ljust(20) + ": " + str(chkpt[key]))
        sys.exit()

    if args.processes:
        try:
            devices = [device for device in map(int, os.getenv("CUDA_VISIBLE_DEVICES").split(","))]
        except:
            devices = range(torch.cuda.device_count())

        pids = set()
        for i in devices:
            lsof = os.popen("lsof /dev/nvidia{}".format(i)).read()
            for row in re.sub("( )+", "|", lsof).split("\n"):
                try:
                    pid = row.split("|")[1]
                    pids.add(int(pid))
                except:
                    continue

        for pid in sorted(pids):
            print("{} ".format(pid), end="")
        sys.exit()

    ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024 * 1000))  # 1000 Mo -> 1Go
    vram = 0
    for i in range(torch.cuda.device_count()):
        vram += torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1000)
    vram = round(vram)

    release = "Unknown"
    with open("/etc/os-release") as fp:
        for line in fp:
            try:
                k, v = line.strip().split("=")
                release = v.replace('"', "") if k == "PRETTY_NAME" else release
            except:
                pass

    try:
        r = requests.get("http://localhost")
        is_httpd = "Running" if r.status_code == 200 else "No"
    except:
        is_httpd = "Not Running"

    if ram < 16 or ram < vram:
        print("========================================")
        print("WARNING: Consider to increase RAM.")
        print("========================================")

    if os.cpu_count() < 8 or os.cpu_count() < (4 * torch.cuda.device_count()):
        print("========================================")
        print("WARNING: Consider to increase CPU cores.")
        print("========================================")

    if not torch.cuda.device_count():
        print("========================================")
        print("WARNING: No GPU found, neo train won't be usable.")
        print("========================================")

    if torch.cuda.device_count() and vram < 8:
        print("========================================")
        print("WARNING: GPUs found don't have enough VRAM to be usable.")
        print("========================================")

    if is_httpd != "Running":
        print("========================================")
        print("WARNING: No HTTPD server running on localhost")
        print("         Will impact WebUI rendering.")
        print("========================================")

    print("========================================")
    print("Neat-EO.pink: " + neo.__version__)
    print("========================================")
    print("Linux   " + release)
    print("Python  " + sys.version[:5])
    print("Torch   " + torch.__version__)
    print("OpenCV  " + cv2.__version__)
    print("GDAL    " + rasterio._base.gdal_version())
    print("Cuda    " + torch.version.cuda)
    print("Cudnn   " + str(torch.backends.cudnn.version()))
    print("NCCL    " + str(torch.cuda.nccl.version()))
    print("========================================")
    print("RAM     " + str(ram) + "Go")
    print("CPUs    " + str(os.cpu_count()))
    print("========================================")
    print("VRAM    " + str(vram) + "Go")
    print("GPUs    " + str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        print(" - " + torch.cuda.get_device_name(i))
    print("========================================")
    print("HTTPD   " + is_httpd)
    print("========================================")
