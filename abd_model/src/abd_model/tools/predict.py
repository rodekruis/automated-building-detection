import os
import uuid
from tqdm import tqdm

import math
import mercantile
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from abd_model.core import load_config, load_module, check_classes, check_channels, make_palette, web_ui, Logs
from abd_model.tiles import tile_label_to_file, tiles_from_csv


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "predict", help="Predict masks, from given inputs and an already trained model", formatter_class=formatter_class
    )

    inp = parser.add_argument_group("Inputs")
    inp.add_argument("--dataset", type=str, help="predict dataset directory path [required]")
    inp.add_argument("--checkpoint", type=str, required=True, help="path to the trained model to use [required]")
    inp.add_argument("--config", type=str, help="path to config file [required, if no global config setting]")
    inp.add_argument("--cover", type=str, help="path to csv tiles cover file, to filter tiles to predict [optional]")

    out = parser.add_argument_group("Outputs")
    out.add_argument("--out", type=str, required=True, help="output directory path [required]")
    out.add_argument("--metatiles", action="store_true", help="if set, use surrounding tiles to avoid margin effects")
    out.add_argument("--keep_borders", action="store_true", help="if set, with --metatiles, force borders tiles to be kept")

    perf = parser.add_argument_group("Performances")
    perf.add_argument("--bs", type=int, help="batch size [default: CPU/GPU]")
    perf.add_argument("--workers", type=int, help="number of pre-processing images workers, per GPU [default: batch_size]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def gpu_worker(rank, world_size, lock_file, args, config, dataset, palette, transparency):

    dist.init_process_group(backend="nccl", init_method="file://" + lock_file, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    chkpt = torch.load(args.checkpoint, map_location=torch.device(rank))
    nn_module = load_module("abd_model.nn.{}".format(chkpt["nn"].lower()))
    nn = getattr(nn_module, chkpt["nn"])(chkpt["shape_in"], chkpt["shape_out"], chkpt["encoder"].lower()).to(rank)
    nn = DistributedDataParallel(nn, device_ids=[rank], find_unused_parameters=True)

    chkpt = torch.load(os.path.expanduser(args.checkpoint), map_location="cuda:{}".format(rank))
    assert nn.module.version == chkpt["model_version"], "Model Version mismatch"
    nn.load_state_dict(chkpt["state_dict"])

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, sampler=sampler)
    assert len(loader), "Empty predict dataset directory. Check your path."

    C, W, H = chkpt["shape_out"]

    nn.eval()
    with torch.no_grad():

        dataloader = tqdm(loader, desc="Predict", unit="Batch/GPU", ascii=True) if rank == 0 else loader

        for images, tiles in dataloader:

            if args.metatiles:
                N = images.shape[0]
                qs = int(W / 4)
                hs = int(W / 2)
                ts = int(W)

                # fmt:off
                probs = np.zeros((N, C, W, H), dtype=np.float)
                probs[:, :, 0:hs, 0:hs] = nn(images[:, :, 0:ts, 0:ts].to(rank)).data.cpu().numpy()[:, :, qs:-qs, qs:-qs]
                probs[:, :, 0:hs,  hs:] = nn(images[:, :, 0:ts,  hs:].to(rank)).data.cpu().numpy()[:, :, qs:-qs, qs:-qs]
                probs[:, :, hs:,  0:hs] = nn(images[:, :, hs:,  0:ts].to(rank)).data.cpu().numpy()[:, :, qs:-qs, qs:-qs]
                probs[:, :, hs:,   hs:] = nn(images[:, :, hs:,   hs:].to(rank)).data.cpu().numpy()[:, :, qs:-qs, qs:-qs]
                # fmt:on
            else:
                probs = nn(images.to(rank)).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))
                mask = np.zeros((W, H), dtype=np.uint8)

                for c in range(C):
                    mask += np.around(prob[c, :, :]).astype(np.uint8) * c

                tile_label_to_file(args.out, mercantile.Tile(x, y, z), palette, transparency, mask)


def main(args):
    config = load_config(args.config)
    check_channels(config)
    check_classes(config)

    assert torch.cuda.is_available(), "No GPU support found. Check CUDA and NVidia Driver install."
    assert torch.distributed.is_nccl_available(), "No NCCL support found. Check your PyTorch install."

    world_size = torch.cuda.device_count()
    args.bs = args.bs if args.bs is not None else math.floor(os.cpu_count() / world_size)
    args.workers = args.workers if args.workers is not None else args.bs

    palette, transparency = make_palette([classe["color"] for classe in config["classes"]])
    args.cover = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))] if args.cover else None

    args.out = os.path.expanduser(args.out)
    log = Logs(os.path.join(args.out, "log"))

    chkpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    log.log("abd predict on {} GPUs, with {} workers/GPU and {} tiles/batch".format(world_size, args.workers, args.bs))
    log.log("Model {} - UUID: {}".format(chkpt["nn"], chkpt["uuid"]))
    log.log("---")
    loader = load_module("abd_model.loaders.{}".format(chkpt["loader"].lower()))

    lock_file = os.path.abspath(os.path.join(args.out, str(uuid.uuid1())))

    dataset = getattr(loader, chkpt["loader"])(
        config,
        chkpt["shape_in"][1:3],
        args.dataset,
        args.cover,
        mode="predict",
        metatiles=args.metatiles,
        keep_borders=args.keep_borders,
    )

    mp.spawn(gpu_worker, nprocs=world_size, args=(world_size, lock_file, args, config, dataset, palette, transparency))

    if os.path.exists(lock_file):
        os.remove(lock_file)

    if not args.no_web_ui and dataset.cover:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "."
        web_ui(args.out, base_url, dataset.cover, dataset.cover, "png", template)
