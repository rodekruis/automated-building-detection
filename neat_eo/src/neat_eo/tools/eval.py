import os
import math
import uuid
from tqdm import tqdm

import torch
import torch.optim
import torch.backends.cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from neat_eo.core import load_config, load_module, check_model, check_channels, check_classes
from neat_eo.tiles import tiles_from_csv
from neat_eo.metrics.core import Metrics
from neat_eo.tools.dataset import compute_classes_weights


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("eval", help="Evals a model on a dataset", formatter_class=formatter_class)
    parser.add_argument("--config", type=str, help="path to config file [required, if no global config setting]")

    data = parser.add_argument_group("Dataset")
    data.add_argument("--dataset", type=str, required=True, help="dataset path [required]")
    data.add_argument("--cover", type=str, help="path to csv tiles cover file, to filter tiles dataset on [optional]")
    data.add_argument("--classes_weights", type=str, help="classes weights separated with comma or 'auto' [optional]")
    data.add_argument("--tiles_weights", type=str, help="path to csv tiles cover file, with specials weights on [optional]")
    data.add_argument("--loader", type=str, help="dataset loader name [if set override config file value]")

    ev = parser.add_argument_group("Eval")
    ev.add_argument("--bs", type=int, help="batch size")
    ev.add_argument("--metrics", type=str, nargs="+", help="metric name (e.g QoD IoU MCC)")
    ev.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint.")
    ev.add_argument("--workers", type=int, help="number of pre-processing images workers, per GPU [default: batch size]")

    parser.set_defaults(func=main)


def main(args):
    config = load_config(args.config)
    args.cover = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))] if args.cover else None
    if args.classes_weights:
        try:
            args.classes_weights = list(map(float, args.classes_weights.split(",")))
        except:
            assert args.classes_weights == "auto", "invalid --classes_weights value"
            args.classes_weights = compute_classes_weights(args.dataset, config["classes"], args.cover, os.cpu_count())
    else:
        args.classes_weights = [classe["weight"] for classe in config["classes"]]

    args.tiles_weights = (
        [(tile, weight) for tile, weight in tiles_from_csv(os.path.expanduser(args.tiles_weights), extra_columns=True)]
        if args.tiles_weights
        else None
    )

    args.bs = args.bs if args.bs else config["train"]["bs"]
    check_classes(config)
    check_channels(config)
    check_model(config)

    assert torch.cuda.is_available(), "No GPU support found. Check CUDA and NVidia Driver install."
    assert torch.distributed.is_nccl_available(), "No NCCL support found. Check your PyTorch install."
    world_size = 1  # Hard Coded since eval MultiGPUs not yet implemented

    args.workers = min(args.bs if not args.workers else args.workers, math.floor(os.cpu_count() / world_size))

    print("neo eval on 1 GPU, with {} workers, and {} tiles/batch".format(args.workers, args.bs))

    loader = load_module("neat_eo.loaders.{}".format(config["model"]["loader"].lower()))

    assert os.path.isdir(os.path.expanduser(args.dataset)), "--dataset path is not a directory"
    dataset = getattr(loader, config["model"]["loader"])(
        config, config["model"]["ts"], args.dataset, args.cover, args.tiles_weights, "eval"
    )
    assert len(dataset), "Empty or Invalid --dataset content"
    shape_in = dataset.shape_in
    shape_out = dataset.shape_out
    print("DataSet Eval:            {}".format(args.dataset))

    print("\n--- Input tensor")
    num_channel = 1  # 1-based numerotation
    for channel in config["channels"]:
        for band in channel["bands"]:
            print("Channel {}:\t\t {} - (band:{})".format(num_channel, channel["name"], band))
            num_channel += 1

    print("\n--- Output Classes ---")
    for c, classe in enumerate(config["classes"]):
        print("Class {}:\t\t {} ({:.2f})".format(c, classe["title"], args.classes_weights[c]))

    print("\n--- Model ---")
    for hp in config["model"]:
        print("{}{}".format(hp.ljust(25, " "), config["model"][hp]))

    lock_file = os.path.abspath(os.path.join("/tmp", str(uuid.uuid1())))
    mp.spawn(gpu_worker, nprocs=world_size, args=(world_size, lock_file, dataset, shape_in, shape_out, args, config))
    if os.path.exists(lock_file):
        os.remove(lock_file)


def gpu_worker(rank, world_size, lock_file, dataset, shape_in, shape_out, args, config):

    dist.init_process_group(backend="nccl", init_method="file://" + lock_file, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, drop_last=True, num_workers=args.workers)

    nn_module = load_module("neat_eo.nn.{}".format(config["model"]["nn"].lower()))
    nn = getattr(nn_module, config["model"]["nn"])(
        shape_in, shape_out, config["model"]["encoder"].lower(), config["train"]
    ).cuda(rank)
    nn = DistributedDataParallel(nn, device_ids=[rank], find_unused_parameters=True)

    if args.checkpoint:
        chkpt = torch.load(os.path.expanduser(args.checkpoint), map_location="cuda:{}".format(rank))
        assert nn.module.version == chkpt["model_version"], "Model Version mismatch"
        nn.load_state_dict(chkpt["state_dict"])

        if rank == 0:
            print("\n--- Using Checkpoint ---")
            print("Path:\t\t {}".format(args.checkpoint))
            print("UUID:\t\t {}".format(chkpt["uuid"]))

    nn.eval()
    with torch.no_grad():
        args.metrics = args.metrics if args.metrics else config["train"]["metrics"]
        metrics = Metrics(args.metrics, config["classes"], config=config)

        assert len(loader), "Empty or Inconsistent DataSet"
        dataloader = tqdm(loader, desc="Eval", unit="Batch", ascii=True) if rank == 0 else loader

        for images, masks, tiles, tiles_weights in dataloader:
            images = images.cuda(rank, non_blocking=True)
            masks = masks.cuda(rank, non_blocking=True)
            outputs = nn(images)

            if rank == 0:
                for mask, output in zip(masks, outputs):
                    metrics.add(mask, output)

        if rank == 0:
            print("\n{}  μ\t   σ".format(" ".ljust(25, " ")))
            for c, classe in enumerate(config["classes"]):
                if classe["weight"] != 0.0 and classe["color"] != "transparent":
                    for k, v in metrics.get()[c].items():
                        print("{}{:.3f}\t {:.3f}".format((classe["title"] + " " + k).ljust(25, " "), v["μ"], v["σ"]))

    dist.destroy_process_group()
