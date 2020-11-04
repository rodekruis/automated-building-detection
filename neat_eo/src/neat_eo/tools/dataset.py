import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from neat_eo.core import load_config, check_classes, check_channels
from neat_eo.tiles import tiles_from_dir, tile_label_from_file, tiles_from_csv


def add_parser(subparser, formatter_class):

    parser = subparser.add_parser("dataset", help="train dataset helper", formatter_class=formatter_class)
    parser.add_argument("--config", type=str, help="path to config file [required, if no global config setting]")
    parser.add_argument("--dataset", type=str, required=True, help="dataset path [required]")
    parser.add_argument("--cover", type=str, help="path to csv tiles cover file, to filter tiles dataset on [optional]")
    parser.add_argument("--workers", type=int, help="number of workers [default: CPU]")

    choices = ["check", "weights"]
    parser.add_argument("--mode", type=str, default="check", choices=choices, help="dataset mode [default: check]")
    parser.set_defaults(func=main)


class LabelsDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_classes, cover=None):
        super().__init__()
        self.num_classes = num_classes
        self.tiles = [path for tile, path in tiles_from_dir(os.path.join(root, "labels"), cover=cover, xyz_path=True)]
        assert len(self.tiles), "Empty Dataset"

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        mask = torch.from_numpy(tile_label_from_file(self.tiles[i]))
        return torch.bincount(mask.view(-1), minlength=self.num_classes), mask.nelement()


def compute_classes_weights(dataset, classes, cover, workers):
    label_dataset = LabelsDataset(dataset, len(classes), cover)
    loader = DataLoader(label_dataset, batch_size=workers, num_workers=workers)
    n_classes = np.zeros(len(classes))
    n_pixels = 0
    for c, n in tqdm(loader, desc="Classes Weights", unit="batch", ascii=True):
        n_classes += c.data.numpy()[0]
        n_pixels += int(n.data.numpy()[0])

    weights = 1 / np.log(1.02 + (n_classes / n_pixels))  # cf https://arxiv.org/pdf/1606.02147.pdf
    return weights.round(3, out=weights).tolist()


def main(args):

    assert os.path.isdir(os.path.expanduser(args.dataset)), "--dataset path is not a directory"
    args.cover = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))] if args.cover else None
    config = load_config(args.config)

    if not args.workers:
        args.workers = os.cpu_count()

    print("neo dataset {} on CPU, with {} workers".format(args.mode, args.workers), file=sys.stderr, flush=True)

    if args.mode == "check":
        check_classes(config)
        check_channels(config)

        # TODO check dataset

    if args.mode == "weights":
        check_classes(config)
        weights = compute_classes_weights(args.dataset, config["classes"], args.cover, args.workers)
        print(",".join(map(str, weights)))
