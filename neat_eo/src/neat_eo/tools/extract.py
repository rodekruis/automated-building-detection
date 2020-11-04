import os
import sys

from neat_eo.core import load_module


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("extract", help="Extracts GeoJSON features from OSM .pbf", formatter_class=formatter_class)

    inp = parser.add_argument_group("Inputs")
    inp.add_argument("--type", type=str, required=True, help="OSM feature type to extract (e.g Building, Road) [required]")
    inp.add_argument("--pbf", type=str, required=True, help="path to .osm.pbf file [required]")

    out = parser.add_argument_group("Output")
    out.add_argument("--out", type=str, required=True, help="GeoJSON output file path [required]")

    parser.set_defaults(func=main)


def main(args):

    try:
        module = load_module("neat_eo.osm.{}".format(args.type.lower()))
    except:
        sys.exit("ERROR: Unavailable --type {}".format(args.type))

    if os.path.dirname(os.path.expanduser(args.out)):
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    osmium_handler = getattr(module, "{}Handler".format(args.type))()

    print("neo extract {} from {} to {}".format(args.type, args.pbf, args.out), file=sys.stderr, flush=True)
    print("\nNOTICE: could take time. Be patient...\n".format(args.type, args.pbf, args.out), file=sys.stderr, flush=True)

    osmium_handler.apply_file(filename=os.path.expanduser(args.pbf), locations=True)
    osmium_handler.save(os.path.expanduser(args.out))
