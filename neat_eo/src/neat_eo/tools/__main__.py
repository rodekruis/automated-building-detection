import os
import sys
import argparse

import glob
import shutil
from importlib import import_module


def main():

    if not sys.version_info >= (3, 6):
        sys.exit("ERROR: neo needs Python 3.6 or later.")

    if not len(sys.argv) > 1:
        print("neo: Neat-EO.pink command line tools")
        print("")
        print("Usages:")
        print("neo -h, --help          show tools availables")
        print("neo <tool> -h, --help   show options availables for a tool")
        print("neo <tool> [...]        launch a neo tool command")
        sys.exit()

    tools = [os.path.basename(tool)[:-3] for tool in glob.glob(os.path.join(os.path.dirname(__file__), "[a-z]*.py"))]
    tools = [sys.argv[1]] if sys.argv[1] in tools else tools

    os.environ["COLUMNS"] = str(shutil.get_terminal_size().columns)  # cf https://bugs.python.org/issue13041
    fc = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=40, indent_increment=1)  # noqa: E731
    for i, arg in enumerate(sys.argv):  # handle negative values cf #64
        if len(arg) > 0 and arg[0] == "-" and arg[1].isdigit():
            sys.argv[i] = " " + arg
    parser = argparse.ArgumentParser(prog="neo", formatter_class=fc)
    subparser = parser.add_subparsers(title="Neat-EO.pink tools", metavar="")

    for tool in tools:
        if tool[0] != "_":
            module = import_module("neat_eo.tools.{}".format(tool))
            module.add_parser(subparser, formatter_class=fc)

    args = parser.parse_args()

    if "NEO_DEBUG" in os.environ and os.environ["NEO_DEBUG"] == "1":
        args.func(args)

    else:

        try:
            args.func(args)
        except (Exception) as error:
            sys.exit("{}ERROR: {}".format(os.linesep, error))
