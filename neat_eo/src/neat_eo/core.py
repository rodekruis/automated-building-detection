import os
import sys
import glob
import toml
from importlib import import_module

import re
import colorsys
import webcolors
from pathlib import Path

from neat_eo.tiles import tile_pixel_to_location, tiles_to_geojson


#
# Import module
#
def load_module(module):
    module = import_module(module)
    assert module, "Unable to import module {}".format(module)
    return module


#
# Config
#
def load_config(path):
    """Loads a dictionary from configuration file."""

    if not path and "NEO_CONFIG" in os.environ:
        path = os.environ["NEO_CONFIG"]
    if not path and os.path.isfile(os.path.expanduser("~/.neo_config")):
        path = "~/.neo_config"
    assert path, "Either ~/.neo_config or NEO_CONFIG env var or --config parameter, is required."

    config = toml.load(os.path.expanduser(path))
    assert config, "Unable to parse config file"

    # Set default values

    if "model" not in config.keys():
        config["model"] = {}

    if "ts" not in config["model"].keys():
        config["model"]["ts"] = (512, 512)

    if "train" not in config.keys():
        config["train"] = {}

    if "pretrained" not in config["train"].keys():
        config["train"]["pretrained"] = True

    if "bs" not in config["train"].keys():
        config["train"]["bs"] = 4

    if "auth" not in config.keys():
        config["auth"] = {}

    if "da" in config["train"].keys():
        config["train"]["da"] = dict(config["train"]["da"])  # dict is serializable

    if "optimizer" in config["train"].keys():
        config["train"]["optimizer"] = dict(config["train"]["optimizer"])  # dict is serializable
    else:
        config["train"]["optimizer"] = {"name": "Adam", "lr": 0.0001}

    assert "classes" in config.keys(), "CONFIG: Classes are mandatory"
    for c, classe in enumerate(config["classes"]):
        config["classes"][c]["weight"] = config["classes"][c]["weight"] if "weight" in config["classes"][c].keys() else 1.0
        if config["classes"][c]["color"] == "transparent" and "weight" not in config["classes"][c].keys():
            config["classes"][c]["weight"] = 0.0

    return config


def check_channels(config):
    assert "channels" in config.keys(), "CONFIG: At least one Channel is mandatory"

    # TODO


def check_classes(config):
    """Check if config file classes subpart is consistent. Exit on error if not."""

    assert "classes" in config.keys() and len(config["classes"]) >= 2, "CONFIG: At least 2 Classes are mandatory"

    for classe in config["classes"]:
        assert "title" in classe.keys() and len(classe["title"]), "CONFIG: Missing or Empty classes.title value"
        assert "color" in classe.keys() and check_color(classe["color"]), "CONFIG: Missing or Invalid classes.color value"


def check_model(config):

    assert "model" in config.keys(), "CONFIG: Missing or Invalid model"

    # TODO


#
# Logs
#
class Logs:
    def __init__(self, path, out=sys.stderr):
        """Create a logs instance on a logs file."""

        self.fp = None
        self.out = out
        if path:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fp = open(path, mode="a")

    def log(self, msg):
        """Log a new message to the opened logs file, and optionnaly on stdout or stderr too."""
        if self.fp:
            self.fp.write(msg + os.linesep)
            self.fp.flush()

        if self.out:
            print(msg, file=self.out)


#
# Colors
#
def make_palette(colors, complementary=False):
    """Builds a PNG PIL color palette from Classes CSS3 color names, or hex values patterns as #RRGGBB."""

    assert 0 < len(colors) < 256

    try:
        transparency = [key for key, color in enumerate(colors) if color == "transparent"][0]
    except:
        transparency = None

    colors = ["white" if color.lower() == "transparent" else color for color in colors]
    hex_colors = [webcolors.CSS3_NAMES_TO_HEX[color.lower()] if color[0] != "#" else color for color in colors]
    rgb_colors = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hex_colors]

    palette = list(sum(rgb_colors, ()))  # flatten
    palette = palette if not complementary else complementary_palette(palette)

    return palette, transparency


def complementary_palette(palette):
    """Creates a PNG PIL complementary colors palette based on an initial PNG PIL palette."""

    comp_palette = []
    colors = [palette[i : i + 3] for i in range(0, len(palette), 3)]

    for color in colors:
        r, g, b = [v for v in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        comp_palette.extend(map(int, colorsys.hsv_to_rgb((h + 0.5) % 1, s, v)))

    return comp_palette


def check_color(color):
    """Check if an input color is or not valid (i.e CSS3 color name, transparent, or #RRGGBB)."""

    color = "white" if color.lower() == "transparent" else color
    hex_color = webcolors.CSS3_NAMES_TO_HEX[color.lower()] if color[0] != "#" else color
    return bool(re.match(r"^#([0-9a-fA-F]){6}$", hex_color))


#
# Web UI
#
def web_ui(out, base_url, coverage_tiles, selected_tiles, ext, template, union_tiles=True):

    out = os.path.expanduser(out)
    template = os.path.expanduser(template)

    templates = glob.glob(os.path.join(Path(__file__).parent, "web_ui", "*"))
    if os.path.isfile(template):
        templates.append(template)
    if os.path.lexists(os.path.join(out, "index.html")):
        os.remove(os.path.join(out, "index.html"))  # if already existing output dir, as symlink can't be overwriten
    os.symlink(os.path.basename(template), os.path.join(out, "index.html"))

    def process_template(template):
        web_ui = open(template, "r").read()
        web_ui = re.sub("{{base_url}}", base_url, web_ui)
        web_ui = re.sub("{{ext}}", ext, web_ui)
        web_ui = re.sub("{{tiles}}", "tiles.json" if selected_tiles else "''", web_ui)

        if coverage_tiles:
            tile = list(coverage_tiles)[0]  # Could surely be improved, but for now, took the first tile to center on
            x, y, z = map(int, [tile.x, tile.y, tile.z])
            web_ui = re.sub("{{zoom}}", str(z), web_ui)
            web_ui = re.sub("{{center}}", str(list(tile_pixel_to_location(tile, 0.5, 0.5))[::-1]), web_ui)

        with open(os.path.join(out, os.path.basename(template)), "w", encoding="utf-8") as fp:
            fp.write(web_ui)

    for template in templates:
        process_template(template)

    if selected_tiles:
        with open(os.path.join(out, "tiles.json"), "w", encoding="utf-8") as fp:
            fp.write(tiles_to_geojson(selected_tiles, union_tiles))
