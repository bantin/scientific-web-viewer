"""
Helper to display images, numpy arrays, and matplotlib figures in the web viewer.

CLI usage:
    python show.py /path/to/image.tiff
    python show.py --npy /path/to/data.npy
    python show.py --expr "np.random.randn(100, 256, 256)"

From Python / inline scripts:
    from show import show, show_array, show_figure
    show_array(my_array, name="voltage_trace")
    show_figure(fig, name="summary_plot")
    show("/path/to/existing/file.tiff")
"""

import os
import sys
import tempfile
import urllib.parse
from pathlib import Path

import numpy as np

VIEW_PORT = int(os.environ.get("VIEW_PORT", "8089"))
TEMP_DIR = "/tmp/web_viewer_temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def _url(filepath: str) -> str:
    encoded = urllib.parse.quote(os.path.abspath(filepath))
    return f"http://localhost:{VIEW_PORT}/view?path={encoded}"


def _print_url(url: str):
    # OSC 8 clickable hyperlink for modern terminals
    sys.stdout.write(f"\033]8;;{url}\033\\{url}\033]8;;\033\\\n")
    sys.stdout.flush()


def show(filepath: str):
    """Print a clickable viewer URL for an existing image file."""
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        return
    _print_url(_url(filepath))


def show_array(arr: np.ndarray, name: str = "array"):
    """Save a numpy array as a TIFF and print the viewer URL."""
    import tifffile
    path = os.path.join(TEMP_DIR, f"{name}.tiff")
    tifffile.imwrite(path, arr)
    _print_url(_url(path))
    return path


def show_figure(fig, name: str = "figure", dpi: int = 150):
    """Save a matplotlib figure as PNG and print the viewer URL."""
    path = os.path.join(TEMP_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    _print_url(_url(path))
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show an image in the web viewer")
    parser.add_argument("path", nargs="?", help="Path to an image file")
    parser.add_argument("--npy", help="Path to a .npy file to display")
    parser.add_argument("--name", default="array", help="Name for saved temp files")
    args = parser.parse_args()

    if args.npy:
        data = np.load(args.npy)
        show_array(data, name=args.name)
    elif args.path:
        show(args.path)
    else:
        parser.print_help()
