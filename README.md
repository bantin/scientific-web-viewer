# Scientific Web Viewer

A lightweight browser-based viewer for images and TIFF stacks on remote machines. Designed for scientific imaging workflows where you SSH into a server and want to quickly inspect data without transferring files.

## Features

- View single images (PNG, JPEG, BMP, GIF, WebP) and TIFF stacks
- Frame-by-frame scrubbing for z-stacks and time series (slider, arrow keys, play/pause)
- Adjustable contrast (min/max sliders, auto-contrast at 2nd/98th percentile)
- Handles uint8, uint16, int16, and float32 data
- Python helpers for displaying numpy arrays and matplotlib figures
- Runs on localhost behind an SSH tunnel — no data leaves the machine

## Setup

### Requirements

Python 3.10+ with: `flask`, `numpy`, `pillow`, `tifffile`

### Install

```bash
git clone git@github.com:bantin/scientific-web-viewer.git
```

### Start the server

```bash
# Manual
./start.sh

# Or with systemd (auto-start on boot)
cp web-viewer.service ~/.config/systemd/user/
# Edit the service file to point to your install path and Python env
systemctl --user daemon-reload
systemctl --user enable --now web-viewer.service
```

### SSH tunnel

Add to your local `~/.ssh/config`:

```
Host myserver
    HostName <your-server>
    User <your-user>
    LocalForward 8089 localhost:8089
```

## Usage

### From the shell

Source `view.sh` in your `.bashrc`:

```bash
source /path/to/scientific_web_viewer/view.sh
```

Then:

```bash
view /path/to/image.tiff        # prints a clickable URL
view .                           # lists all images in current directory
```

### From Python

```python
import sys
sys.path.insert(0, "/path/to/scientific_web_viewer")
from show import show, show_array, show_figure

# Existing file
show("/path/to/stack.tiff")

# Numpy array
show_array(my_array, name="voltage_trace")

# Matplotlib figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data)
show_figure(fig, name="summary")
```

Each call prints a clickable `http://localhost:8089/view?path=...` URL.

### Directly in the browser

Navigate to `http://localhost:8089/view?path=/absolute/path/to/file.tiff`

## Claude Code integration

To let Claude Code use the viewer, add the `show-image.md` skill to your project:

```bash
mkdir -p /path/to/project/.claude/skills
ln -s /path/to/scientific_web_viewer/show-image.md /path/to/project/.claude/skills/show-image.md
```
