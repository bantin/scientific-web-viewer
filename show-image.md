# show-image

Display images, numpy arrays, and matplotlib figures in the browser via the local web viewer.

A Flask server runs on localhost:8089 and renders images (including TIFF stacks with a z-slider and contrast controls) in the browser. The user accesses it through an SSH tunnel.

## When to use

Use this whenever you need to show the user an image, a plot, a numpy array, or a TIFF stack. This is the primary way to visualize data on this machine.

## How to display data

The helper lives at `/mnt/ssd2tb1/bantin/scientific_web_viewer/show.py`. Activate the conda env first.

### Show an existing image file

```bash
source ~/miniforge3/bin/activate dendrites-py312
python /mnt/ssd2tb1/bantin/scientific_web_viewer/show.py /path/to/image.tiff
```

### Show a numpy array (inline Python)

```bash
source ~/miniforge3/bin/activate dendrites-py312
python -c "
import numpy as np
import sys
sys.path.insert(0, '/mnt/ssd2tb1/bantin/scientific_web_viewer')
from show import show_array

data = np.load('/path/to/data.npy')
show_array(data, name='my_data')
"
```

### Show a matplotlib figure

```bash
source ~/miniforge3/bin/activate dendrites-py312
python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/mnt/ssd2tb1/bantin/scientific_web_viewer')
from show import show_figure

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
show_figure(fig, name='my_plot')
"
```

### In a longer script

When writing a Python script that produces visualizations, add this near the top:

```python
import sys
sys.path.insert(0, "/mnt/ssd2tb1/bantin/scientific_web_viewer")
from show import show, show_array, show_figure
```

Then call `show_array(arr)`, `show_figure(fig)`, or `show("/path/to/file.tiff")` wherever you want to display something. Each call prints a clickable URL.

## Important notes

- The server must be running. Start it with `/mnt/ssd2tb1/bantin/scientific_web_viewer/start.sh` if it's not.
- The user must have an SSH tunnel forwarding port 8089 (they typically do).
- Temp files go to `/tmp/web_viewer_temp/` and can be cleaned up freely.
- For TIFF stacks: the viewer supports scrubbing through frames, play/pause, and contrast adjustment.
- For standard images (PNG, JPG): they display directly in the browser.
- Always use `matplotlib.use('Agg')` before importing pyplot, since there's no display server.
