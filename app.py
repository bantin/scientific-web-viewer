"""
Lightweight image/TIFF stack viewer served over HTTP.
Run behind an SSH tunnel: ssh -L 8089:localhost:8089 bantin@129.236.160.85
"""

import io
import os
import base64
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, send_file, abort, render_template_string
from PIL import Image
import tifffile

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp",
}

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ filename }} — Image Viewer</title>
<style>
  :root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --text: #e0e0e0;
    --accent: #0f3460;
    --highlight: #e94560;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header {
    width: 100%;
    padding: 12px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--accent);
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  header h1 {
    font-size: 14px;
    font-weight: 500;
    opacity: 0.7;
  }
  header .path {
    font-size: 13px;
    font-family: monospace;
    color: var(--highlight);
    word-break: break-all;
  }
  .info-bar {
    width: 100%;
    padding: 8px 24px;
    background: var(--accent);
    font-size: 13px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }
  .info-bar span { opacity: 0.8; }
  .viewer {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
    width: 100%;
  }
  .viewer img, .viewer canvas {
    max-width: 95vw;
    max-height: 75vh;
    image-rendering: pixelated;
    border: 1px solid var(--accent);
  }
  .slider-container {
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 600px;
  }
  .slider-container input[type=range] {
    flex: 1;
    accent-color: var(--highlight);
  }
  .slider-container label {
    font-size: 14px;
    font-family: monospace;
    min-width: 80px;
  }
  .controls {
    display: flex;
    gap: 12px;
    margin-top: 12px;
    flex-wrap: wrap;
    justify-content: center;
  }
  .controls button {
    background: var(--accent);
    color: var(--text);
    border: 1px solid #ffffff22;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
  }
  .controls button:hover { background: var(--highlight); }
  .controls button.active { background: var(--highlight); }
  #contrast-controls {
    margin-top: 12px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    justify-content: center;
  }
  #contrast-controls input[type=range] {
    width: 120px;
    accent-color: var(--highlight);
  }
  #contrast-controls label { font-size: 12px; font-family: monospace; }
</style>
</head>
<body>
<header>
  <h1>Image Viewer</h1>
  <span class="path">{{ filepath }}</span>
</header>
<div class="info-bar">
  <span>Shape: {{ shape }}</span>
  <span>Dtype: {{ dtype }}</span>
  {% if n_frames > 1 %}<span>Frames: {{ n_frames }}</span>{% endif %}
</div>

<div class="viewer">
  <canvas id="canvas"></canvas>

  {% if n_frames > 1 %}
  <div class="slider-container">
    <label id="frame-label">Frame 0 / {{ n_frames - 1 }}</label>
    <input type="range" id="frame-slider" min="0" max="{{ n_frames - 1 }}" value="0">
  </div>
  <div class="controls">
    <button id="play-btn">&#9654; Play</button>
    <button id="speed-down">Slower</button>
    <button id="speed-up">Faster</button>
    <span id="fps-label" style="font-size:12px;font-family:monospace;align-self:center;">30 fps</span>
  </div>
  {% endif %}

  <div id="contrast-controls">
    <label>Min: <span id="min-val">0</span></label>
    <input type="range" id="min-slider" min="0" max="65535" value="0">
    <label>Max: <span id="max-val">65535</span></label>
    <input type="range" id="max-slider" min="0" max="65535" value="65535">
    <button id="auto-contrast">Auto</button>
    <button id="reset-contrast">Reset</button>
  </div>
</div>

<script>
const DATA_URL = "{{ data_url }}";
const N_FRAMES = {{ n_frames }};
const WIDTH = {{ width }};
const HEIGHT = {{ height }};
const DTYPE = "{{ dtype }}";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
canvas.width = WIDTH;
canvas.height = HEIGHT;

let rawFrames = null;
let currentFrame = 0;
let playing = false;
let fps = 30;
let animId = null;
let cmin = 0;
let cmax = 65535;
let dataMin = 0;
let dataMax = 65535;

async function loadData() {
  const resp = await fetch(DATA_URL);
  const buf = await resp.arrayBuffer();
  if (DTYPE === "uint8") {
    rawFrames = new Uint8Array(buf);
    dataMax = 255;
  } else if (DTYPE === "uint16") {
    rawFrames = new Uint16Array(buf);
    dataMax = 65535;
  } else if (DTYPE === "int16") {
    rawFrames = new Int16Array(buf);
    dataMax = 32767;
    dataMin = -32768;
  } else if (DTYPE === "float32") {
    rawFrames = new Float32Array(buf);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < rawFrames.length; i++) {
      if (rawFrames[i] < mn) mn = rawFrames[i];
      if (rawFrames[i] > mx) mx = rawFrames[i];
    }
    dataMin = mn;
    dataMax = mx;
  } else {
    rawFrames = new Float32Array(buf);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < rawFrames.length; i++) {
      if (rawFrames[i] < mn) mn = rawFrames[i];
      if (rawFrames[i] > mx) mx = rawFrames[i];
    }
    dataMin = mn;
    dataMax = mx;
  }

  // Set slider ranges
  const minSlider = document.getElementById("min-slider");
  const maxSlider = document.getElementById("max-slider");
  if (minSlider) {
    minSlider.min = Math.floor(dataMin);
    minSlider.max = Math.ceil(dataMax);
    maxSlider.min = Math.floor(dataMin);
    maxSlider.max = Math.ceil(dataMax);
    minSlider.value = Math.floor(dataMin);
    maxSlider.value = Math.ceil(dataMax);
  }
  cmin = dataMin;
  cmax = dataMax;

  autoContrast();
  renderFrame(0);
}

function autoContrast() {
  if (!rawFrames) return;
  const frameSize = WIDTH * HEIGHT;
  const offset = currentFrame * frameSize;
  const frame = rawFrames.subarray(offset, offset + frameSize);
  let sorted = Array.from(frame).sort((a, b) => a - b);
  let lo = sorted[Math.floor(sorted.length * 0.02)];
  let hi = sorted[Math.floor(sorted.length * 0.98)];
  if (hi <= lo) hi = lo + 1;
  cmin = lo;
  cmax = hi;
  const minSlider = document.getElementById("min-slider");
  const maxSlider = document.getElementById("max-slider");
  if (minSlider) {
    minSlider.value = cmin;
    maxSlider.value = cmax;
    document.getElementById("min-val").textContent = Math.round(cmin);
    document.getElementById("max-val").textContent = Math.round(cmax);
  }
}

function renderFrame(idx) {
  if (!rawFrames) return;
  currentFrame = idx;
  const frameSize = WIDTH * HEIGHT;
  const offset = idx * frameSize;
  const imgData = ctx.createImageData(WIDTH, HEIGHT);
  const range = cmax - cmin || 1;
  for (let i = 0; i < frameSize; i++) {
    let v = (rawFrames[offset + i] - cmin) / range;
    v = Math.max(0, Math.min(1, v));
    const byte = Math.round(v * 255);
    imgData.data[i * 4] = byte;
    imgData.data[i * 4 + 1] = byte;
    imgData.data[i * 4 + 2] = byte;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);

  const label = document.getElementById("frame-label");
  if (label) label.textContent = `Frame ${idx} / ${N_FRAMES - 1}`;
}

// Slider
const slider = document.getElementById("frame-slider");
if (slider) {
  slider.addEventListener("input", (e) => {
    renderFrame(parseInt(e.target.value));
  });
}

// Play/pause
const playBtn = document.getElementById("play-btn");
if (playBtn) {
  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
    playBtn.classList.toggle("active", playing);
    if (playing) startAnimation();
    else stopAnimation();
  });
}

function startAnimation() {
  let lastTime = 0;
  const interval = () => 1000 / fps;
  function step(ts) {
    if (!playing) return;
    if (ts - lastTime >= interval()) {
      lastTime = ts;
      currentFrame = (currentFrame + 1) % N_FRAMES;
      renderFrame(currentFrame);
      if (slider) slider.value = currentFrame;
    }
    animId = requestAnimationFrame(step);
  }
  animId = requestAnimationFrame(step);
}

function stopAnimation() {
  if (animId) cancelAnimationFrame(animId);
}

// Speed controls
const speedDown = document.getElementById("speed-down");
const speedUp = document.getElementById("speed-up");
const fpsLabel = document.getElementById("fps-label");
if (speedDown) {
  speedDown.addEventListener("click", () => {
    fps = Math.max(1, fps / 2);
    fpsLabel.textContent = `${fps} fps`;
  });
}
if (speedUp) {
  speedUp.addEventListener("click", () => {
    fps = Math.min(120, fps * 2);
    fpsLabel.textContent = `${fps} fps`;
  });
}

// Contrast controls
const minSlider = document.getElementById("min-slider");
const maxSlider = document.getElementById("max-slider");
if (minSlider) {
  minSlider.addEventListener("input", (e) => {
    cmin = parseFloat(e.target.value);
    document.getElementById("min-val").textContent = Math.round(cmin);
    renderFrame(currentFrame);
  });
  maxSlider.addEventListener("input", (e) => {
    cmax = parseFloat(e.target.value);
    document.getElementById("max-val").textContent = Math.round(cmax);
    renderFrame(currentFrame);
  });
}
document.getElementById("auto-contrast")?.addEventListener("click", () => {
  autoContrast();
  renderFrame(currentFrame);
});
document.getElementById("reset-contrast")?.addEventListener("click", () => {
  cmin = dataMin;
  cmax = dataMax;
  if (minSlider) { minSlider.value = cmin; document.getElementById("min-val").textContent = Math.round(cmin); }
  if (maxSlider) { maxSlider.value = cmax; document.getElementById("max-val").textContent = Math.round(cmax); }
  renderFrame(currentFrame);
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (N_FRAMES <= 1) return;
  if (e.key === "ArrowRight") {
    currentFrame = (currentFrame + 1) % N_FRAMES;
    renderFrame(currentFrame);
    if (slider) slider.value = currentFrame;
  } else if (e.key === "ArrowLeft") {
    currentFrame = (currentFrame - 1 + N_FRAMES) % N_FRAMES;
    renderFrame(currentFrame);
    if (slider) slider.value = currentFrame;
  } else if (e.key === " ") {
    e.preventDefault();
    playBtn?.click();
  }
});

loadData();
</script>
</body>
</html>
"""


def normalize_to_viewable(data):
    """Convert array data to a dtype the JS viewer can handle directly."""
    if data.dtype == np.uint8 or data.dtype == np.uint16:
        return data, str(data.dtype)
    if data.dtype == np.int16:
        return data, "int16"
    return data.astype(np.float32), "float32"


@app.route("/")
def index():
    return """
    <html><head><title>Image Viewer</title>
    <style>
      body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 40px; }
      h1 { color: #e94560; }
      code { background: #16213e; padding: 2px 6px; border-radius: 3px; }
      pre { background: #16213e; padding: 16px; border-radius: 6px; margin: 12px 0; }
    </style>
    </head><body>
    <h1>Image Viewer</h1>
    <p>Use the <code>view</code> command in your terminal to open an image:</p>
    <pre>view /path/to/image.tiff</pre>
    <p>Or navigate directly:</p>
    <pre>http://localhost:8089/view?path=/path/to/image.tiff</pre>
    </body></html>
    """


@app.route("/view")
def view_image():
    filepath = request.args.get("path", "")
    if not filepath:
        abort(400, "Missing ?path= parameter")

    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        abort(404, f"File not found: {filepath}")

    ext = Path(filepath).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        abort(400, f"Unsupported file type: {ext}")

    filename = Path(filepath).name

    if ext in (".tif", ".tiff"):
        data = tifffile.imread(filepath)
    else:
        img = Image.open(filepath)
        data = np.array(img)

    # Ensure at least 2D
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # single frame
    elif data.ndim == 3 and data.shape[-1] in (3, 4):
        # RGB/RGBA — convert to grayscale for the raw viewer, or handle separately
        if ext not in (".tif", ".tiff"):
            # For non-TIFF RGB images, just serve the file directly
            return render_template_string(
                """<!DOCTYPE html><html><head><title>{{ filename }}</title>
                <style>
                  body { background: #1a1a2e; display: flex; flex-direction: column;
                         align-items: center; justify-content: center; min-height: 100vh;
                         font-family: monospace; color: #e0e0e0; margin: 0; }
                  img { max-width: 95vw; max-height: 90vh; }
                  .path { color: #e94560; font-size: 13px; margin: 12px; word-break: break-all; }
                </style></head><body>
                <div class="path">{{ filepath }}</div>
                <img src="/raw?path={{ filepath }}">
                </body></html>""",
                filename=filename,
                filepath=filepath,
            )
        # For multi-channel TIFFs, take first channel or treat as stack
        if data.shape[0] in (3, 4) and data.shape[0] < data.shape[1]:
            # Likely (C, H, W) — take first channel
            data = data[0:1, :, :]
        else:
            # (H, W, C) — convert to gray
            data = np.mean(data, axis=-1, keepdims=False)[np.newaxis, :, :]
    elif data.ndim == 3:
        pass  # (T, H, W) stack — good as is
    elif data.ndim > 3:
        # Flatten extra dims into the time axis
        shape = data.shape
        data = data.reshape(-1, shape[-2], shape[-1])

    data, dtype_str = normalize_to_viewable(data)
    n_frames, height, width = data.shape
    shape_str = " x ".join(str(s) for s in data.shape)

    return render_template_string(
        HTML_TEMPLATE,
        filename=filename,
        filepath=filepath,
        shape=shape_str,
        dtype=dtype_str,
        n_frames=n_frames,
        width=width,
        height=height,
        data_url=f"/data?path={filepath}",
    )


@app.route("/data")
def serve_data():
    filepath = request.args.get("path", "")
    if not filepath or not os.path.isfile(filepath):
        abort(404)

    ext = Path(filepath).suffix.lower()
    if ext in (".tif", ".tiff"):
        data = tifffile.imread(filepath)
    else:
        data = np.array(Image.open(filepath))

    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3 and data.shape[-1] in (3, 4):
        if ext in (".tif", ".tiff"):
            if data.shape[0] in (3, 4) and data.shape[0] < data.shape[1]:
                data = data[0:1, :, :]
            else:
                data = np.mean(data, axis=-1, keepdims=False)[np.newaxis, :, :]
    elif data.ndim > 3:
        shape = data.shape
        data = data.reshape(-1, shape[-2], shape[-1])

    data, _ = normalize_to_viewable(data)
    buf = io.BytesIO(data.tobytes())
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream")


@app.route("/raw")
def serve_raw():
    filepath = request.args.get("path", "")
    if not filepath or not os.path.isfile(filepath):
        abort(404)
    return send_file(filepath)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    print(f"Starting image viewer on http://{args.host}:{args.port}")
    print("Use: view /path/to/image.tiff")
    app.run(host=args.host, port=args.port, debug=False)
