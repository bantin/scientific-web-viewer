"""
Lightweight image/TIFF/NWB viewer served over HTTP.
Run behind an SSH tunnel: ssh -L 8089:localhost:8089 bantin@129.236.160.85
"""

import io
import os
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import h5py
from flask import Flask, request, jsonify, send_file, abort, render_template_string
from PIL import Image
import tifffile

sys.path.insert(0, "/mnt/ssd2tb1/bantin/masknmf-toolbox")

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".nwb",
}

# ---------------------------------------------------------------------------
# PMD cache — keeps the most recently loaded PMDArray in memory so frame
# requests don't re-deserialize the sparse matrices every time.
# ---------------------------------------------------------------------------
_pmd_cache = {}  # (path, name) -> PMDArray


def _get_pmd_array(filepath, name):
    key = (filepath, name)
    if key in _pmd_cache:
        return _pmd_cache[key]
    from dendro.utils.data import load_pmd_from_nwb
    result = load_pmd_from_nwb(filepath, name=name)
    if result is None:
        return None
    pmd_arr, _ = result
    _pmd_cache.clear()
    _pmd_cache[key] = pmd_arr
    return pmd_arr


# ---------------------------------------------------------------------------
# NWB dataset classification
# ---------------------------------------------------------------------------

def _classify_nwb_datasets(filepath):
    items = []
    with h5py.File(filepath, "r") as f:
        acq = f.get("acquisition")
        if acq is None:
            return items
        for name in sorted(acq.keys()):
            grp = acq[name]
            if not isinstance(grp, h5py.Group):
                continue

            if "pmd" in grp:
                pmd_grp = grp["pmd"]
                shape_ds = pmd_grp.get("shape")
                if shape_ds is not None:
                    shape = tuple(int(x) for x in shape_ds[()])
                else:
                    v = pmd_grp.get("v")
                    shape = (int(v.shape[1]),) if v is not None else ()
                items.append({
                    "name": name,
                    "kind": "pmd",
                    "shape": shape,
                    "dtype": "float32",
                    "extra": "",
                })
                for sub in ("mean_img", "var_img"):
                    ds = pmd_grp.get(sub)
                    if ds is not None:
                        items.append({
                            "name": f"{name}/pmd/{sub}",
                            "kind": "image",
                            "shape": ds.shape,
                            "dtype": str(ds.dtype),
                            "extra": "",
                        })
                continue

            ds = grp.get("data")
            if ds is None:
                continue
            ndim = len(ds.shape)
            if ndim == 3:
                kind = "video"
                chunks_str = str(ds.chunks) if ds.chunks else "contiguous"
                comp = ds.compression or "none"
                extra = f"{chunks_str}, {comp}"
            elif ndim == 2:
                kind = "image"
                extra = ""
            elif ndim == 1:
                kind = "signal"
                extra = ""
            else:
                kind = "unknown"
                extra = ""
            items.append({
                "name": name,
                "kind": kind,
                "shape": ds.shape,
                "dtype": str(ds.dtype),
                "extra": extra,
            })
    return items


# ---------------------------------------------------------------------------
# Shared CSS (used by all templates)
# ---------------------------------------------------------------------------

SHARED_CSS = """
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
"""

# ---------------------------------------------------------------------------
# NWB Table of Contents template
# ---------------------------------------------------------------------------

NWB_TOC_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ filename }} — NWB Viewer</title>
<style>
""" + SHARED_CSS + """
  .content {
    width: 100%;
    max-width: 1000px;
    padding: 24px;
  }
  h2 {
    font-size: 16px;
    margin: 24px 0 12px 0;
    color: var(--highlight);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th {
    text-align: left;
    padding: 8px 12px;
    background: var(--accent);
    border-bottom: 2px solid var(--highlight);
  }
  td {
    padding: 8px 12px;
    border-bottom: 1px solid #ffffff11;
  }
  tr:hover td { background: var(--surface); }
  a { color: var(--highlight); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
  }
  .tag-video { background: #0f3460; color: #4cc9f0; }
  .tag-pmd { background: #1a0f3e; color: #b794f4; }
  .tag-image { background: #0f3e2a; color: #69db7c; }
  .tag-signal { background: #3e2a0f; color: #ffd43b; }
  .dim { opacity: 0.5; }
</style>
</head>
<body>
<header>
  <h1>NWB Viewer</h1>
  <span class="path">{{ filepath }}</span>
</header>
<div class="content">
  <h2>Acquisitions</h2>
  <table>
    <thead>
      <tr><th>Name</th><th>Type</th><th>Shape</th><th>Dtype</th><th>Details</th><th></th></tr>
    </thead>
    <tbody>
    {% for d in datasets %}
      <tr>
        <td><code>{{ d.name }}</code></td>
        <td>
          {% if d.kind == "video" %}<span class="tag tag-video">video</span>
          {% elif d.kind == "pmd" %}<span class="tag tag-pmd">video (PMD)</span>
          {% elif d.kind == "image" %}<span class="tag tag-image">image</span>
          {% elif d.kind == "signal" %}<span class="tag tag-signal">signal</span>
          {% else %}<span class="tag">{{ d.kind }}</span>{% endif %}
        </td>
        <td><code>{{ d.shape | join(' x ') }}</code></td>
        <td><code>{{ d.dtype }}</code></td>
        <td class="dim">{{ d.extra }}</td>
        <td>
          {% if d.kind in ("video", "pmd", "image") %}
            <a href="/nwb/view?path={{ filepath_encoded }}&dataset={{ d.name | urlencode }}">View</a>
          {% else %}
            <span class="dim">—</span>
          {% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# NWB lazy frame viewer template
# ---------------------------------------------------------------------------

NWB_VIEWER_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ dataset_name }} — NWB Viewer</title>
<style>
""" + SHARED_CSS + """
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
  .viewer canvas {
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
  #loading-indicator {
    position: fixed;
    top: 80px;
    right: 24px;
    background: var(--highlight);
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-family: monospace;
    display: none;
  }
</style>
</head>
<body>
<header>
  <h1>NWB Viewer</h1>
  <span class="path">{{ filepath }} / {{ dataset_name }}</span>
</header>
<div class="info-bar">
  <span>Shape: {{ shape }}</span>
  <span>Dtype: {{ dtype }}</span>
  {% if n_frames > 1 %}<span>Frames: {{ n_frames }}</span>{% endif %}
</div>
<div id="loading-indicator">Loading...</div>

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
    <input type="range" id="min-slider" min="0" max="65535" value="0" step="any">
    <label>Max: <span id="max-val">65535</span></label>
    <input type="range" id="max-slider" min="0" max="65535" value="65535" step="any">
    <button id="auto-contrast">Auto</button>
    <button id="reset-contrast">Reset</button>
  </div>
</div>

<script>
const NWB_PATH = "{{ filepath }}";
const DATASET = "{{ dataset_name }}";
const N_FRAMES = {{ n_frames }};
const WIDTH = {{ width }};
const HEIGHT = {{ height }};
const DTYPE = "{{ dtype }}";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
canvas.width = WIDTH;
canvas.height = HEIGHT;

const FRAME_SIZE = WIDTH * HEIGHT;
const MAX_CACHE = 200;
const PREFETCH_AHEAD = 15;
const PREFETCH_BEHIND = 5;

const frameCache = new Map();
const pendingFetches = new Set();
let currentFrame = 0;
let playing = false;
let fps = 30;
let animId = null;
let cmin = 0;
let cmax = 1;
let dataMin = 0;
let dataMax = 1;
let contrastInitialized = false;
const loadingEl = document.getElementById("loading-indicator");

function createTypedArray(buf) {
  if (DTYPE === "uint8") return new Uint8Array(buf);
  if (DTYPE === "uint16") return new Uint16Array(buf);
  if (DTYPE === "int16") return new Int16Array(buf);
  return new Float32Array(buf);
}

function frameUrl(idx) {
  return `/nwb/frame?path=${encodeURIComponent(NWB_PATH)}&dataset=${encodeURIComponent(DATASET)}&frame=${idx}`;
}

async function fetchFrame(idx) {
  if (idx < 0 || idx >= N_FRAMES) return;
  if (frameCache.has(idx) || pendingFetches.has(idx)) return;
  pendingFetches.add(idx);
  loadingEl.style.display = "block";
  try {
    const resp = await fetch(frameUrl(idx));
    if (!resp.ok) return;
    const buf = await resp.arrayBuffer();
    frameCache.set(idx, createTypedArray(buf));
    evictCache(idx);
  } finally {
    pendingFetches.delete(idx);
    if (pendingFetches.size === 0) loadingEl.style.display = "none";
  }
}

function evictCache(centerIdx) {
  if (frameCache.size <= MAX_CACHE) return;
  let furthest = -1, furthestDist = -1;
  for (const k of frameCache.keys()) {
    const dist = Math.abs(k - centerIdx);
    if (dist > furthestDist) { furthestDist = dist; furthest = k; }
  }
  if (furthest >= 0) frameCache.delete(furthest);
}

function prefetch(idx) {
  for (let i = 1; i <= PREFETCH_AHEAD; i++) {
    const fi = idx + i;
    if (fi < N_FRAMES) fetchFrame(fi);
  }
  for (let i = 1; i <= PREFETCH_BEHIND; i++) {
    const fi = idx - i;
    if (fi >= 0) fetchFrame(fi);
  }
}

function renderFromCache(idx) {
  const arr = frameCache.get(idx);
  if (!arr) return;
  const imgData = ctx.createImageData(WIDTH, HEIGHT);
  const range = cmax - cmin || 1;
  for (let i = 0; i < FRAME_SIZE; i++) {
    let v = (arr[i] - cmin) / range;
    v = Math.max(0, Math.min(1, v));
    const byte = Math.round(v * 255);
    imgData.data[i * 4] = byte;
    imgData.data[i * 4 + 1] = byte;
    imgData.data[i * 4 + 2] = byte;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

async function showFrame(idx) {
  currentFrame = idx;
  const label = document.getElementById("frame-label");
  if (label) label.textContent = `Frame ${idx} / ${N_FRAMES - 1}`;

  if (frameCache.has(idx)) {
    renderFromCache(idx);
  } else {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = '#e94560';
    ctx.font = '14px monospace';
    ctx.fillText(`Loading frame ${idx}...`, 10, HEIGHT / 2);
    await fetchFrame(idx);
    if (currentFrame === idx) renderFromCache(idx);
  }

  if (!contrastInitialized && frameCache.has(idx)) {
    autoContrast();
    contrastInitialized = true;
    renderFromCache(idx);
  }

  prefetch(idx);
}

function autoContrast() {
  const arr = frameCache.get(currentFrame);
  if (!arr) return;
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.02)];
  const hi = sorted[Math.floor(sorted.length * 0.98)];
  dataMin = sorted[0];
  dataMax = sorted[sorted.length - 1];
  cmin = lo;
  cmax = hi > lo ? hi : lo + 1;

  const minSlider = document.getElementById("min-slider");
  const maxSlider = document.getElementById("max-slider");
  if (minSlider) {
    minSlider.min = dataMin;
    minSlider.max = dataMax;
    maxSlider.min = dataMin;
    maxSlider.max = dataMax;
    minSlider.value = cmin;
    maxSlider.value = cmax;
    document.getElementById("min-val").textContent = Math.round(cmin * 100) / 100;
    document.getElementById("max-val").textContent = Math.round(cmax * 100) / 100;
  }
}

// --- Controls ---

const slider = document.getElementById("frame-slider");
if (slider) {
  slider.addEventListener("input", (e) => {
    const idx = parseInt(e.target.value);
    showFrame(idx);
  });
}

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
  function step(ts) {
    if (!playing) return;
    if (ts - lastTime >= 1000 / fps) {
      lastTime = ts;
      const next = (currentFrame + 1) % N_FRAMES;
      if (frameCache.has(next)) {
        currentFrame = next;
        renderFromCache(currentFrame);
        if (slider) slider.value = currentFrame;
        const label = document.getElementById("frame-label");
        if (label) label.textContent = `Frame ${currentFrame} / ${N_FRAMES - 1}`;
        prefetch(currentFrame);
      }
    }
    animId = requestAnimationFrame(step);
  }
  animId = requestAnimationFrame(step);
}

function stopAnimation() {
  if (animId) cancelAnimationFrame(animId);
}

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

const minSlider = document.getElementById("min-slider");
const maxSlider = document.getElementById("max-slider");
if (minSlider) {
  minSlider.addEventListener("input", (e) => {
    cmin = parseFloat(e.target.value);
    document.getElementById("min-val").textContent = Math.round(cmin * 100) / 100;
    renderFromCache(currentFrame);
  });
  maxSlider.addEventListener("input", (e) => {
    cmax = parseFloat(e.target.value);
    document.getElementById("max-val").textContent = Math.round(cmax * 100) / 100;
    renderFromCache(currentFrame);
  });
}
document.getElementById("auto-contrast")?.addEventListener("click", () => {
  autoContrast();
  renderFromCache(currentFrame);
});
document.getElementById("reset-contrast")?.addEventListener("click", () => {
  cmin = dataMin;
  cmax = dataMax;
  if (minSlider) { minSlider.value = cmin; document.getElementById("min-val").textContent = Math.round(cmin * 100) / 100; }
  if (maxSlider) { maxSlider.value = cmax; document.getElementById("max-val").textContent = Math.round(cmax * 100) / 100; }
  renderFromCache(currentFrame);
});

document.addEventListener("keydown", (e) => {
  if (N_FRAMES <= 1) return;
  if (e.key === "ArrowRight") {
    const next = (currentFrame + 1) % N_FRAMES;
    showFrame(next);
    if (slider) slider.value = next;
  } else if (e.key === "ArrowLeft") {
    const prev = (currentFrame - 1 + N_FRAMES) % N_FRAMES;
    showFrame(prev);
    if (slider) slider.value = prev;
  } else if (e.key === " ") {
    e.preventDefault();
    playBtn?.click();
  }
});

showFrame(0);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Original TIFF/image viewer template (unchanged)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ filename }} — Image Viewer</title>
<style>
""" + SHARED_CSS + """
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

const slider = document.getElementById("frame-slider");
if (slider) {
  slider.addEventListener("input", (e) => {
    renderFrame(parseInt(e.target.value));
  });
}

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_to_viewable(data):
    """Convert array data to a dtype the JS viewer can handle directly."""
    if data.dtype == np.uint8 or data.dtype == np.uint16:
        return data, str(data.dtype)
    if data.dtype == np.int16:
        return data, "int16"
    return data.astype(np.float32), "float32"


# ---------------------------------------------------------------------------
# Routes — existing TIFF/image viewer
# ---------------------------------------------------------------------------

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
    <pre>view /path/to/file.nwb</pre>
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

    # NWB files go to the table-of-contents page
    if ext == ".nwb":
        datasets = _classify_nwb_datasets(filepath)
        return render_template_string(
            NWB_TOC_TEMPLATE,
            filename=Path(filepath).name,
            filepath=filepath,
            filepath_encoded=quote(filepath),
            datasets=datasets,
        )

    filename = Path(filepath).name

    if ext in (".tif", ".tiff"):
        data = tifffile.imread(filepath)
    else:
        img = Image.open(filepath)
        data = np.array(img)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3 and data.shape[-1] in (3, 4):
        if ext not in (".tif", ".tiff"):
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
        if data.shape[0] in (3, 4) and data.shape[0] < data.shape[1]:
            data = data[0:1, :, :]
        else:
            data = np.mean(data, axis=-1, keepdims=False)[np.newaxis, :, :]
    elif data.ndim == 3:
        pass
    elif data.ndim > 3:
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


# ---------------------------------------------------------------------------
# Routes — NWB viewer
# ---------------------------------------------------------------------------

def _nwb_dataset_info(filepath, dataset_name):
    """Get shape/dtype/n_frames for an NWB dataset without loading pixel data."""
    with h5py.File(filepath, "r") as f:
        acq = f["acquisition"]
        grp = acq[dataset_name] if "/" not in dataset_name else f[f"acquisition/{dataset_name}"]

        if isinstance(grp, h5py.Dataset):
            ds = grp
        elif "pmd" in grp:
            shape_ds = grp["pmd"].get("shape")
            if shape_ds is not None:
                shape = tuple(int(x) for x in shape_ds[()])
            else:
                v = grp["pmd"]["v"]
                shape = (int(v.shape[1]),)
            return {
                "shape": shape,
                "dtype": "float32",
                "n_frames": shape[0] if len(shape) == 3 else 1,
                "height": shape[1] if len(shape) >= 2 else 1,
                "width": shape[2] if len(shape) >= 3 else 1,
                "kind": "pmd",
            }
        elif "data" in grp:
            ds = grp["data"]
        else:
            abort(404, f"No viewable data in {dataset_name}")
            return

        shape = ds.shape
        dtype = str(ds.dtype)
        if len(shape) == 3:
            return {"shape": shape, "dtype": dtype, "n_frames": shape[0],
                    "height": shape[1], "width": shape[2], "kind": "video"}
        elif len(shape) == 2:
            return {"shape": shape, "dtype": dtype, "n_frames": 1,
                    "height": shape[0], "width": shape[1], "kind": "image"}
        else:
            abort(400, f"Unsupported shape {shape}")


@app.route("/nwb/view")
def nwb_view():
    filepath = request.args.get("path", "")
    dataset_name = request.args.get("dataset", "")
    if not filepath or not dataset_name:
        abort(400)
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        abort(404)

    info = _nwb_dataset_info(filepath, dataset_name)
    shape_str = " x ".join(str(s) for s in info["shape"])

    dtype_str = info["dtype"]
    if dtype_str not in ("uint8", "uint16", "int16", "float32"):
        dtype_str = "float32"

    return render_template_string(
        NWB_VIEWER_TEMPLATE,
        filepath=filepath,
        dataset_name=dataset_name,
        shape=shape_str,
        dtype=dtype_str,
        n_frames=info["n_frames"],
        width=info["width"],
        height=info["height"],
    )


@app.route("/nwb/frame")
def nwb_frame():
    filepath = request.args.get("path", "")
    dataset_name = request.args.get("dataset", "")
    frame = request.args.get("frame", type=int)
    if not filepath or not dataset_name or frame is None:
        abort(400)
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        abort(404)

    # Determine the base acquisition name (before any sub-path like /pmd/mean_img)
    base_name = dataset_name.split("/")[0]

    # Check if this is a PMD dataset
    with h5py.File(filepath, "r") as f:
        grp = f[f"acquisition/{base_name}"]
        is_pmd = "pmd" in grp

        if is_pmd and "/" in dataset_name:
            # Sub-dataset like camera_1_pmd/pmd/mean_img — read directly
            ds = f[f"acquisition/{dataset_name}"]
            data = ds[()]
        elif is_pmd:
            # Full PMD reconstruction — use masknmf
            pmd = _get_pmd_array(filepath, base_name)
            if pmd is None:
                abort(404, f"Could not load PMD for {base_name}")
            frame = max(0, min(frame, pmd.shape[0] - 1))
            data = pmd[frame]
        elif "data" in grp:
            ds = grp["data"]
            if len(ds.shape) == 3:
                frame = max(0, min(frame, ds.shape[0] - 1))
                data = ds[frame]
            elif len(ds.shape) == 2:
                data = ds[()]
            else:
                abort(400)
        else:
            abort(404)

    data = np.ascontiguousarray(data)
    if data.dtype not in (np.uint8, np.uint16, np.int16, np.float32):
        data = data.astype(np.float32)

    buf = io.BytesIO(data.tobytes())
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    print(f"Starting image viewer on http://{args.host}:{args.port}")
    print("Use: view /path/to/image.tiff")
    print("     view /path/to/file.nwb")
    app.run(host=args.host, port=args.port, debug=False)
