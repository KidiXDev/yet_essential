# yet_essential

A collection of high-performance, quality-of-life, and artist-focused nodes for ComfyUI.

## 🚀 Key Features

### 🔍 Smart Tag Autocomplete
Highly optimized autocomplete for the **YE Prompt** node.
- **Fuzzy & Prefix Matching:** Fast and flexible searching through large tag lists (e.g., Danbooru datasets).
- **Aliases Support:** Match tags via their recognized aliases.
- **Configurable:** Tweak algorithms, limits, and formatting via `config/setting.cfg`.
- **High Performance:** Uses bucketed search for near-instant results even with tens of thousands of tags.

### 🖼️ Model Previews
- **Grid View:** Beautiful, searchable grid view for easy model selection.
- **Auto-Discovery:** Automatically finds preview images for Checkpoints, LoRAs, and Diffusion models.
- **Thumbnail Caching:** Generates and caches optimized thumbnails to keep the UI snappy.
- **Flexible Patterns:** Supports patterns like `{model}.preview.png` or simply `{model}.png` in the same directory.

### 🧪 Variation Sampling (SLERP)
Unlock subtle creativity with the **YE KSampler**.
- **Variation Seed & Strength:** Interpolate between two seeds.
- **SLERP (Spherical Linear Interpolation):** Uses SLERP instead of linear interpolation for high-quality, stable noise blending in high-dimensional space.

### 🎨 Modular PostFX Pipeline
A professional-grade post-processing system with stackable effects.
- **Adjustments:** Brightness, Contrast, Saturation, and Sharpness (Unsharp Mask).
- **Styling:**
    - **Bloom:** Creamy highlight glows with threshold and radius control.
    - **Chromatic Aberration:** Directional color fringing with angle control.
    - **Film Grain:** Seed-based procedural grain for a cinematic Look.
    - **Vignette:** Adjustable strength and softness.
- **Non-Destructive:** Build effect pipelines and apply them late in your workflow.

### 🛠️ Essential Utilities
- **YE Empty Latent Image:** Includes industry-standard presets for SD1.5 and SDXL (Square, Landscape, Portrait).
- **YE Image Upscale:** Tiled, memory-safe upscaling using external models.
- **Streamlined Loaders:** Simplified loaders for Checkpoints, UNETs, and LoRAs.

## 🧩 Compatibility

- **Node 2.0 Support:** Fully compatible with ComfyUI Node 2.0 (V2 UI) features, working perfectly whether the new UI is **enabled or disabled**.

## 📦 Installation

Simply clone this repository into your `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/KidiXDev/yet_essential
```

## ⚙️ Configuration

Settings can be managed in `config/setting.cfg` after the first run.
Place your tag CSV files in `config/tag/` to enable autocomplete for specific datasets.

## 📄 License

Apache Software License 2.0
