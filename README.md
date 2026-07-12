# yet_essential

A collection of high-performance, quality-of-life, and artist-focused nodes for ComfyUI.

## 🚀 Key Features

### 🔍 Smart Tag Autocomplete
Highly optimized autocomplete for the **YE Prompt** node.
- **ComfyUI Node 2.0 Ready:** Fully supports the new ComfyUI Node 2.0 frontend interface.
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

### 🤖 LLM Prompt Suite
Generate and enhance prompts using Large Language Models directly inside ComfyUI.
- **Multi-Provider Support:** Seamlessly connect to API providers like **OpenRouter**, **NanoGPT**, or any standard **OpenAI-Compatible** API.
- **Asynchronous Model List:** The UI automatically fetches and lists available models from your provider in real-time.
- **Dynamic Interface:** Inputs adapt dynamically to show only the relevant configuration options for your selected provider.
- **Chat & Pipeline Builder:** Chain messages (system, user, assistant) and build structured workflows using dynamic chat inputs.
- **Ready-Made Templates:** Instantly load visual and anime prompt-engineering presets (stored in `config/prompt.json`):
    - **Prompt Creator (Flux/SDXL):** Expand simple user concepts into highly detailed scene descriptions.
    - **Booru Tag Prompt Creator:** Convert simple phrases into Danbooru-style tag lists.
    - **Prompt Enhancer:** Embellish and refine existing prompts with details, lighting, and composition.
    - **Cinematic Scene Builder:** Generate cinematic camera, lens, lighting, and atmospheric parameters.
- **Reasoning Cleanup:** Automatically strips `<think>` / reasoning tags (common in models like DeepSeek-R1) to output clean prompt text.

### 🛠️ Essential Utilities
- **YE Empty Latent Image:** Includes industry-standard presets for SD1.5 and SDXL (Square, Landscape, Portrait).
- **YE Image Upscale:** Tiled, memory-safe upscaling using external models.
- **Streamlined Loaders:** Simplified, clean loaders for Checkpoints, UNETs, and LoRAs (including model-only loaders), plus a stackable LoRA loader with a dynamic slot interface.

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

[Apache Software License 2.0](./LICENSE)
