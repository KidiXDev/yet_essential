import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

const Z_INDEX = 2147483647;

const STYLES = `
.ye-dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(8px);
    z-index: ${Z_INDEX};
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Segoe UI', sans-serif;
    color: #fff;
}

.ye-dialog-content {
    background: #111;
    width: 90vw;
    max-width: 1100px;
    height: 85vh;
    border: 1px solid #333;
    border-radius: 12px;
    box-shadow: 0 20px 80px rgba(0,0,0,0.8);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    animation: ye-pop 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes ye-pop {
    from { opacity: 0; transform: scale(0.95) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

.ye-dialog-header {
    padding: 15px 20px;
    background: #1a1a1a;
    border-bottom: 1px solid #222;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.ye-dialog-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
    color: #4a90e2;
}

.ye-close-button {
    background: transparent;
    border: none;
    color: #888;
    font-size: 28px;
    cursor: pointer;
    line-height: 1;
    padding: 0 10px;
}

.ye-close-button:hover {
    color: #fff;
}

.ye-filter-container {
    padding: 10px 20px;
    background: #111;
    border-bottom: 1px solid #222;
}

.ye-dialog-filter {
    width: 100%;
    background: #000;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 15px;
    color: #fff;
    font-size: 1rem;
    outline: none;
    box-sizing: border-box;
}

.ye-dialog-grid {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 15px;
    align-content: start;
    scroll-behavior: smooth;
    scrollbar-width: thin;
}

.ye-dialog-card {
    height: 200px;
    background-color: #080808;
    border: 1px solid #222;
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    transition: all 0.2s ease;
    overflow: hidden;
}

.ye-dialog-card:hover {
    border-color: #4a90e2;
    transform: translateY(-4px);
    z-index: 5;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}

.ye-card-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0;
    transition: opacity 0.4s;
}

.ye-card-img.loaded {
    opacity: 1;
}

.ye-dialog-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(4px);
    padding: 8px 5px;
    color: #fff;
    font-size: 11px;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    pointer-events: none;
}
`;

class YESelectionDialog {
    constructor(title, values, folderType, onSelect) {
        if (document.querySelector(".ye-dialog-overlay")) return null;

        this.title = title;
        this.values = values.filter(v => v && v !== "None");
        this.folderType = folderType;
        this.onSelect = onSelect;
        this.filterText = "";
        
        this.el = this.render();
        document.body.appendChild(this.el);
        this.filterInput.focus();
        
        this.onKeyDown = (e) => {
             if (e.key === "Escape") this.close();
             if (e.key === "Enter") {
                const filtered = this.values.filter(v => v.toLowerCase().includes(this.filterText));
                if (filtered.length > 0) { this.onSelect(filtered[0]); this.close(); }
             }
        };
        window.addEventListener("keydown", this.onKeyDown, true);
    }

    close() {
        window.removeEventListener("keydown", this.onKeyDown, true);
        this.el.remove();
    }

    render() {
        this.grid = $el("div.ye-dialog-grid");
        this.filterInput = $el("input.ye-dialog-filter", {
            placeholder: "Search models...",
            oninput: (e) => {
                this.filterText = e.target.value.toLowerCase();
                this.updateGrid();
            }
        });

        const overlay = $el("div.ye-dialog-overlay", {
            onclick: (e) => { if (e.target === overlay) this.close(); }
        }, [
            $el("div.ye-dialog-content", [
                $el("div.ye-dialog-header", [
                    $el("h2.ye-dialog-title", [this.title]),
                    $el("button.ye-close-button", { onclick: () => this.close() }, ["×"])
                ]),
                $el("div.ye-filter-container", [this.filterInput]),
                this.grid
            ])
        ]);

        this.updateGrid();
        return overlay;
    }

    updateGrid() {
        this.grid.innerHTML = "";
        const filtered = this.values.filter(v => v.toLowerCase().includes(this.filterText));
        
        for (const val of filtered) {
             const cleanName = val.split(/[\\\/]/).pop().replace(/\.[^/.]+$/, "");
             const thumbUrl = `/yet_essential/model/preview?type=${encodeURIComponent(this.folderType)}&name=${encodeURIComponent(val)}&res=300`;
             
             // Use a safer image approach
             const img = document.createElement("img");
             img.className = "ye-card-img";
             img.loading = "lazy";
             img.src = thumbUrl;
             img.onload = () => img.classList.add("loaded");

             const card = $el("div.ye-dialog-card", {
                onclick: () => { this.onSelect(val); this.close(); }
             }, [
                img,
                $el("div.ye-dialog-label", [cleanName])
             ]);
             
             this.grid.appendChild(card);
        }
    }
}

const MODEL_EXTENSIONS = [".safetensors", ".ckpt", ".pt", ".bin", ".gguf", ".sft"];

class YENativeModelPreview {
    constructor() {
        this.setupStyles();
        this.setupBridge();
    }

    setupStyles() {
        const s = document.createElement("style");
        s.innerText = STYLES;
        document.head.appendChild(s);
    }

    setupBridge() {
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (added.nodeType !== 1) continue;
                    if (added.ye_bridged) continue;
                    const style = window.getComputedStyle(added);
                    if (style.position === "absolute" || style.position === "fixed") {
                         if (this.isLikelyModelMenu(added)) this.bridgeToDialog(added);
                    }
                    const nested = added.querySelectorAll?.('[class*="menu"], [class*="panel"], [class*="list"]');
                    if (nested) {
                        for (const n of nested) {
                            if (n.ye_bridged) continue;
                            if (this.isLikelyModelMenu(n)) this.bridgeToDialog(n);
                        }
                    }
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    isLikelyModelMenu(el) {
        if (!el || !el.innerText || el.ye_bridged) return false;
        const style = window.getComputedStyle(el);
        if (style.position !== "absolute" && style.position !== "fixed") return false;
        const txt = el.innerText.toLowerCase();
        let count = 0;
        for (const ext of MODEL_EXTENSIONS) {
            const matches = txt.match(new RegExp(ext.replace(".", "\\."), "g"));
            if (matches) count += matches.length;
        }
        return count > 2;
    }

    bridgeToDialog(menu) {
        menu.ye_bridged = true;
        const itemSelector = '[class*="item"], [class*="entry"], li';
        const items = Array.from(menu.querySelectorAll(itemSelector))
                           .map(i => i.innerText.trim())
                           .filter(t => t && t !== "Cancel" && t !== "Filter");
        if (items.length < 2) return;

        menu.style.display = "none";
        menu.style.visibility = "hidden";
        menu.style.opacity = "0";

        const target = this.findTargetByValues(items);
        if (!target) return;
        const { node, widget } = target;
        const folderType = this.getFolderType(node, widget);

        new YESelectionDialog(`Select ${folderType.replace('_', ' ')}`, items, folderType, (selected) => {
            widget.value = selected;
            if (widget.callback) widget.callback(selected);
            app.canvas.setDirty(true, true);
        });
    }

    findTargetByValues(items) {
        if (!app.graph || !app.graph._nodes) return null;
        const sample = items.slice(0, 5); 
        for (const node of app.graph._nodes) {
            if (!node.widgets) continue;
            for (const w of node.widgets) {
                if (w.type === "combo" && w.options && Array.isArray(w.options.values)) {
                    const values = w.options.values;
                    const matchCount = sample.filter(s => values.includes(s)).length;
                    if (matchCount >= 3 || (sample.length > 0 && matchCount === sample.length)) {
                        return { node, widget: w };
                    }
                }
            }
        }
        return null;
    }

    getFolderType(node, widget) {
        const wn = widget.name.toLowerCase();
        if (wn.includes("ckpt")) return "checkpoints";
        if (wn.includes("lora")) return "loras";
        if (wn.includes("unet") || wn.includes("diffusion")) return "diffusion_models";
        const nt = (node.comfyClass || node.type || "").toLowerCase();
        if (nt.includes("checkpoint")) return "checkpoints";
        if (nt.includes("lora")) return "loras";
        if (nt.includes("unet") || nt.includes("diffusion")) return "diffusion_models";
        return "checkpoints";
    }
}

app.registerExtension({
    name: "yet_essential.native_previews",
    init() {
        if (!window.ye_native_preview_instance) {
            window.ye_native_preview_instance = new YENativeModelPreview();
        }
    },
});
