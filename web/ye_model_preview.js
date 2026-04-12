import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

const IMAGE_SIZE = 300;

const STYLES = `
.ye-combo-image {
    position: absolute;
    width: ${IMAGE_SIZE}px;
    height: ${IMAGE_SIZE}px;
    object-fit: contain;
    background: #000;
    border: 1px solid #444;
    border-radius: 8px;
    z-index: 10001;
    pointer-events: none;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    transition: opacity 0.1s;
}

.ye-combo-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
    max-width: 80vw;
    max-height: 80vh !important;
    overflow-y: auto !important;
    padding: 10px !important;
    background: #111 !important;
}

.ye-combo-grid .litemenu-entry {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    height: 180px !important;
    border: 1px solid #333;
    border-radius: 6px;
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden;
    position: relative;
    color: #fff;
    text-shadow: 0 1px 4px rgba(0,0,0,0.8);
    background-color: #050505;
}

.ye-combo-grid .litemenu-entry:hover {
    background-color: #1a1a1a;
    border-color: #888;
    z-index: 2;
}

.ye-combo-grid .litemenu-entry span {
    background: rgba(0,0,0,0.7);
    width: 100%;
    text-align: center;
    padding: 4px 2px;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.ye-combo-grid .comfy-context-menu-filter {
    grid-column: 1 / -1;
    position: sticky;
    top: -10px;
    z-index: 10;
    background: #111;
    padding: 5px 0;
}

.ye-combo-grid .litemenu-entry.selected {
    border-color: #4a90e2;
    background-color: #003366;
}
`;

const YE_NODES = ["YELoadCheckpoint", "YELoadDiffusionModel", "YELoadLora", "YELoadLoraModel"];

function getFolderType(node, widgetName) {
    if (widgetName === "ckpt_name") return "checkpoints";
    if (widgetName === "unet_name") return "diffusion_models";
    if (widgetName === "lora_name") return "loras";
    return null;
}

class YENativeModelPreview {
    constructor() {
        this.imageHost = $el("img.ye-combo-image", { style: { display: "none" } });
        document.body.appendChild(this.imageHost);
        this.setupStyles();
        this.setupObserver();
    }

    setupStyles() {
        const s = document.createElement("style");
        s.id = "ye-native-styles";
        s.innerText = STYLES;
        document.head.appendChild(s);
    }

    setupObserver() {
        // Watch for context menus
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (added.classList?.contains("litecontextmenu")) {
                        // Delay slightly to allow ComfyUI extensions to populate the menu (like filter)
                        requestAnimationFrame(() => this.handleContextMenu(added));
                    }
                }
                for (const removed of mutation.removedNodes) {
                    if (removed.classList?.contains("litecontextmenu")) {
                        this.hideImage();
                    }
                }
            }
        });
        observer.observe(document.body, { childList: true });
    }

    hideImage() {
        this.imageHost.style.display = "none";
        this.imageHost.src = "";
    }

    showImage(item, type, name) {
        if (!name || name === "Cancel" || name === "Filter") return;
        
        const rect = item.getBoundingClientRect();
        const bodyRect = document.body.getBoundingClientRect();
        
        let x = rect.right + 10;
        let y = rect.top;

        if (x + IMAGE_SIZE + 20 > bodyRect.width) {
            x = rect.left - IMAGE_SIZE - 10;
        }
        
        const constrainedY = Math.min(y, bodyRect.height - IMAGE_SIZE - 10);
        y = Math.max(10, constrainedY);

        this.imageHost.src = `/yet_essential/model/preview?type=${encodeURIComponent(type)}&name=${encodeURIComponent(name)}`;
        this.imageHost.style.left = `${x}px`;
        this.imageHost.style.top = `${y}px`;
        this.imageHost.style.display = "block";
    }

    async handleContextMenu(menu) {
        console.log("YE Debug: Context menu detected");
        
        let node = app.canvas.current_node;
        if (!node && app.canvas.selected_nodes) {
             const selectedIds = Object.keys(app.canvas.selected_nodes);
             if (selectedIds.length > 0) {
                 node = app.canvas.selected_nodes[selectedIds[0]];
             }
        }
        
        if (!node) {
            console.log("YE Debug: No active node found");
            return;
        }

        const nodeClass = node.comfyClass || node.type;
        console.log("YE Debug: Node class is", nodeClass);
        
        if (!YE_NODES.includes(nodeClass)) {
            console.log("YE Debug: Not a YE node");
            return;
        }

        const overWidget = node.widgets?.find(w => w.name && getFolderType(node, w.name)) || app.canvas.getWidgetAtCursor();
        if (!overWidget) {
            console.log("YE Debug: No model widget found at cursor");
            return;
        }

        const folderType = getFolderType(node, overWidget.name);
        if (!folderType) {
            console.log("YE Debug: Widget is not a model selection widget", overWidget.name);
            return;
        }

        const items = menu.querySelectorAll(".litemenu-entry");
        if (!items.length) {
            console.log("YE Debug: Menu has no items yet");
            // If items are not yet populated, we might need another frame
            requestAnimationFrame(() => {
                const retryItems = menu.querySelectorAll(".litemenu-entry");
                if (retryItems.length) this.handleContextMenu(menu);
            });
            return;
        }

        console.log("YE Debug: Applying YE Grid Mode to", folderType);
        
        // Apply grid mode
        menu.classList.add("ye-combo-grid");
        
        items.forEach(item => {
            let name = item.getAttribute("data-value") || item.innerText.trim();
            
            if (!name || name === "Cancel" || name === "Filter") return;

            // Wrap text in a span for better styling over image
            const cleanName = name.split(/[\\\/]/).pop();
            item.innerHTML = `<span>${cleanName}</span>`;
            
            const url = `/yet_essential/model/preview?type=${encodeURIComponent(folderType)}&name=${encodeURIComponent(name)}`;
            item.style.backgroundImage = `url("${url}")`;

            item.addEventListener("mouseenter", () => {
                this.showImage(item, folderType, name);
            });
            item.addEventListener("mouseleave", () => {
                this.hideImage();
            });
        });
        
        // Final position correction
        const menuRect = menu.getBoundingClientRect();
        const bodyRect = document.body.getBoundingClientRect();
        if (menuRect.right > bodyRect.width) {
            menu.style.left = Math.max(0, bodyRect.width - menuRect.width - 20) + "px";
        }
        if (menuRect.bottom > bodyRect.height) {
            menu.style.top = Math.max(0, bodyRect.height - menuRect.height - 20) + "px";
        }
    }
}

app.registerExtension({
    name: "yet_essential.native_previews",
    init() {
        // Initialize once
        if (!window.ye_native_preview_instance) {
            window.ye_native_preview_instance = new YENativeModelPreview();
        }
    }
});
