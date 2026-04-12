import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

console.log("YE Model Preview: Applying Iron-Clad event isolation...");

const STYLES = `
.ye-modal-overlay {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.95);
    z-index: 10001;
    display: flex; justify-content: center; align-items: center;
    backdrop-filter: blur(12px);
}
.ye-modal {
    width: 95vw; height: 90vh;
    background: #080808; border: 1px solid #333; border-radius: 12px;
    display: flex; flex-direction: column; overflow: hidden; color: #eee;
}
.ye-modal-header {
    padding: 20px 30px; border-bottom: 1px solid #222;
    display: flex; justify-content: space-between; align-items: center;
    background: #111;
}
.ye-search {
    flex: 1; margin: 0 40px; background: #000; border: 1px solid #444;
    color: #fff; padding: 12px 20px; border-radius: 8px; font-size: 16px;
    outline: none;
}
.ye-grid {
    flex: 1; overflow-y: auto; padding: 30px;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); grid-auto-rows: min-content; gap: 30px;
}
.ye-card {
    background: #121212; border-radius: 10px; overflow: hidden; cursor: pointer; border: 1px solid #222;
    display: flex; flex-direction: column; height: 380px;
}
.ye-card:hover { border-color: #666; background: #1a1a1a; }
.ye-card-img-container { flex: 1; width: 100%; display: flex; align-items: center; justify-content: center; overflow: hidden; pointer-events: none; }
.ye-card-img { width: 100%; height: 100%; object-fit: cover; }
.ye-card-title { padding: 15px; font-size: 12px; text-align: center; border-top: 1px solid #222; color: #888; pointer-events: none; }
.ye-close-btn { background: #444; border: none; color: #fff; cursor: pointer; padding: 10px 30px; border-radius: 6px; font-weight: bold; }
`;

// Strongest possible state lockout
window.ye_browser_active = false;
let ye_lockout_until = 0;

class ModelBrowser {
    constructor(folderType, onSelect) {
        this.folderType = folderType;
        this.onSelect = onSelect;
        this.allModels = [];
        this.el = null;
    }

    async show() {
        if (window.ye_browser_active) return;
        try {
            const r = await fetch(`/yet_essential/model/list?type=${encodeURIComponent(this.folderType)}`);
            this.allModels = await r.json();
            window.ye_browser_active = true;
            this.render();
        } catch (e) { console.error("YE Error:", e); }
    }

    render(filter = "") {
        if (!this.el) {
            if (!document.getElementById("ye-styles")) {
                const s = document.createElement("style");
                s.id = "ye-styles"; s.innerText = STYLES; document.head.appendChild(s);
            }
            this.el = document.createElement("div");
            this.el.className = "ye-modal-overlay";
            this.el.innerHTML = `<div class="ye-modal">
                <div class="ye-modal-header">
                    <div style="font-weight:bold; font-size:18px; color: #666;">BROWSER: ${this.folderType.toUpperCase()}</div>
                    <input type="text" class="ye-search" placeholder="Filter..." />
                    <button class="ye-close-btn">CANCEL</button>
                </div>
                <div class="ye-grid"></div>
            </div>`;
            document.body.appendChild(this.el);
            
            // IRON-CLAD: Stop ALL mouse signaling from leaking to background
            const stop = (e) => { e.stopPropagation(); e.stopImmediatePropagation(); };
            this.el.addEventListener("mousedown", stop);
            this.el.addEventListener("mouseup", stop);
            this.el.addEventListener("click", stop);
            this.el.addEventListener("pointerdown", stop);
            this.el.addEventListener("pointerup", stop);
            this.el.addEventListener("wheel", stop);

            this.el.querySelector(".ye-close-btn").onclick = (e) => this.close();
            this.el.querySelector(".ye-search").oninput = (e) => this.render(e.target.value);
            this.el.onmousedown = (e) => { if(e.target === this.el) this.close(); };
            setTimeout(() => this.el.querySelector(".ye-search").focus(), 100);
        }

        const grid = this.el.querySelector(".ye-grid");
        grid.innerHTML = "";
        this.allModels.filter(m => m.name.toLowerCase().includes(filter.toLowerCase())).forEach(model => {
            const card = document.createElement("div");
            card.className = "ye-card";
            card.innerHTML = `
                <div class="ye-card-img-container">
                    ${model.has_preview ? `<img class="ye-card-img" src="/yet_essential/model/preview?type=${encodeURIComponent(this.folderType)}&name=${encodeURIComponent(model.name)}">` : `<span style="color:#222;">N/A</span>`}
                </div>
                <div class="ye-card-title">${model.name}</div>
            `;
            
            // Selection on click, but eat the event entirely
            card.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                console.log("YE Browser: Finalizing selection", model.name);
                this.onSelect(model.name);
                this.close();
            };
            grid.appendChild(card);
        });
    }

    close() { 
        if (this.el) { 
            ye_lockout_until = Date.now() + 1000; // 1 second iron-clad lockout
            window.ye_browser_active = false;
            this.el.remove(); 
            this.el = null; 
            console.log("YE Browser: Gallery closed.");
        } 
    }
}

ComfyWidgets["YE_MODEL_SELECT"] = function(node, inputName, inputData, app) {
    const folderType = inputData[1]?.folder || "checkpoints";
    
    // NATIVE BUTTON BRIDGE
    const widget = node.addWidget("button", inputName, "Select Model...", () => {
        if (window.ye_browser_active) return;
        if (Date.now() < ye_lockout_until) return;

        const browser = new ModelBrowser(folderType, (val) => {
            widget.value = val;
            if (widget.callback) widget.callback(val);
        });
        browser.show();
    });

    widget.type = "YE_MODEL_SELECT";
    widget.value = inputData[1]?.default || "";
    widget.serializeValue = async () => widget.value;
    widget.h = 28; // Standard height to prevent overlap

    widget.draw = function(ctx, node, width, y, spare) {
        const margin = 15;
        const w = width - margin * 2;
        const h = 28;
        
        const isHover = this.mouse_over;
        ctx.fillStyle = isHover ? "#2a2a2a" : "#0d0d0d"; 
        ctx.strokeStyle = isHover ? "#555" : "#222";
        ctx.beginPath(); ctx.roundRect(margin, y, w, h, 4); ctx.fill(); ctx.stroke();
        ctx.fillStyle = isHover ? "#fff" : "#999"; ctx.font = "11px sans-serif";
        const txt = this.value || "Select Model...";
        ctx.save(); ctx.beginPath(); ctx.rect(margin + 5, y, w - 25, h); ctx.clip();
        ctx.fillText(txt, margin + 8, y + (h / 2) + 4); ctx.restore();
        
        ctx.fillStyle = isHover ? "#888" : "#444";
        const dotX = margin + w - 10;
        const dotY = y + (h / 2);
        for(let i=0; i<3; i++) {
            ctx.beginPath(); ctx.arc(dotX, dotY - 4 + (i*4), 1.2, 0, Math.PI * 2); ctx.fill();
        }
    };

    return widget;
};

app.registerExtension({
    name: "yet_essential.model_browser"
});
