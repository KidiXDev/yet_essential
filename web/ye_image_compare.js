import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "yet_essential.image_compare";
const TARGET_NODE_NAME = "YEImageComparer";
const MIN_NODE_WIDTH = 420;
const MIN_NODE_HEIGHT = 320;
const PAD_X = 12;
const PAD_Y = 8;
const SLOT_ROW_HEIGHT = 22;
const TITLE_HEIGHT = 30;
const DOM_WIDGET_NAME = "ye_compare_dom";

function createState() {
    return {
        beforeUrls: [],
        afterUrls: [],
        beforeIndex: 0,
        afterIndex: 0,
        slider: 0.5,
        beforeRef: { url: "", img: null },
        afterRef: { url: "", img: null },
        dom: null,
        domWidget: null,
        useDom: true,
    };
}

function parseUiPayload(payload) {
    if (!payload || typeof payload !== "object") {
        return { a_images: [], b_images: [] };
    }
    const ui = payload.ui && typeof payload.ui === "object" ? payload.ui : payload;
    const a_images = Array.isArray(ui.a_images) ? ui.a_images : [];
    const b_images = Array.isArray(ui.b_images) ? ui.b_images : [];
    return { a_images, b_images };
}

function toImageUrl(meta) {
    if (!meta) return "";
    const query = new URLSearchParams({
        filename: String(meta.filename || ""),
        type: String(meta.type || "temp"),
        subfolder: String(meta.subfolder || ""),
    });

    const previewParam =
        typeof app.getPreviewFormatParam === "function"
            ? app.getPreviewFormatParam()
            : "";
    const randParam =
        typeof app.getRandParam === "function" ? app.getRandParam() : "";

    return api.apiURL(`/view?${query}${previewParam}${randParam}`);
}

function fitContain(srcW, srcH, dstX, dstY, dstW, dstH) {
    if (!srcW || !srcH || !dstW || !dstH) {
        return { x: dstX, y: dstY, w: 0, h: 0 };
    }

    const srcAspect = srcW / srcH;
    const dstAspect = dstW / dstH;

    if (srcAspect > dstAspect) {
        const w = dstW;
        const h = w / srcAspect;
        return { x: dstX, y: dstY + (dstH - h) * 0.5, w, h };
    }

    const h = dstH;
    const w = h * srcAspect;
    return { x: dstX + (dstW - w) * 0.5, y: dstY, w, h };
}

function fitCover(srcW, srcH, dstX, dstY, dstW, dstH) {
    if (!srcW || !srcH || !dstW || !dstH) {
        return { x: dstX, y: dstY, w: 0, h: 0 };
    }

    const srcAspect = srcW / srcH;
    const dstAspect = dstW / dstH;

    if (srcAspect > dstAspect) {
        const h = dstH;
        const w = h * srcAspect;
        return { x: dstX + (dstW - w) * 0.5, y: dstY, w, h };
    }

    const w = dstW;
    const h = w / srcAspect;
    return { x: dstX, y: dstY + (dstH - h) * 0.5, w, h };
}

function getDrawRect(node) {
    const inputRows = Array.isArray(node.inputs) ? node.inputs.length : 0;
    const top = TITLE_HEIGHT + inputRows * SLOT_ROW_HEIGHT + PAD_Y;
    const x = PAD_X;
    const y = top;
    const w = Math.max(10, node.size[0] - PAD_X * 2);
    const h = Math.max(80, node.size[1] - top - PAD_Y);
    return { x, y, w, h };
}

function getCompareHeight(node) {
    return getDrawRect(node).h;
}

function syncDomLayout(node, state) {
    if (!state?.dom) return;
    const height = getCompareHeight(node);
    state.dom.root.style.height = `${height}px`;
    if (state.domWidget) {
        state.domWidget.computeSize = (width) => [
            Math.max(10, width - PAD_X * 2),
            height,
        ];
    }
}

function ensureImage(ref, node) {
    if (!ref?.url) return null;
    if (ref.img && ref.img.__ye_url === ref.url) return ref.img;

    const img = new Image();
    img.__ye_url = ref.url;
    img.onload = () => app.canvas.setDirty(true, true);
    img.onerror = () => app.canvas.setDirty(true, true);
    img.src = ref.url;
    ref.img = img;
    return img;
}

function applyDomCompare(state) {
    const dom = state?.dom;
    if (!dom) return;

    const pair = selectPair(state);
    const beforeUrl = pair.before || "";
    const afterUrl = pair.after || "";

    dom.empty.style.display = beforeUrl || afterUrl ? "none" : "flex";
    dom.after.style.display = afterUrl ? "block" : "none";
    dom.before.style.display = beforeUrl ? "block" : "none";

    if (dom.before.src !== beforeUrl) {
        dom.before.src = beforeUrl;
    }
    if (dom.after.src !== afterUrl) {
        dom.after.src = afterUrl;
    }

    const pct = Math.max(0, Math.min(100, state.slider * 100));
    dom.before.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    dom.line.style.left = `${pct}%`;
}

function createDomCompare(node, state) {
    const root = document.createElement("div");
    root.style.position = "relative";
    root.style.width = "100%";
    root.style.height = `${getCompareHeight(node)}px`;
    root.style.background = "rgba(20,20,20,0.85)";
    root.style.border = "1px solid rgba(255,255,255,0.15)";
    root.style.borderRadius = "8px";
    root.style.overflow = "hidden";
    root.style.userSelect = "none";

    const after = document.createElement("img");
    after.draggable = false;
    after.style.position = "absolute";
    after.style.inset = "0";
    after.style.width = "100%";
    after.style.height = "100%";
    after.style.objectFit = "contain";

    const before = document.createElement("img");
    before.draggable = false;
    before.style.position = "absolute";
    before.style.inset = "0";
    before.style.width = "100%";
    before.style.height = "100%";
    before.style.objectFit = "contain";

    const line = document.createElement("div");
    line.style.position = "absolute";
    line.style.top = "0";
    line.style.bottom = "0";
    line.style.width = "2px";
    line.style.background = "rgba(255,255,255,0.95)";
    line.style.boxShadow = "0 0 8px rgba(0,0,0,0.6)";
    line.style.pointerEvents = "none";

    const empty = document.createElement("div");
    empty.textContent = "No image to compare";
    empty.style.position = "absolute";
    empty.style.inset = "0";
    empty.style.display = "flex";
    empty.style.alignItems = "center";
    empty.style.justifyContent = "center";
    empty.style.color = "rgba(220,220,220,0.8)";
    empty.style.fontSize = "12px";
    empty.style.fontFamily = "sans-serif";

    root.appendChild(after);
    root.appendChild(before);
    root.appendChild(line);
    root.appendChild(empty);

    const updateSlider = (clientX) => {
        const rect = root.getBoundingClientRect();
        if (rect.width <= 0) return;
        state.slider = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        applyDomCompare(state);
    };

    root.addEventListener("mousemove", (event) => {
        updateSlider(event.clientX);
    });
    root.addEventListener("pointermove", (event) => {
        updateSlider(event.clientX);
    });

    const domWidget = node.addDOMWidget?.(DOM_WIDGET_NAME, DOM_WIDGET_NAME, root, {
        serialize: false,
        hideOnZoom: false,
    });
    if (domWidget) {
        domWidget.serialize = false;
        state.domWidget = domWidget;
    }

    state.dom = { root, before, after, line, empty };
    syncDomLayout(node, state);
    applyDomCompare(state);
}

function selectPair(state) {
    const before = state.beforeUrls[state.beforeIndex] || "";
    const after = state.afterUrls[state.afterIndex] || "";

    if (before && after) return { before, after };

    const all = [...state.beforeUrls, ...state.afterUrls];
    if (all.length >= 2) return { before: all[0], after: all[1] };
    if (all.length === 1) return { before: all[0], after: "" };
    return { before: "", after: "" };
}

app.registerExtension({
    name: EXTENSION_NAME,
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE_NAME) return;

        function ensureSetup(node) {
            node.setSize([
                Math.max(node.size[0], MIN_NODE_WIDTH),
                Math.max(node.size[1], MIN_NODE_HEIGHT),
            ]);
            if (!node.__yeCompare) {
                node.__yeCompare = createState();
            }
            if (!node.__yeCompare.dom) {
                createDomCompare(node, node.__yeCompare);
            }
            syncDomLayout(node, node.__yeCompare);
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            ensureSetup(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            ensureSetup(this);
            return result;
        };

        const originalOnResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function patchedOnResize(size) {
            const result = originalOnResize?.apply(this, arguments);
            const state = this.__yeCompare;
            if (state) {
                syncDomLayout(this, state);
                if (state.useDom) {
                    applyDomCompare(state);
                }
            }
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function patchedOnExecuted(payload) {
            originalOnExecuted?.apply(this, arguments);
            ensureSetup(this);
            const state = this.__yeCompare;
            const { a_images, b_images } = parseUiPayload(payload);

            state.beforeUrls = a_images.map(toImageUrl).filter(Boolean);
            state.afterUrls = b_images.map(toImageUrl).filter(Boolean);
            state.beforeIndex = 0;
            state.afterIndex = 0;

            state.beforeRef.url = selectPair(state).before || "";
            state.afterRef.url = selectPair(state).after || "";
            ensureImage(state.beforeRef, this);
            ensureImage(state.afterRef, this);

            if (state.useDom) {
                applyDomCompare(state);
            }

            app.canvas.setDirty(true, true);
        };

        const originalOnMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function patchedOnMouseMove(event, pos, graphCanvas) {
            const r = originalOnMouseMove?.apply(this, arguments);
            const state = this.__yeCompare;
            if (!state) return r;
            if (state.useDom) return r;

            const rect = getDrawRect(this);
            const x = pos?.[0] ?? 0;
            const y = pos?.[1] ?? 0;
            if (y >= rect.y && y <= rect.y + rect.h) {
                state.slider = Math.max(0, Math.min(1, (x - rect.x) / rect.w));
                app.canvas.setDirty(true, false);
            }
            return r;
        };

        const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function patchedOnDrawBackground(ctx) {
            originalOnDrawBackground?.apply(this, arguments);
            const state = this.__yeCompare;
            if (!state || this.flags?.collapsed) return;
            if (state.useDom) return;

            const rect = getDrawRect(this);
            ctx.save();

            ctx.fillStyle = "rgba(20, 20, 20, 0.85)";
            ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

            const pair = selectPair(state);
            state.beforeRef.url = pair.before || "";
            state.afterRef.url = pair.after || "";
            const beforeImg = ensureImage(state.beforeRef, this);
            const afterImg = ensureImage(state.afterRef, this);

            const baseImg = afterImg || beforeImg;
            if (!baseImg || !baseImg.complete || !baseImg.naturalWidth || !baseImg.naturalHeight) {
                ctx.fillStyle = "rgba(220, 220, 220, 0.8)";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("No image to compare", rect.x + rect.w * 0.5, rect.y + rect.h * 0.5);
                ctx.restore();
                return;
            }

            const fitted = fitCover(
                baseImg.naturalWidth,
                baseImg.naturalHeight,
                rect.x,
                rect.y,
                rect.w,
                rect.h,
            );

            if (afterImg && afterImg.complete && afterImg.naturalWidth > 0) {
                ctx.drawImage(afterImg, fitted.x, fitted.y, fitted.w, fitted.h);
            } else if (beforeImg && beforeImg.complete && beforeImg.naturalWidth > 0) {
                ctx.drawImage(beforeImg, fitted.x, fitted.y, fitted.w, fitted.h);
            }

            if (beforeImg && beforeImg.complete && beforeImg.naturalWidth > 0) {
                const splitX = fitted.x + fitted.w * state.slider;
                ctx.save();
                ctx.beginPath();
                ctx.rect(fitted.x, fitted.y, Math.max(0, splitX - fitted.x), fitted.h);
                ctx.clip();
                ctx.drawImage(beforeImg, fitted.x, fitted.y, fitted.w, fitted.h);
                ctx.restore();

                ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(splitX, fitted.y);
                ctx.lineTo(splitX, fitted.y + fitted.h);
                ctx.stroke();
            }

            ctx.restore();
        };
    },
});
