import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "yet_essential.lora_stack_dynamic_widgets";
const TARGET_NODE_NAMES = ["YELoraStack", "YELoraStackModel"];
const MAX_SLOTS = 25;
const SLOT_NONE = "None";
const SYNC_DELAYS_MS = [0, 60, 200, 600];
const WATCH_INTERVAL_MS = 250;
const IS_V2_FRONTEND = typeof window !== "undefined" && !!window.comfyAPI;
const LEGACY_HIDDEN_TYPE = "ye_hidden";

const originalWidgetProps = new WeakMap();
const watchers = new WeakMap();

function findWidgetByName(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function toggleWidget(widget, show = false) {
    if (!widget) {
        return;
    }

    if (!widget.options || typeof widget.options !== "object") {
        widget.options = {};
    }
    widget.options.hidden = !show;
    widget.hidden = !show;

    if (!originalWidgetProps.has(widget)) {
        originalWidgetProps.set(widget, {
            type: widget.type,
            computeSize: widget.computeSize,
            computedHeight: widget.computedHeight,
        });
    }

    const original = originalWidgetProps.get(widget);
    if (!IS_V2_FRONTEND) {
        if (show) {
            widget.type = original.type;
            widget.computeSize = original.computeSize;
            widget.computedHeight = original.computedHeight;
        } else {
            widget.type = LEGACY_HIDDEN_TYPE;
            widget.computeSize = () => [0, -4];
            widget.computedHeight = 0;
        }
    }
}

function refreshNodeLayout(node) {
    if (!node || typeof node.computeSize !== "function") {
        return;
    }
    node.setSize([node.size[0], node.computeSize()[1]]);
    app.canvas.setDirty(true, true);
}

function slotIsFilled(node, idx) {
    const widget = findWidgetByName(node, `lora_name_${idx}`);
    if (!widget) {
        return false;
    }
    const value = String(widget.value ?? "").trim();
    return value.length > 0 && value !== SLOT_NONE;
}

function computeVisibleSlots(node) {
    let lastFilled = 0;
    for (let idx = 1; idx <= MAX_SLOTS; idx += 1) {
        if (slotIsFilled(node, idx)) {
            lastFilled = idx;
        }
    }
    return Math.min(MAX_SLOTS, Math.max(1, lastFilled + 1));
}

function updateSlotVisibility(node) {
    const visibleSlots = computeVisibleSlots(node);
    const nodeName = node?.comfyClass || node?.type;
    const isModelOnly = nodeName === "YELoraStackModel";
    for (let idx = 1; idx <= MAX_SLOTS; idx += 1) {
        const show = idx <= visibleSlots;
        toggleWidget(findWidgetByName(node, `lora_name_${idx}`), show);
        toggleWidget(findWidgetByName(node, `strength_model_${idx}`), show);
        if (!isModelOnly) {
            toggleWidget(findWidgetByName(node, `strength_clip_${idx}`), show);
        }
    }
    refreshNodeLayout(node);
}

function scheduleSyncPasses(node) {
    for (const delay of SYNC_DELAYS_MS) {
        window.setTimeout(() => updateSlotVisibility(node), delay);
    }
}

function hookLoraStackNode(node) {
    if (!node || node.__yeLoraStackHooked) {
        scheduleSyncPasses(node);
        ensureWatcher(node);
        return;
    }
    node.__yeLoraStackHooked = true;

    for (let idx = 1; idx <= MAX_SLOTS; idx += 1) {
        const nameWidget = findWidgetByName(node, `lora_name_${idx}`);
        if (!nameWidget || nameWidget.__yeLoraSlotHooked) {
            continue;
        }
        nameWidget.__yeLoraSlotHooked = true;
        const originalCallback = nameWidget.callback;
        nameWidget.callback = function patchedLoraNameCallback(value, ...args) {
            const result = typeof originalCallback === "function"
                ? originalCallback.call(this, value, ...args)
                : undefined;
            updateSlotVisibility(node);
            return result;
        };
    }

    scheduleSyncPasses(node);
    ensureWatcher(node);
}

function ensureWatcher(node) {
    if (watchers.has(node)) {
        return;
    }
    const timer = window.setInterval(() => {
        if (!node || !node.graph) {
            window.clearInterval(timer);
            watchers.delete(node);
            return;
        }

        let stateKey = "";
        for (let idx = 1; idx <= MAX_SLOTS; idx += 1) {
            const value = findWidgetByName(node, `lora_name_${idx}`)?.value;
            stateKey += `|${idx}:${String(value ?? "")}`;
        }

        if (node.__yeLoraStackStateKey !== stateKey) {
            node.__yeLoraStackStateKey = stateKey;
            updateSlotVisibility(node);
        }
    }, WATCH_INTERVAL_MS);

    watchers.set(node, timer);
}

app.registerExtension({
    name: EXTENSION_NAME,
    nodeCreated(node) {
        const nodeName = node?.comfyClass || node?.type;
        if (TARGET_NODE_NAMES.includes(nodeName)) {
            hookLoraStackNode(node);
        }
    },
    loadedGraphNode(node) {
        const nodeName = node?.comfyClass || node?.type;
        if (TARGET_NODE_NAMES.includes(nodeName)) {
            hookLoraStackNode(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODE_NAMES.includes(nodeData.name)) {
            return;
        }

        const originalOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function patchedOnWidgetChanged() {
            const result = originalOnWidgetChanged?.apply(this, arguments);
            const [name] = arguments;
            if (typeof name === "string" && name.startsWith("lora_name_")) {
                scheduleSyncPasses(this);
            }
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            hookLoraStackNode(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            hookLoraStackNode(this);
            return result;
        };
    },
});
