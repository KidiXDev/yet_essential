import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "yet_essential.empty_latent_dynamic_widgets";
const TARGET_NODE_NAME = "YEEmptyLatentImage";
const PRESET_WIDGET_NAME = "preset";
const CUSTOM_PRESET_VALUE = "Custom";
const DYNAMIC_WIDGET_NAMES = ["width", "height"];
const SYNC_DELAYS_MS = [0, 60, 200, 600];
const PRESET_WATCH_INTERVAL_MS = 250;
const LEGACY_HIDDEN_TYPE = "ye_hidden";
const IS_V2_FRONTEND = typeof window !== "undefined" && !!window.comfyAPI; // Compability support for Node 2.0

const originalWidgetProps = new WeakMap();
const presetWatchers = new WeakMap();

function findWidgetByName(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function toggleWidget(node, widget, show = false) {
    if (!widget) {
        return;
    }

    // Node 2.0 (Vue) reads visibility from widget options/state.
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
    // Legacy frontend path: hide via widget type + size.
    // Vue frontend path: avoid type swapping because it can create hidden-but-clickable ghost rows.
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

    if (Array.isArray(widget.linkedWidgets)) {
        for (const linkedWidget of widget.linkedWidgets) {
            toggleWidget(node, linkedWidget, show);
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

function updatePresetWidgets(node) {
    const presetWidget = findWidgetByName(node, PRESET_WIDGET_NAME);
    if (!presetWidget) {
        return;
    }

    const rawPresetValue = presetWidget.value;
    const presetValue =
        typeof rawPresetValue === "string"
            ? rawPresetValue
            : String(rawPresetValue ?? "");
    const normalizedPreset = presetValue.trim();
    const showCustomDimensions =
        rawPresetValue == null ||
        normalizedPreset.length === 0 ||
        normalizedPreset === CUSTOM_PRESET_VALUE;
    for (const widgetName of DYNAMIC_WIDGET_NAMES) {
        toggleWidget(
            node,
            findWidgetByName(node, widgetName),
            showCustomDimensions,
        );
    }

    refreshNodeLayout(node);
}

function hookPresetWidget(node) {
    const presetWidget = findWidgetByName(node, PRESET_WIDGET_NAME);
    if (!presetWidget || presetWidget.__yeDynamicHooked) {
        scheduleSyncPasses(node);
        ensurePresetWatcher(node);
        return;
    }
    presetWidget.__yeDynamicHooked = true;

    const originalCallback = presetWidget.callback;
    presetWidget.callback = function patchedPresetCallback(value, ...args) {
        const result =
            typeof originalCallback === "function"
                ? originalCallback.call(this, value, ...args)
                : undefined;
        updatePresetWidgets(node);
        return result;
    };

    scheduleSyncPasses(node);
    ensurePresetWatcher(node);
}

function scheduleSyncPasses(node) {
    for (const delay of SYNC_DELAYS_MS) {
        window.setTimeout(() => {
            updatePresetWidgets(node);
        }, delay);
    }
}

function ensurePresetWatcher(node) {
    if (presetWatchers.has(node)) {
        return;
    }
    const timer = window.setInterval(() => {
        if (!node || !node.graph) {
            window.clearInterval(timer);
            presetWatchers.delete(node);
            return;
        }
        const presetWidget = findWidgetByName(node, PRESET_WIDGET_NAME);
        if (!presetWidget) {
            return;
        }
        const currentValue = String(presetWidget.value ?? "");
        if (node.__yeLastPresetValue !== currentValue) {
            node.__yeLastPresetValue = currentValue;
            updatePresetWidgets(node);
        }
    }, PRESET_WATCH_INTERVAL_MS);
    presetWatchers.set(node, timer);
}

app.registerExtension({
    name: EXTENSION_NAME,
    nodeCreated(node) {
        if (
            node?.comfyClass === TARGET_NODE_NAME ||
            node?.type === TARGET_NODE_NAME
        ) {
            hookPresetWidget(node);
        }
    },
    loadedGraphNode(node) {
        if (
            node?.comfyClass === TARGET_NODE_NAME ||
            node?.type === TARGET_NODE_NAME
        ) {
            hookPresetWidget(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE_NAME) {
            return;
        }

        const originalOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function patchedOnWidgetChanged() {
            const result = originalOnWidgetChanged?.apply(this, arguments);
            const [name] = arguments;
            if (name === PRESET_WIDGET_NAME) {
                scheduleSyncPasses(this);
            }
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            hookPresetWidget(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            hookPresetWidget(this);
            return result;
        };
    },
});
