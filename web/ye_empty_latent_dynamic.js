import { app } from "../../scripts/app.js";
import { findWidgetByName, refreshNodeLayout, removeWidget } from "./shared/widget_utils.js";

const EXTENSION_NAME = "yet_essential.empty_latent_dynamic_widgets";
const TARGET_NODE_NAME = "YEEmptyLatentImage";
const PRESET_WIDGET_NAME = "preset";
const BATCH_WIDGET_NAME = "batch_size";
const CUSTOM_PRESET_VALUE = "Custom";
const DYNAMIC_WIDGET_NAMES = ["width", "height"];
const SYNC_DELAYS_MS = [0, 60, 200, 600];
const PRESET_WATCH_INTERVAL_MS = 250;

const presetWatchers = new WeakMap();

function captureTemplate(widget) {
    if (!widget) {
        return null;
    }
    return {
        type: widget.type,
        value: widget.value,
        callback: widget.callback || (() => {}),
        options: widget.options ? { ...widget.options } : {},
    };
}

function ensureTemplates(node) {
    if (node.__yeEmptyLatentTemplates) {
        return node.__yeEmptyLatentTemplates;
    }

    node.__yeEmptyLatentTemplates = {
        width: captureTemplate(findWidgetByName(node, "width")),
        height: captureTemplate(findWidgetByName(node, "height")),
    };
    return node.__yeEmptyLatentTemplates;
}

function ensureDimensionValues(node) {
    if (!node.__yeDimensionValues) {
        const templates = ensureTemplates(node);
        node.__yeDimensionValues = {
            width: findWidgetByName(node, "width")?.value ?? templates.width?.value ?? 1024,
            height: findWidgetByName(node, "height")?.value ?? templates.height?.value ?? 1024,
        };
    }
    return node.__yeDimensionValues;
}

function shouldShowCustomDimensions(node) {
    const presetWidget = findWidgetByName(node, PRESET_WIDGET_NAME);
    if (!presetWidget) {
        return true;
    }

    const rawPresetValue = presetWidget.value;
    const presetValue =
        typeof rawPresetValue === "string"
            ? rawPresetValue
            : String(rawPresetValue ?? "");
    const normalizedPreset = presetValue.trim();
    return (
        rawPresetValue == null ||
        normalizedPreset.length === 0 ||
        normalizedPreset === CUSTOM_PRESET_VALUE
    );
}

function enforceWidgetOrder(node) {
    if (!Array.isArray(node.widgets)) {
        return;
    }

    const orderedWidgets = [
        findWidgetByName(node, PRESET_WIDGET_NAME),
        findWidgetByName(node, BATCH_WIDGET_NAME),
        findWidgetByName(node, "width"),
        findWidgetByName(node, "height"),
    ].filter(Boolean);

    if (orderedWidgets.length < 2) {
        return;
    }

    const indexes = orderedWidgets
        .map((widget) => node.widgets.indexOf(widget))
        .filter((index) => index >= 0);
    if (indexes.length === 0) {
        return;
    }

    const insertIndex = Math.min(...indexes);
    for (const widget of orderedWidgets) {
        removeWidget(node, widget);
    }

    node.widgets.splice(insertIndex, 0, ...orderedWidgets);
}

function addDimensionWidget(node, name, value, template) {
    const options = template?.options ? { ...template.options } : {};
    const callback = template?.callback || (() => {});
    return node.addWidget(template?.type || "number", name, value, callback, options);
}

function removeDimensions(node) {
    const values = ensureDimensionValues(node);
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");

    if (widthWidget) {
        values.width = widthWidget.value;
        removeWidget(node, widthWidget);
    }
    if (heightWidget) {
        values.height = heightWidget.value;
        removeWidget(node, heightWidget);
    }
}

function addDimensions(node) {
    const templates = ensureTemplates(node);
    const values = ensureDimensionValues(node);
    const created = [];

    if (!findWidgetByName(node, "width")) {
        created.push(addDimensionWidget(node, "width", values.width, templates.width));
    }
    if (!findWidgetByName(node, "height")) {
        created.push(addDimensionWidget(node, "height", values.height, templates.height));
    }

    if (created.length > 0) {
        enforceWidgetOrder(node);
    }
}

function dimensionsMatchDesiredState(node) {
    const shouldShow = shouldShowCustomDimensions(node);
    const hasWidth = !!findWidgetByName(node, "width");
    const hasHeight = !!findWidgetByName(node, "height");
    return shouldShow ? hasWidth && hasHeight : !hasWidth && !hasHeight;
}

function updatePresetWidgets(node) {
    if (shouldShowCustomDimensions(node)) {
        addDimensions(node);
    } else {
        removeDimensions(node);
    }

    enforceWidgetOrder(node);
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
    ensureTemplates(node);
    ensureDimensionValues(node);

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
        if (
            node.__yeLastPresetValue !== currentValue ||
            !dimensionsMatchDesiredState(node)
        ) {
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
