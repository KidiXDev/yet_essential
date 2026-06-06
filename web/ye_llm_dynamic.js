import { app } from "../../scripts/app.js";
import { findWidgetByName, refreshNodeLayout, removeWidget } from "./shared/widget_utils.js";

const EXTENSION_NAME = "yet_essential.llm_dynamic_widgets";
const TARGET_NODE_NAMES = ["YELLMProvider"];
const PROVIDER_WIDGET_NAME = "provider";
const SYNC_DELAYS_MS = [0, 60, 200, 600];
const DEFAULT_MODEL_OPTIONS = ["Select a model"];

function isTargetNode(node) {
    const nodeName = node?.comfyClass || node?.type;
    return TARGET_NODE_NAMES.includes(nodeName);
}

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
    if (node.__yeLlmDynamicTemplates) {
        return node.__yeLlmDynamicTemplates;
    }

    node.__yeLlmDynamicTemplates = {
        base_url: captureTemplate(findWidgetByName(node, "base_url")),
        model: captureTemplate(findWidgetByName(node, "model")),
        custom_model: captureTemplate(findWidgetByName(node, "custom_model")),
    };

    if (node.__yeLlmDynamicTemplates.model) {
        node.__yeLlmDynamicTemplates.model.type = "combo";
        node.__yeLlmDynamicTemplates.model.options = {
            ...node.__yeLlmDynamicTemplates.model.options,
            values: Array.isArray(node.__yeLlmDynamicTemplates.model.options?.values)
                && node.__yeLlmDynamicTemplates.model.options.values.length > 0
                ? [...node.__yeLlmDynamicTemplates.model.options.values]
                : [...DEFAULT_MODEL_OPTIONS],
        };
    }
    return node.__yeLlmDynamicTemplates;
}

function ensureValues(node) {
    if (node.__yeLlmDynamicValues) {
        return node.__yeLlmDynamicValues;
    }

    const templates = ensureTemplates(node);
    node.__yeLlmDynamicValues = {
        base_url: findWidgetByName(node, "base_url")?.value ?? templates.base_url?.value ?? "",
        model: findWidgetByName(node, "model")?.value ?? templates.model?.value ?? "",
        custom_model: findWidgetByName(node, "custom_model")?.value ?? templates.custom_model?.value ?? "",
    };
    return node.__yeLlmDynamicValues;
}

function getProvider(node) {
    return String(findWidgetByName(node, PROVIDER_WIDGET_NAME)?.value || "")
        .trim()
        .toLowerCase();
}

function shouldShowWidget(node, widgetName) {
    const provider = getProvider(node);
    if (widgetName === "base_url") {
        return provider === "openai compatible";
    }
    if (widgetName === "model") {
        return provider === "openrouter" || provider === "nanogpt";
    }
    if (widgetName === "custom_model") {
        return provider === "openai compatible";
    }
    return true;
}

function addWidgetFromTemplate(node, name, template, value) {
    if (!template) {
        return null;
    }
    const options = template.options ? { ...template.options } : {};
    const callback = template.callback || (() => {});
    return node.addWidget(template.type || "text", name, value, callback, options);
}

function ensureWidgetMatchesTemplate(node, name) {
    const widget = findWidgetByName(node, name);
    const template = ensureTemplates(node)[name];
    const values = ensureValues(node);
    if (!widget || !template) {
        return widget;
    }
    if ((widget.type || "text") === (template.type || "text")) {
        return widget;
    }

    storeWidgetValue(node, name);
    removeWidget(node, widget);
    return addWidgetFromTemplate(node, name, template, values[name]);
}

function storeWidgetValue(node, name) {
    const values = ensureValues(node);
    const widget = findWidgetByName(node, name);
    if (widget) {
        values[name] = widget.value;
    }
}

function removeDynamicWidget(node, name) {
    storeWidgetValue(node, name);
    const widget = findWidgetByName(node, name);
    if (widget) {
        removeWidget(node, widget);
    }
}

function insertAfterProvider(node, widgets) {
    if (!Array.isArray(node?.widgets) || widgets.length === 0) {
        return;
    }

    for (const widget of widgets) {
        removeWidget(node, widget);
    }

    const desiredOrder = [
        PROVIDER_WIDGET_NAME,
        "base_url",
        "api_key",
        "timeout",
        "model",
        "custom_model",
    ];
    const lastWidget = [...node.widgets]
        .reverse()
        .find((widget) => desiredOrder.indexOf(String(widget?.name || "")) < desiredOrder.indexOf(String(widgets[0]?.name || "")));
    const insertIndex = lastWidget ? node.widgets.indexOf(lastWidget) + 1 : 0;
    node.widgets.splice(insertIndex, 0, ...widgets);
}

function syncNodeWidgets(node) {
    if (!isTargetNode(node)) {
        return;
    }

    ensureTemplates(node);
    ensureValues(node);

    const visibleWidgets = [];
    for (const name of ["base_url", "model", "custom_model"]) {
        if (shouldShowWidget(node, name)) {
            let widget = findWidgetByName(node, name);
            if (!widget) {
                const templates = ensureTemplates(node);
                const values = ensureValues(node);
                widget = addWidgetFromTemplate(node, name, templates[name], values[name]);
            } else {
                widget = ensureWidgetMatchesTemplate(node, name);
            }
            if (widget) {
                visibleWidgets.push(widget);
            }
        } else {
            removeDynamicWidget(node, name);
        }
    }

    if (visibleWidgets.length > 0) {
        visibleWidgets.sort((a, b) => {
            const order = ["base_url", "model", "custom_model"];
            return order.indexOf(a.name) - order.indexOf(b.name);
        });
        insertAfterProvider(node, visibleWidgets);
    }

    refreshNodeLayout(node);
}

function scheduleSyncPasses(node) {
    for (const delay of SYNC_DELAYS_MS) {
        window.setTimeout(() => {
            syncNodeWidgets(node);
        }, delay);
    }
}

function hookProviderWidget(node) {
    const providerWidget = findWidgetByName(node, PROVIDER_WIDGET_NAME);
    if (!providerWidget || providerWidget.__yeLlmDynamicHooked) {
        scheduleSyncPasses(node);
        return;
    }

    providerWidget.__yeLlmDynamicHooked = true;
    ensureTemplates(node);
    ensureValues(node);

    const originalCallback = providerWidget.callback;
    providerWidget.callback = function patchedProviderCallback(value, ...args) {
        const result =
            typeof originalCallback === "function"
                ? originalCallback.call(this, value, ...args)
                : undefined;
        syncNodeWidgets(node);
        return result;
    };

    scheduleSyncPasses(node);
}

app.registerExtension({
    name: EXTENSION_NAME,
    nodeCreated(node) {
        if (isTargetNode(node)) {
            hookProviderWidget(node);
        }
    },
    loadedGraphNode(node) {
        if (isTargetNode(node)) {
            hookProviderWidget(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODE_NAMES.includes(nodeData.name)) {
            return;
        }

        const originalOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function patchedOnWidgetChanged(name) {
            const result = originalOnWidgetChanged?.apply(this, arguments);
            if (name === PROVIDER_WIDGET_NAME) {
                scheduleSyncPasses(this);
            }
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            hookProviderWidget(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            hookProviderWidget(this);
            return result;
        };
    },
});
