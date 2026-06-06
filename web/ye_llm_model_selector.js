import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { findWidgetByName, refreshNodeLayout } from "./shared/widget_utils.js";

const EXTENSION_NAME = "yet_essential.llm_model_selector";
const TARGET_NODE_NAME = "YELLMProvider";
const WATCHED_WIDGETS = ["provider", "base_url", "api_key", "timeout"];
const DEFAULT_OPTIONS = ["Select a model"];
const REMOTE_MODEL_PROVIDERS = new Set(["openrouter", "nanogpt"]);

function isTargetNode(node) {
    return node?.comfyClass === TARGET_NODE_NAME || node?.type === TARGET_NODE_NAME;
}

function setModelOptions(widget, options, value = null) {
    if (!widget) {
        return;
    }
    widget.options = widget.options || {};
    widget.options.values = options;
    widget.value = value ?? options[0] ?? "";
    if (typeof widget.callback === "function") {
        widget.callback(widget.value);
    }
}

function getWidgetValue(node, name) {
    return findWidgetByName(node, name)?.value ?? "";
}

function getFetchPayload(node) {
    return {
        provider: String(getWidgetValue(node, "provider") || "").trim(),
        base_url: String(getWidgetValue(node, "base_url") || "").trim(),
        api_key: String(getWidgetValue(node, "api_key") || "").trim(),
        timeout: Number(getWidgetValue(node, "timeout") || 60),
    };
}

async function refreshModelList(node) {
    if (!isTargetNode(node)) {
        return;
    }

    const modelWidget = findWidgetByName(node, "model");
    if (!modelWidget) {
        return;
    }

    const payload = getFetchPayload(node);
    const provider = payload.provider.toLowerCase();
    const currentValue = String(modelWidget.value || "").trim();
    const requestKey = JSON.stringify(payload);
    if (node.__yeModelSelectorRequestKey === requestKey && node.__yeModelSelectorLoaded) {
        return;
    }
    node.__yeModelSelectorRequestKey = requestKey;

    if (!REMOTE_MODEL_PROVIDERS.has(provider)) {
        setModelOptions(modelWidget, DEFAULT_OPTIONS, DEFAULT_OPTIONS[0]);
        node.__yeModelSelectorLoaded = false;
        refreshNodeLayout(node);
        return;
    }

    if (provider === "nanogpt" && !payload.api_key) {
        setModelOptions(modelWidget, ["Enter NanoGPT API key"], "Enter NanoGPT API key");
        node.__yeModelSelectorLoaded = false;
        refreshNodeLayout(node);
        return;
    }

    setModelOptions(modelWidget, ["Loading models..."], "Loading models...");
    refreshNodeLayout(node);

    try {
        const response = await api.fetchApi("/yet_essential/llm/models", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }

        const models = Array.isArray(data?.models) ? data.models : [];
        const values = models
            .map((item) => String(item?.id || "").trim())
            .filter(Boolean);
        if (values.length === 0) {
            setModelOptions(modelWidget, ["No models found"], "No models found");
            node.__yeModelSelectorLoaded = false;
            refreshNodeLayout(node);
            return;
        }

        modelWidget.options = modelWidget.options || {};
        modelWidget.options.values = values;
        const selected = values.includes(currentValue) ? currentValue : values[0];
        modelWidget.value = selected;
        node.__yeModelSelectorLoaded = true;
        refreshNodeLayout(node);
    } catch (error) {
        setModelOptions(modelWidget, ["Failed to load models"], "Failed to load models");
        modelWidget.options.yeError = String(error?.message || error || "Unknown error");
        node.__yeModelSelectorLoaded = false;
        refreshNodeLayout(node);
    }
}

function scheduleRefresh(node) {
    window.clearTimeout(node.__yeModelSelectorTimer);
    node.__yeModelSelectorTimer = window.setTimeout(() => {
        refreshModelList(node);
    }, 150);
}

function hookNode(node) {
    if (!isTargetNode(node) || node.__yeModelSelectorHooked) {
        if (isTargetNode(node)) {
            scheduleRefresh(node);
        }
        return;
    }

    node.__yeModelSelectorHooked = true;
    for (const name of WATCHED_WIDGETS) {
        const widget = findWidgetByName(node, name);
        if (!widget || widget.__yeModelSelectorWidgetHooked) {
            continue;
        }
        widget.__yeModelSelectorWidgetHooked = true;
        const originalCallback = widget.callback;
        widget.callback = function patchedModelSelectorWidgetCallback(value, ...args) {
            const result =
                typeof originalCallback === "function"
                    ? originalCallback.call(this, value, ...args)
                    : undefined;
            scheduleRefresh(node);
            return result;
        };
    }

    const modelWidget = findWidgetByName(node, "model");
    if (modelWidget && Array.isArray(modelWidget.options?.values)) {
        setModelOptions(modelWidget, DEFAULT_OPTIONS, DEFAULT_OPTIONS[0]);
    }

    scheduleRefresh(node);
}

app.registerExtension({
    name: EXTENSION_NAME,
    nodeCreated(node) {
        hookNode(node);
    },
    loadedGraphNode(node) {
        hookNode(node);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            hookNode(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            hookNode(this);
            return result;
        };

        const originalOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function patchedOnWidgetChanged(name) {
            const result = originalOnWidgetChanged?.apply(this, arguments);
            if (WATCHED_WIDGETS.includes(String(name || ""))) {
                scheduleRefresh(this);
            }
            return result;
        };
    },
});
