import { app } from "../../scripts/app.js";
import { findWidgetByName, refreshNodeLayout, removeWidget } from "./shared/widget_utils.js";

const EXTENSION_NAME = "yet_essential.lora_stack_dynamic_widgets";
const TARGET_NODE_NAMES = ["YELoraStack", "YELoraStackModel"];
const SLOT_NONE = "None";
const DEFAULT_STRENGTH = 1.0;

function getSlotWidgetNames(node, index) {
    const names = [
        `enabled_${index}`,
        `lora_name_${index}`,
        `strength_model_${index}`,
    ];

    const nodeName = node?.comfyClass || node?.type;
    if (nodeName !== "YELoraStackModel") {
        names.push(`strength_clip_${index}`);
    }

    return names;
}

function getSlotWidgets(node, index) {
    return getSlotWidgetNames(node, index)
        .map((name) => findWidgetByName(node, name))
        .filter(Boolean);
}

function countLoraRows(node) {
    let maxIndex = 0;
    for (const widget of node?.widgets || []) {
        const match = widget?.name?.match(/^lora_name_(\d+)$/);
        if (match) {
            maxIndex = Math.max(maxIndex, Number.parseInt(match[1], 10));
        }
    }
    return maxIndex;
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
    if (node.__yeLoraTemplates) {
        return node.__yeLoraTemplates;
    }

    const templates = {
        enabled: captureTemplate(findWidgetByName(node, "enabled_1")),
        loraName: captureTemplate(findWidgetByName(node, "lora_name_1")),
        strengthModel: captureTemplate(findWidgetByName(node, "strength_model_1")),
        strengthClip: captureTemplate(findWidgetByName(node, "strength_clip_1")),
    };

    node.__yeLoraTemplates = templates;
    return templates;
}

function hookNameWidget(node, widget) {
    if (!widget || widget.__yeLoraSlotHooked) {
        return;
    }

    widget.__yeLoraSlotHooked = true;
    const originalCallback = widget.callback;
    widget.callback = function patchedLoraNameCallback(value, ...args) {
        const result = typeof originalCallback === "function"
            ? originalCallback.call(this, value, ...args)
            : undefined;
        if (node.updateRemoveBtn) {
            node.updateRemoveBtn();
        }
        return result;
    };
}

function addToggleWidget(node, name, value, template) {
    const options = template?.options ? { ...template.options } : {};
    const callback = template?.callback || (() => {});
    return node.addWidget(template?.type || "toggle", name, value, callback, options);
}

function addComboWidget(node, name, value, template) {
    const options = template?.options ? { ...template.options } : {};
    const callback = template?.callback || (() => {});
    return node.addWidget("combo", name, value, callback, options);
}

function addNumberWidget(node, name, value, template) {
    const options = template?.options ? { ...template.options } : {};
    const callback = template?.callback || (() => {});
    return node.addWidget("number", name, value, callback, options);
}

function addLoraRow(
    node,
    index,
    {
        enabled = true,
        loraName = null,
        strengthModel = DEFAULT_STRENGTH,
        strengthClip = DEFAULT_STRENGTH,
    } = {},
) {
    const templates = ensureTemplates(node);
    const defaultLora = templates.loraName?.options?.values?.[0] ?? SLOT_NONE;

    addToggleWidget(node, `enabled_${index}`, enabled, templates.enabled);
    const combo = addComboWidget(
        node,
        `lora_name_${index}`,
        loraName ?? defaultLora,
        templates.loraName,
    );
    addNumberWidget(
        node,
        `strength_model_${index}`,
        strengthModel,
        templates.strengthModel,
    );

    const nodeName = node?.comfyClass || node?.type;
    if (nodeName !== "YELoraStackModel") {
        addNumberWidget(
            node,
            `strength_clip_${index}`,
            strengthClip,
            templates.strengthClip,
        );
    }

    hookNameWidget(node, combo);
}

function clearSlotValues(node, index) {
    const enabledWidget = findWidgetByName(node, `enabled_${index}`);
    const nameWidget = findWidgetByName(node, `lora_name_${index}`);
    const strengthModelWidget = findWidgetByName(node, `strength_model_${index}`);
    const strengthClipWidget = findWidgetByName(node, `strength_clip_${index}`);

    if (enabledWidget) {
        enabledWidget.value = true;
    }
    if (nameWidget) {
        nameWidget.value = SLOT_NONE;
    }
    if (strengthModelWidget) {
        strengthModelWidget.value = DEFAULT_STRENGTH;
    }
    if (strengthClipWidget) {
        strengthClipWidget.value = DEFAULT_STRENGTH;
    }
}

function removeLoraRow(node, index) {
    clearSlotValues(node, index);
    for (const widget of getSlotWidgets(node, index)) {
        removeWidget(node, widget);
    }
}

function parseSavedSlots(node, values) {
    const rows = [];
    if (!Array.isArray(values)) {
        return rows;
    }

    const chunkSize = (node?.comfyClass || node?.type) === "YELoraStackModel" ? 3 : 4;
    for (let offset = 0; offset < values.length; offset += 1) {
        const enabled = values[offset];
        const loraName = values[offset + 1];
        const strengthModel = values[offset + 2];
        const strengthClip = chunkSize === 4 ? values[offset + 3] : DEFAULT_STRENGTH;

        const isChunkStart = typeof enabled === "boolean"
            && typeof loraName === "string"
            && typeof strengthModel === "number"
            && (chunkSize === 3 || typeof strengthClip === "number");

        if (!isChunkStart) {
            continue;
        }

        rows.push({
            enabled,
            loraName,
            strengthModel,
            strengthClip: chunkSize === 4 ? strengthClip : DEFAULT_STRENGTH,
        });

        offset += chunkSize - 1;
    }

    return rows;
}

function ensureLoraRows(node, count) {
    for (let index = countLoraRows(node) + 1; index <= count; index += 1) {
        addLoraRow(node, index);
    }
}

function applySavedRows(node, rows) {
    for (let i = 0; i < rows.length; i += 1) {
        const index = i + 1;
        const row = rows[i];
        const enabledWidget = findWidgetByName(node, `enabled_${index}`);
        const nameWidget = findWidgetByName(node, `lora_name_${index}`);
        const strengthModelWidget = findWidgetByName(node, `strength_model_${index}`);
        const strengthClipWidget = findWidgetByName(node, `strength_clip_${index}`);

        if (enabledWidget) {
            enabledWidget.value = row.enabled;
        }
        if (nameWidget) {
            nameWidget.value = row.loraName;
        }
        if (strengthModelWidget) {
            strengthModelWidget.value = row.strengthModel;
        }
        if (strengthClipWidget) {
            strengthClipWidget.value = row.strengthClip;
        }
    }
}

function ensureButtonOrder(node) {
    const addBtn = node.__yeAddLoraButton;
    const removeBtn = node.__yeRemoveLoraButton;

    if (!addBtn) {
        return;
    }

    removeWidget(node, addBtn);
    node.widgets.unshift(addBtn);

    if (removeBtn) {
        removeWidget(node, removeBtn);
        node.widgets.push(removeBtn);
    }
}

function removeInitialExtraRows(node) {
    for (let index = countLoraRows(node); index >= 2; index -= 1) {
        for (const widget of getSlotWidgets(node, index)) {
            removeWidget(node, widget);
        }
    }
}

function ensureButtons(node) {
    if (!node.__yeAddLoraButton) {
        const addBtn = node.addWidget("button", "Add LoRA", "Add LoRA", () => {
            const nextIndex = countLoraRows(node) + 1;
            addLoraRow(node, nextIndex);
            node.updateRemoveBtn?.();
            refreshNodeLayout(node);
        });
        addBtn.serialize = false;
        node.__yeAddLoraButton = addBtn;
    }

    if (!node.__yeRemoveLoraButton) {
        const removeBtn = node.addWidget("button", "Remove LoRA", "Remove LoRA", () => {
            const maxIndex = countLoraRows(node);
            if (maxIndex <= 1) {
                return;
            }
            removeLoraRow(node, maxIndex);
            node.updateRemoveBtn?.();
            refreshNodeLayout(node);
        });
        removeBtn.serialize = false;
        node.__yeRemoveLoraButton = removeBtn;
    }
}

function hookLoraStackNode(node) {
    if (!node || node.__yeLoraStackHooked) {
        if (node?.updateRemoveBtn) {
            node.updateRemoveBtn();
            refreshNodeLayout(node);
        }
        return;
    }

    node.__yeLoraStackHooked = true;
    ensureTemplates(node);
    hookNameWidget(node, findWidgetByName(node, "lora_name_1"));
    removeInitialExtraRows(node);

    node.updateRemoveBtn = () => {
        const rowCount = countLoraRows(node);
        const removeBtn = node.__yeRemoveLoraButton;
        if (!removeBtn) {
            return;
        }

        if (rowCount > 1) {
            if (!node.widgets.includes(removeBtn)) {
                node.widgets.push(removeBtn);
            }
        } else {
            removeWidget(node, removeBtn);
        }

        ensureButtonOrder(node);
    };

    ensureButtons(node);
    node.updateRemoveBtn();
    refreshNodeLayout(node);
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

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function patchedOnNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            hookLoraStackNode(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function patchedOnConfigure(info) {
            const savedRows = parseSavedSlots(this, info?.widgets_values);
            const requiredRows = Math.max(1, savedRows.length);

            if (info?.widgets_values) {
                ensureLoraRows(this, requiredRows);
                this.updateRemoveBtn?.();
            }

            const result = originalOnConfigure?.apply(this, arguments);
            applySavedRows(this, savedRows);
            this.__yeAddLoraButton && (this.__yeAddLoraButton.value = "Add LoRA");
            this.__yeRemoveLoraButton && (this.__yeRemoveLoraButton.value = "Remove LoRA");
            this.updateRemoveBtn?.();
            refreshNodeLayout(this);
            return result;
        };
    },
});
