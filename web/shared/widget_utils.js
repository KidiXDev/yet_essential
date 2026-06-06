import { app } from "../../../scripts/app.js";

export function findWidgetByName(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

export function removeWidget(node, widget) {
    if (!node?.widgets || !widget) {
        return;
    }
    const index = node.widgets.indexOf(widget);
    if (index !== -1) {
        node.widgets.splice(index, 1);
    }
}

export function refreshNodeLayout(node) {
    if (!node || typeof node.computeSize !== "function") {
        return;
    }
    const size = node.computeSize();
    node.size[0] = Math.max(node.size[0], size[0]);
    node.size[1] = size[1];
    node.setDirtyCanvas(true, true);
    app.canvas.setDirty(true, true);
}
