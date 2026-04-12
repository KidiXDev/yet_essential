import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const EXTENSION_NAME = "yet_essential.prompt_autocomplete";
const TARGET_NODE_NAME = "YEPrompt";
const TARGET_WIDGET_NAME = "prompt";
const SEARCH_LIMIT = 200;
const SEARCH_DEBOUNCE_MS = 120;
const PROMPT_PLACEHOLDER_HINT = "Prompt...";
const ATTACH_RETRY_INTERVAL_MS = 150;
const ATTACH_RETRY_MAX_ATTEMPTS = 80;
const TARGET_INPUT_ATTR = "data-ye-autocomplete-target";

let styleLoaded = false;
let globalHooksInstalled = false;
let comfyStringPatched = false;
let comfyTextareaPatched = false;
let graphWidgetPatched = false;

function ensureStyle() {
    if (styleLoaded) {
        return;
    }

    const style = document.createElement("style");
    style.id = "ye-prompt-autocomplete-style";
    style.textContent = `
    .ye-autocomplete-dropdown {
      position: fixed;
      z-index: 2147483647;
      max-height: 260px;
      overflow-y: auto;
      background: #1d1f24;
      border: 1px solid #4e5564;
      border-radius: 6px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
      display: none;
    }

    .ye-autocomplete-item {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      line-height: 1.3;
      color: #e5e9f0;
    }

    .ye-autocomplete-item:hover,
    .ye-autocomplete-item.ye-autocomplete-item--active {
      background: #2c313b;
    }

    .ye-autocomplete-main {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 600;
    }

    .ye-autocomplete-meta {
      color: #9ca6b8;
      font-size: 11px;
      white-space: nowrap;
    }
  `;

    document.head.appendChild(style);
    styleLoaded = true;
}

function normalizeQuery(value) {
    return value.trim().toLowerCase().replaceAll(" ", "_");
}

class PromptAutocompleteController {
    constructor(inputEl) {
        this.inputEl = inputEl;
        this.dropdownEl = document.createElement("div");
        this.dropdownEl.className = "ye-autocomplete-dropdown";
        document.body.appendChild(this.dropdownEl);

        this.items = [];
        this.activeIndex = -1;
        this.visible = false;
        this.debounceTimer = null;
        this.abortController = null;
        this.cachedRange = null;
        this.lastSearchQuery = "";
        this.settings = {
            spacing_mode: "space",
            insertion_suffix: ", ",
            escape_parentheses: true,
        };

        this.boundOnInput = this.onInput.bind(this);
        this.boundOnKeyUp = this.onInput.bind(this);
        this.boundOnKeyDown = this.onKeyDown.bind(this);
        this.boundOnBlur = this.onBlur.bind(this);
        this.boundOnFocus = this.onFocus.bind(this);
        this.boundOnResize = this.onViewportChanged.bind(this);
        this.boundOnScroll = this.onViewportChanged.bind(this);

        this.inputEl.addEventListener("input", this.boundOnInput);
        this.inputEl.addEventListener("keyup", this.boundOnKeyUp);
        this.inputEl.addEventListener("compositionend", this.boundOnInput);
        this.inputEl.addEventListener("keydown", this.boundOnKeyDown);
        this.inputEl.addEventListener("blur", this.boundOnBlur);
        this.inputEl.addEventListener("focus", this.boundOnFocus);
        window.addEventListener("resize", this.boundOnResize);
        window.addEventListener("scroll", this.boundOnScroll, true);
    }

    destroy() {
        this.hide();
        this.inputEl.removeEventListener("input", this.boundOnInput);
        this.inputEl.removeEventListener("keyup", this.boundOnKeyUp);
        this.inputEl.removeEventListener("compositionend", this.boundOnInput);
        this.inputEl.removeEventListener("keydown", this.boundOnKeyDown);
        this.inputEl.removeEventListener("blur", this.boundOnBlur);
        this.inputEl.removeEventListener("focus", this.boundOnFocus);
        window.removeEventListener("resize", this.boundOnResize);
        window.removeEventListener("scroll", this.boundOnScroll, true);
        this.dropdownEl.remove();
    }

    onFocus() {
        this.scheduleSearch();
    }

    onBlur() {
        window.setTimeout(() => this.hide(), 100);
    }

    onInput(event) {
        if (
            event?.key &&
            ["ArrowUp", "ArrowDown", "Enter", "Tab", "Escape"].includes(
                event.key,
            )
        ) {
            return;
        }
        this.scheduleSearch();
    }

    onViewportChanged() {
        if (!this.visible) {
            return;
        }
        this.positionDropdown();
    }

    onKeyDown(event) {
        if (!this.visible || this.items.length === 0) {
            return;
        }

        if (event.key === "ArrowDown") {
            event.preventDefault();
            this.setActiveIndex((this.activeIndex + 1) % this.items.length);
            return;
        }

        if (event.key === "ArrowUp") {
            event.preventDefault();
            const nextIndex =
                this.activeIndex <= 0
                    ? this.items.length - 1
                    : this.activeIndex - 1;
            this.setActiveIndex(nextIndex);
            return;
        }

        if (event.key === "Enter" || event.key === "Tab") {
            event.preventDefault();
            const selectedIndex = this.activeIndex >= 0 ? this.activeIndex : 0;
            this.insertItem(selectedIndex);
            return;
        }

        if (event.key === "Escape") {
            this.hide();
        }
    }

    scheduleSearch() {
        if (this.debounceTimer !== null) {
            window.clearTimeout(this.debounceTimer);
        }

        this.debounceTimer = window.setTimeout(() => {
            void this.fetchAndRender();
        }, SEARCH_DEBOUNCE_MS);
    }

    getTokenRange() {
        const fullText = this.inputEl.value ?? "";
        const cursor = this.inputEl.selectionStart ?? fullText.length;

        let start = cursor;
        while (
            start > 0 &&
            fullText[start - 1] !== "," &&
            fullText[start - 1] !== "\n"
        ) {
            start -= 1;
        }

        let end = cursor;
        while (
            end < fullText.length &&
            fullText[end] !== "," &&
            fullText[end] !== "\n"
        ) {
            end += 1;
        }

        const tokenFragment = fullText.slice(start, cursor);
        const query = normalizeQuery(tokenFragment);
        if (!query) {
            return null;
        }

        return { start, end, query, tokenFragment };
    }

    async fetchAndRender() {
        const tokenRange = this.getTokenRange();
        if (!tokenRange) {
            this.hide();
            return;
        }

        this.cachedRange = tokenRange;

        if (this.abortController) {
            this.abortController.abort();
        }
        this.abortController = new AbortController();

        const url = `/yet_essential/autocomplete/search?q=${encodeURIComponent(tokenRange.query)}&limit=${SEARCH_LIMIT}`;

        let response;
        try {
            response = await api.fetchApi(url, {
                cache: "no-store",
                signal: this.abortController.signal,
            });
        } catch (error) {
            if (error?.name !== "AbortError") {
                console.error(`[${EXTENSION_NAME}] search failed`, error);
            }
            return;
        }

        if (!response.ok) {
            this.hide();
            return;
        }

        const payload = await response.json();
        const items = Array.isArray(payload?.items) ? payload.items : [];
        if (items.length === 0) {
            this.hide();
            return;
        }

        if (payload?.settings) {
            this.settings = payload.settings;
        }

        const isNewResults = tokenRange.query !== this.lastSearchQuery;
        this.lastSearchQuery = tokenRange.query;

        if (isNewResults) {
            this.activeIndex = 0;
        }

        this.renderItems(items, payload);
    }

    renderItems(items, payload) {
        this.items = items;
        this.dropdownEl.innerHTML = "";

        const categoryMap = {
            0: "General",
            1: "Artist",
            3: "Copyright",
            4: "Character",
            5: "Meta",
        };

        const showPostCount = !!payload?.settings?.show_post_count;
        const showCategoryId = !!payload?.settings?.show_category_id;

        items.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = "ye-autocomplete-item";
            if (index === this.activeIndex) {
                row.classList.add("ye-autocomplete-item--active");
            }

            const title = item.label || item.tag || item.insert_text || "";
            const metaParts = [];

            if (
                showPostCount &&
                Number.isFinite(item.total_post) &&
                item.total_post > 0
            ) {
                metaParts.push(`${item.total_post.toLocaleString()}`);
            }

            const categoryName =
                categoryMap[item.category] ||
                (Number.isFinite(item.category)
                    ? `Other (${item.category})`
                    : "");
            if (categoryName) {
                if (showCategoryId) {
                    metaParts.push(`${categoryName} (${item.category})`);
                } else {
                    metaParts.push(categoryName);
                }
            }

            const mainEl = document.createElement("div");
            mainEl.className = "ye-autocomplete-main";
            mainEl.textContent = title;

            const metaEl = document.createElement("div");
            metaEl.className = "ye-autocomplete-meta";
            metaEl.textContent = metaParts.join(" | ");

            row.appendChild(mainEl);
            row.appendChild(metaEl);

            row.addEventListener("mousedown", (event) => {
                event.preventDefault();
            });
            row.addEventListener("click", () => {
                this.insertItem(index);
            });

            this.dropdownEl.appendChild(row);
        });

        this.show();
        this.scrollActiveIntoView();
    }

    setActiveIndex(index) {
        this.activeIndex = index;
        const rows = this.dropdownEl.querySelectorAll(".ye-autocomplete-item");
        rows.forEach((row, rowIndex) => {
            row.classList.toggle(
                "ye-autocomplete-item--active",
                rowIndex === index,
            );
        });
        this.scrollActiveIntoView();
    }

    scrollActiveIntoView() {
        const activeRow = this.dropdownEl.querySelector(
            ".ye-autocomplete-item--active",
        );
        if (activeRow) {
            activeRow.scrollIntoView({ block: "nearest", behavior: "auto" });
        }
    }

    insertItem(index) {
        const item = this.items[index];
        if (!item) {
            return;
        }

        const tokenRange = this.getTokenRange() || this.cachedRange;
        if (!tokenRange) {
            this.hide();
            return;
        }

        let insertText = item.insert_text || item.label || item.tag;
        if (!insertText) {
            this.hide();
            return;
        }

        // Apply transformations based on settings
        if (this.settings.spacing_mode === "space") {
            insertText = insertText.replaceAll("_", " ");
        }

        if (this.settings.escape_parentheses) {
            insertText = insertText
                .replaceAll("(", "\\(")
                .replaceAll(")", "\\)");
        }

        const suffix =
            typeof this.settings.insertion_suffix === "string"
                ? this.settings.insertion_suffix
                : "";

        const fullText = this.inputEl.value ?? "";
        const before = fullText.slice(0, tokenRange.start);
        const after = fullText.slice(tokenRange.end);
        const leadingWhitespace =
            tokenRange.tokenFragment.match(/^\s*/)?.[0] ?? "";
        const replacement = `${leadingWhitespace}${insertText}${suffix}`;

        this.inputEl.value = `${before}${replacement}${after}`;
        const cursor = before.length + replacement.length;
        this.inputEl.selectionStart = cursor;
        this.inputEl.selectionEnd = cursor;
        this.inputEl.dispatchEvent(
            new Event("input", { bubbles: true, composed: true }),
        );
        this.inputEl.dispatchEvent(
            new Event("change", { bubbles: true, composed: true }),
        );
        this.inputEl.focus();

        this.hide();
    }

    show() {
        this.positionDropdown();
        this.dropdownEl.style.display = "block";
        this.visible = true;
    }

    hide() {
        this.dropdownEl.style.display = "none";
        this.dropdownEl.innerHTML = "";
        this.items = [];
        this.activeIndex = -1;
        this.visible = false;
    }

    positionDropdown() {
        const rect = this.inputEl.getBoundingClientRect();
        this.dropdownEl.style.left = `${Math.round(rect.left)}px`;
        this.dropdownEl.style.top = `${Math.round(rect.bottom + 4)}px`;
        this.dropdownEl.style.width = `${Math.round(rect.width)}px`;
    }
}

const controllerMap = new WeakMap();
const attachTimerMap = new WeakMap();
let activeController = null;

function isTextInputElement(inputEl) {
    return (
        inputEl instanceof HTMLInputElement ||
        inputEl instanceof HTMLTextAreaElement
    );
}

function markTargetInputElement(inputEl) {
    if (!isTextInputElement(inputEl)) {
        return;
    }
    inputEl.setAttribute(TARGET_INPUT_ATTR, "1");
}

function findInputDeep(root) {
    if (!root) {
        return null;
    }

    if (isTextInputElement(root)) {
        return root;
    }

    // Check shadow root first as Vue components often wrap their inputs here
    if (root.shadowRoot) {
        const found = findInputDeep(root.shadowRoot);
        if (found) {
            return found;
        }
    }

    // Check children
    const children = root.children || root.childNodes;
    if (children) {
        for (let i = 0; i < children.length; i++) {
            const found = findInputDeep(children[i]);
            if (found) {
                return found;
            }
        }
    }

    return null;
}

function isMarkedTargetInputElement(inputEl) {
    return (
        isTextInputElement(inputEl) &&
        inputEl.getAttribute(TARGET_INPUT_ATTR) === "1"
    );
}

function isTargetNode(node) {
    if (!node) {
        return false;
    }
    return (
        node.comfyClass === TARGET_NODE_NAME || node.type === TARGET_NODE_NAME
    );
}

function isLikelyTargetInput(inputEl) {
    if (!isTextInputElement(inputEl)) {
        return false;
    }

    if (isMarkedTargetInputElement(inputEl)) {
        return true;
    }

    const placeholder = (inputEl.placeholder || "").toLowerCase();
    if (placeholder.includes(PROMPT_PLACEHOLDER_HINT.toLowerCase())) {
        return true;
    }

    // Check for common Vue/V2 attributes or classes
    const dataWidgetName = (
        inputEl.dataset?.widgetName ||
        inputEl.getAttribute("data-widget-name") ||
        ""
    ).toLowerCase();
    const dataTestId = (inputEl.dataset?.testid || "").toLowerCase();
    const className = (inputEl.className || "").toLowerCase();
    const tagName = (inputEl.tagName || "").toLowerCase();

    if (
        dataWidgetName === TARGET_WIDGET_NAME &&
        (dataTestId === "dom-widget-textarea" || tagName === "textarea")
    ) {
        return true;
    }

    // Very aggressive check for V2 prompt textareas
    if (
        tagName === "textarea" &&
        (className.includes("comfy-") ||
            className.includes("vue-") ||
            className.includes("prompt") ||
            placeholder.includes("prompt"))
    ) {
        return true;
    }

    if (
        className.includes("comfy-multiline-input") ||
        className.includes("comfy-textarea") ||
        className.includes("comfy-input")
    ) {
        return true;
    }

    const inputName = (inputEl.name || "").toLowerCase();
    if (
        inputName === TARGET_WIDGET_NAME &&
        (placeholder.includes("autocomplete") || placeholder.includes("prompt"))
    ) {
        return true;
    }

    return false;
}

function attachPromptAutocomplete(inputEl) {
    if (!isTextInputElement(inputEl)) {
        return;
    }

    const existing = controllerMap.get(inputEl);
    if (existing) {
        activeController = existing;
        if (document.activeElement === inputEl) {
            existing.scheduleSearch();
        }
        return;
    }

    if (activeController && activeController.inputEl !== inputEl) {
        controllerMap.delete(activeController.inputEl);
        activeController.destroy();
        activeController = null;
    }

    const controller = new PromptAutocompleteController(inputEl);
    controllerMap.set(inputEl, controller);
    activeController = controller;

    if (document.activeElement === inputEl) {
        controller.scheduleSearch();
    }
}

function getInputFromWidget(widget) {
    if (!widget) {
        return null;
    }

    if (isTextInputElement(widget.inputEl)) {
        return widget.inputEl;
    }

    if (isTextInputElement(widget.element)) {
        return widget.element;
    }

    if (isTextInputElement(widget.input)) {
        return widget.input;
    }

    const root = widget.element || widget.input || widget.inputEl;
    if (root && typeof root.querySelector === "function") {
        const inputEl = findInputDeep(root);
        if (inputEl) {
            return inputEl;
        }
    }

    return null;
}

function getPromptInputElement(node) {
    const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
    if (widgets.length === 0) {
        return null;
    }

    const namedWidget = widgets.find(
        (candidate) =>
            String(candidate?.name || "").toLowerCase() === TARGET_WIDGET_NAME,
    );
    const namedInput = getInputFromWidget(namedWidget);
    if (namedInput) {
        return namedInput;
    }

    const textCandidates = [];
    for (const widget of widgets) {
        const inputEl = getInputFromWidget(widget);
        if (!isTextInputElement(inputEl)) {
            continue;
        }

        const widgetName = String(widget?.name || "").toLowerCase();
        if (widgetName === TARGET_WIDGET_NAME) {
            return inputEl;
        }

        if (inputEl.tagName === "TEXTAREA") {
            textCandidates.push(inputEl);
        }
    }

    if (textCandidates.length === 1) {
        return textCandidates[0];
    }

    return (
        textCandidates.find((inputEl) => isLikelyTargetInput(inputEl)) || null
    );
}

function findNodeIdFromElement(el) {
    let current = el;
    while (current) {
        if (current.dataset?.nodeId) {
            return current.dataset.nodeId;
        }
        const attrId =
            typeof current.getAttribute === "function"
                ? current.getAttribute("data-node-id")
                : null;
        if (attrId) {
            return attrId;
        }

        if (current.parentNode) {
            current = current.parentNode;
        } else if (current instanceof ShadowRoot) {
            current = current.host;
        } else {
            break;
        }
    }
    return null;
}

function markIfOwnedByAnyTargetNode(inputEl) {
    if (!isTextInputElement(inputEl)) {
        return false;
    }

    if (isMarkedTargetInputElement(inputEl)) {
        return true;
    }

    // Strategy 1: Hierarchy climbing (Fastest, best for V2/Vue)
    const nodeId = findNodeIdFromElement(inputEl);
    if (nodeId) {
        const node = app.graph.getNodeById(nodeId);
        if (node && isTargetNode(node)) {
            markTargetInputElement(inputEl);
            return true;
        }
    }

    // Strategy 2: Scan active nodes
    const graphNodes = app?.graph?._nodes;
    if (!Array.isArray(graphNodes)) {
        return false;
    }

    for (const node of graphNodes) {
        if (!isTargetNode(node)) {
            continue;
        }

        if (getPromptInputElement(node) === inputEl) {
            markTargetInputElement(inputEl);
            return true;
        }
    }

    return false;
}

function scheduleAttachFromNode(node) {
    if (!node || typeof node !== "object") {
        return;
    }

    const existingTimer = attachTimerMap.get(node);
    if (existingTimer) {
        window.clearInterval(existingTimer);
        attachTimerMap.delete(node);
    }

    let attempts = 0;
    const timer = window.setInterval(() => {
        attempts += 1;
        const inputEl = getPromptInputElement(node);
        if (inputEl) {
            markTargetInputElement(inputEl);
            attachPromptAutocomplete(inputEl);
            window.clearInterval(timer);
            attachTimerMap.delete(node);
            return;
        }

        if (attempts >= ATTACH_RETRY_MAX_ATTEMPTS) {
            window.clearInterval(timer);
            attachTimerMap.delete(node);
        }
    }, ATTACH_RETRY_INTERVAL_MS);

    attachTimerMap.set(node, timer);
}

function getInputFromComposedEvent(event) {
    const path =
        typeof event.composedPath === "function" ? event.composedPath() : [];
    for (const item of path) {
        if (isTextInputElement(item)) {
            return item;
        }
    }

    const target = event.target;
    if (isTextInputElement(target)) {
        return target;
    }

    return null;
}

function getDeepActiveElement() {
    let current = document.activeElement;
    while (current?.shadowRoot?.activeElement) {
        current = current.shadowRoot.activeElement;
    }
    return current;
}

function tryAttachFromElement(inputEl) {
    if (!isTextInputElement(inputEl)) {
        return false;
    }

    const isOwnedInput = markIfOwnedByAnyTargetNode(inputEl);
    if (isOwnedInput || isLikelyTargetInput(inputEl)) {
        markTargetInputElement(inputEl);
        attachPromptAutocomplete(inputEl);
        return true;
    }

    return false;
}

function installGlobalHooks() {
    if (globalHooksInstalled) {
        return;
    }
    globalHooksInstalled = true;

    const handleInteraction = (event) => {
        const target = getInputFromComposedEvent(event);
        if (tryAttachFromElement(target)) {
            return;
        }

        const active = getDeepActiveElement();
        tryAttachFromElement(active);
    };

    document.addEventListener("focusin", handleInteraction, true);
    document.addEventListener("click", handleInteraction, true);

    // As a last resort, check on keyup if we haven't attached yet
    document.addEventListener(
        "keyup",
        (event) => {
            const target = getInputFromComposedEvent(event);
            if (
                target &&
                isTextInputElement(target) &&
                !controllerMap.has(target)
            ) {
                handleInteraction(event);
            }
        },
        true,
    );
}

function shouldAttachFromNodeAndWidgetName(node, inputName) {
    if (!inputName) {
        return false;
    }

    const normalizedName = String(inputName).toLowerCase();
    if (normalizedName !== TARGET_WIDGET_NAME) {
        return false;
    }

    if (
        node?.comfyClass === TARGET_NODE_NAME ||
        node?.type === TARGET_NODE_NAME
    ) {
        return true;
    }

    return false;
}

function shouldAttachFromWidgetArgs(node, inputName, inputData) {
    if (shouldAttachFromNodeAndWidgetName(node, inputName)) {
        return true;
    }

    const widgetConfig = inputData?.[1];
    return widgetConfig?.["yet_essential.autocomplete"] === true;
}

function patchComfyStringWidget() {
    if (comfyStringPatched) {
        return;
    }

    const originalStringFactory = ComfyWidgets?.STRING;
    if (typeof originalStringFactory !== "function") {
        return;
    }

    ComfyWidgets.STRING = function patchedStringFactory(
        node,
        inputName,
        inputData,
    ) {
        const result = originalStringFactory.apply(this, arguments);

        try {
            if (shouldAttachFromWidgetArgs(node, inputName, inputData)) {
                const inputEl = getInputFromWidget(result?.widget);
                if (inputEl) {
                    markTargetInputElement(inputEl);
                    attachPromptAutocomplete(inputEl);
                } else if (node) {
                    scheduleAttachFromNode(node);
                }
            }
        } catch (error) {
            console.error(
                `[${EXTENSION_NAME}] failed to attach via ComfyWidgets.STRING`,
                error,
            );
        }

        return result;
    };

    comfyStringPatched = true;
}

function patchComfyTextareaWidget() {
    if (comfyTextareaPatched) {
        return;
    }

    const originalTextareaFactory = ComfyWidgets?.TEXTAREA;
    if (typeof originalTextareaFactory !== "function") {
        return;
    }

    ComfyWidgets.TEXTAREA = function patchedTextareaFactory(
        node,
        inputName,
        inputData,
    ) {
        const result = originalTextareaFactory.apply(this, arguments);

        try {
            if (shouldAttachFromWidgetArgs(node, inputName, inputData)) {
                const inputEl = getInputFromWidget(result?.widget);
                if (inputEl) {
                    markTargetInputElement(inputEl);
                    attachPromptAutocomplete(inputEl);
                } else if (node) {
                    scheduleAttachFromNode(node);
                }
            }
        } catch (error) {
            console.error(
                `[${EXTENSION_NAME}] failed to attach via ComfyWidgets.TEXTAREA`,
                error,
            );
        }

        return result;
    };

    comfyTextareaPatched = true;
}

function patchNodeWidgetFactories() {
    if (graphWidgetPatched) {
        return;
    }

    const nodeProto = globalThis?.LGraphNode?.prototype;
    if (!nodeProto) {
        return;
    }

    const originalAddWidget = nodeProto.addWidget;
    if (typeof originalAddWidget === "function") {
        nodeProto.addWidget = function patchedAddWidget(type, name) {
            const widget = originalAddWidget.apply(this, arguments);

            try {
                if (shouldAttachFromNodeAndWidgetName(this, name)) {
                    const inputEl = getInputFromWidget(widget);
                    if (inputEl) {
                        markTargetInputElement(inputEl);
                        attachPromptAutocomplete(inputEl);
                    } else {
                        scheduleAttachFromNode(this);
                    }
                }
            } catch (error) {
                console.error(
                    `[${EXTENSION_NAME}] failed to attach via LGraphNode.addWidget`,
                    error,
                );
            }

            return widget;
        };
    }

    const originalAddDOMWidget = nodeProto.addDOMWidget;
    if (typeof originalAddDOMWidget === "function") {
        nodeProto.addDOMWidget = function patchedAddDOMWidget(
            name,
            type,
            element,
        ) {
            const widget = originalAddDOMWidget.apply(this, arguments);

            try {
                if (shouldAttachFromNodeAndWidgetName(this, name)) {
                    if (isTextInputElement(element)) {
                        markTargetInputElement(element);
                        attachPromptAutocomplete(element);
                    } else {
                        const inputEl = getInputFromWidget(widget);
                        if (inputEl) {
                            markTargetInputElement(inputEl);
                            attachPromptAutocomplete(inputEl);
                        } else {
                            scheduleAttachFromNode(this);
                        }
                    }
                }
            } catch (error) {
                console.error(
                    `[${EXTENSION_NAME}] failed to attach via LGraphNode.addDOMWidget`,
                    error,
                );
            }

            return widget;
        };
    }

    graphWidgetPatched = true;
}

ensureStyle();

app.registerExtension({
    name: EXTENSION_NAME,
    setup() {
        installGlobalHooks();
        patchComfyStringWidget();
        patchComfyTextareaWidget();
        patchNodeWidgetFactories();
    },
    nodeCreated(node) {
        if (
            node?.comfyClass === TARGET_NODE_NAME ||
            node?.type === TARGET_NODE_NAME
        ) {
            scheduleAttachFromNode(node);
        }
    },
    loadedGraphNode(node) {
        if (
            node?.comfyClass === TARGET_NODE_NAME ||
            node?.type === TARGET_NODE_NAME
        ) {
            scheduleAttachFromNode(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            scheduleAttachFromNode(this);
            return result;
        };
    },
});
