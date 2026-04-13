import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "yet_essential.settings";

async function getTagFiles() {
    try {
        const response = await api.fetchApi("/yet_essential/tags/list", {
            cache: "no-store",
        });
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] failed to fetch tag files`, error);
    }
    return ["booru.csv"];
}

(async () => {
    const tagFiles = await getTagFiles();

    app.registerExtension({
        name: EXTENSION_NAME,
        settings: [
            {
                id: "yet_essential.search_algorithm",
                name: "Search Algorithm",
                type: "combo",
                defaultValue: "fuzzy",
                options: ["fuzzy", "contains", "prefix"],
                category: ["Yet Essential", "Search", "Algorithm"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ search_algorithm: newVal }),
                    });
                },
            },
            {
                id: "yet_essential.csv_file",
                name: "CSV File",
                type: "combo",
                defaultValue: "booru.csv",
                options: tagFiles,
                category: ["Yet Essential", "Search", "Source File"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ csv_file: newVal }),
                    });
                },
            },
            {
                id: "yet_essential.search_limit",
                name: "Search Limit",
                type: "number",
                defaultValue: 20,
                attrs: { min: 1, max: 200 },
                category: ["Yet Essential", "Search", "Limit"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ search_limit: newVal }),
                    });
                },
            },
            {
                id: "yet_essential.sort_mode",
                name: "Sort Mode",
                type: "combo",
                defaultValue: "score",
                options: ["score", "alphabet", "count"],
                category: ["Yet Essential", "Search", "Sort Priority"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ sort_mode: newVal }),
                    });
                },
            },
            {
                id: "yet_essential.insertion_suffix",
                name: "Insertion Suffix",
                type: "text",
                defaultValue: ", ",
                category: ["Yet Essential", "Formatting", "Suffix"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ insertion_suffix: newVal }),
                    });
                },
            },
            {
                id: "yet_essential.spacing_mode",
                name: "Spacing Mode",
                type: "combo",
                defaultValue: "space",
                options: ["space", "underscore"],
                category: ["Yet Essential", "Formatting", "Space Conversion"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ spacing_mode: newVal }),
                    });
                },
            },
            {
                id: "yet_essential.escape_parentheses",
                name: "Escape Parentheses",
                type: "boolean",
                defaultValue: true,
                category: ["Yet Essential", "Formatting", "Clean Escape"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ escape_parentheses: !!newVal }),
                    });
                },
            },
            {
                id: "yet_essential.smart_suffix",
                name: "Smart Suffix",
                type: "boolean",
                defaultValue: true,
                category: ["Yet Essential", "Formatting", "Smart Suffix"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ smart_suffix: !!newVal }),
                    });
                },
            },
            {
                id: "yet_essential.show_post_count",
                name: "Show Post Count",
                type: "boolean",
                defaultValue: false,
                category: ["Yet Essential", "UI", "Count Visibility"],
                onChange: (newVal) => {
                    api.fetchApi("/yet_essential/settings/update", {
                        method: "POST",
                        body: JSON.stringify({ show_post_count: !!newVal }),
                    });
                },
            },
        ],
        async setup() {
            // Sync settings from server
            try {
                const response = await api.fetchApi(
                    "/yet_essential/settings/get",
                    {
                        cache: "no-store",
                    },
                );
                if (response.ok) {
                    const settings = await response.json();
                    for (const [key, value] of Object.entries(settings)) {
                        const settingId = `yet_essential.${key}`;
                        const currentValue =
                            app.extensionManager.setting.get(settingId);
                        if (currentValue !== value) {
                            await app.extensionManager.setting.set(
                                settingId,
                                value,
                            );
                        }
                    }
                }
            } catch (error) {
                console.error(
                    `[${EXTENSION_NAME}] failed to sync settings`,
                    error,
                );
            }
        },
    });
})();
