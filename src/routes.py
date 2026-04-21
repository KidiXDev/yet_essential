import os
from aiohttp import web
from server import PromptServer
from .core import BASE_DIR, SETTINGS, TAG_INDEX, MODEL_PREVIEW_MANAGER

@PromptServer.instance.routes.get("/yet_essential/autocomplete/search")
async def search_autocomplete(request: web.Request) -> web.Response:
    query = request.query.get("q", "")
    try:
        requested_limit = int(request.query.get("limit", str(SETTINGS.limit)))
    except ValueError:
        requested_limit = SETTINGS.limit

    limit = min(requested_limit, SETTINGS.limit)
    return web.json_response(
        {
            "query": query,
            "settings": {
                "show_post_count": SETTINGS.show_post_count,
                "spacing_mode": SETTINGS.spacing_mode,
                "insertion_suffix": SETTINGS.insertion_suffix,
                "smart_suffix": SETTINGS.smart_suffix,
                "escape_parentheses": SETTINGS.escape_parentheses,
            },
            "items": TAG_INDEX.search(
                query=query,
                limit=limit,
                algorithm=SETTINGS.algorithm,
                sort_mode=SETTINGS.sort_mode
            ),
        }
    )


@PromptServer.instance.routes.get("/yet_essential/model/preview")
async def get_model_preview(request: web.Request) -> web.Response:
    folder_type = request.query.get("type", "")
    model_name = request.query.get("name", "")
    res = request.query.get("res", None)
    try:
        if res: res = int(res)
    except ValueError:
        res = None

    if not folder_type or not model_name:
        return web.Response(status=400)

    preview_path = MODEL_PREVIEW_MANAGER.find_preview(folder_type, model_name, res=res)
    if not preview_path or not os.path.exists(preview_path):
        return web.Response(status=404)

    return web.FileResponse(preview_path)


@PromptServer.instance.routes.get("/yet_essential/model/list")
async def get_model_list(request: web.Request) -> web.Response:
    folder_type = request.query.get("type", "")
    if not folder_type:
        return web.Response(status=400)

    models = MODEL_PREVIEW_MANAGER.list_models_with_previews(folder_type)
    return web.json_response(models)


@PromptServer.instance.routes.post("/yet_essential/settings/update")
async def update_settings(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400)

    old_csv = SETTINGS.csv_file
    SETTINGS.update(data)

    if SETTINGS.csv_file != old_csv:
        TAG_INDEX.update_path(BASE_DIR / "config" / "tag" / SETTINGS.csv_file)

    return web.Response(status=200)


@PromptServer.instance.routes.get("/yet_essential/settings/get")
async def get_settings(request: web.Request) -> web.Response:
    return web.json_response({
        "search_algorithm": SETTINGS.algorithm,
        "search_limit": SETTINGS.limit,
        "sort_mode": SETTINGS.sort_mode,
        "insertion_suffix": SETTINGS.insertion_suffix,
        "spacing_mode": SETTINGS.spacing_mode,
        "escape_parentheses": SETTINGS.escape_parentheses,
        "show_post_count": SETTINGS.show_post_count,
        "smart_suffix": SETTINGS.smart_suffix,
        "csv_file": SETTINGS.csv_file,
    })


@PromptServer.instance.routes.get("/yet_essential/tags/list")
async def list_tags(request: web.Request) -> web.Response:
    tag_dir = BASE_DIR / "config" / "tag"
    if not tag_dir.exists():
        return web.json_response([])

    files = [f.name for f in tag_dir.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
    return web.json_response(sorted(files))
