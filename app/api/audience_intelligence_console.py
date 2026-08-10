from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.core.local_ui_session import (
    issue_local_ui_session,
    require_local_ui_request,
)


router = APIRouter(tags=["Audience Intelligence Workspace"])
STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
WORKSPACE_HTML = STATIC_ROOT / "audience_workspace.html"
WORKSPACE_CSS = STATIC_ROOT / "audience_workspace.css"
WORKSPACE_JS = STATIC_ROOT / "audience_workspace.js"
WORKSPACE_CSP = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'; "
    "form-action 'none'; "
    "object-src 'none'; "
    "img-src 'self' data:; "
    "style-src 'self'; "
    "script-src 'self'; "
    "connect-src 'self'"
)


def _workspace_headers(*, content_type: str) -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Content-Security-Policy": WORKSPACE_CSP,
        "Content-Type": content_type,
        "Cross-Origin-Resource-Policy": "same-origin",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }


@router.get("/audience-workspace", response_class=HTMLResponse)
def audience_workspace(request: Request) -> HTMLResponse:
    response = HTMLResponse(
        WORKSPACE_HTML.read_text(encoding="utf-8"),
        headers=_workspace_headers(
            content_type="text/html; charset=utf-8"
        ),
    )
    issue_local_ui_session(request, response)
    return response


@router.get(
    "/audience-workspace/assets/styles.css",
    dependencies=[Depends(require_local_ui_request)],
)
def audience_workspace_styles() -> FileResponse:
    return FileResponse(
        WORKSPACE_CSS,
        media_type="text/css",
        headers=_workspace_headers(
            content_type="text/css; charset=utf-8"
        ),
    )


@router.get(
    "/audience-workspace/assets/app.js",
    dependencies=[Depends(require_local_ui_request)],
)
def audience_workspace_script() -> FileResponse:
    return FileResponse(
        WORKSPACE_JS,
        media_type="application/javascript",
        headers=_workspace_headers(
            content_type="application/javascript; charset=utf-8"
        ),
    )
