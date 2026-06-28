"""FastAPI application factory + entrypoint.

Run:  uvicorn app.main:app --reload --port 8000
(from the backend/ directory, with backend/app on the path)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import get_settings
from .core.logging import get_logger

log = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")
    # CORS: allow the browser extension (any origin). Tighten in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
 allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    @app.get("/")
    def root():
        return {"message": "PageSense AI Backend is running!"}

    log.info("%s initialized (env=%s)", settings.app_name, settings.env)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
