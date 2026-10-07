from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import ERRORS, router
from app.services.model_service import ModelNotReady, ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("api")


def create_app(config_path=None) -> FastAPI:
    app = FastAPI(title="Fake Review Intelligence API", version="2.0.0",
                  description="Calibrated fake-review classification with explanations. Predictions are model estimates, not proof.")
    app.state.service = ModelService(config_path)
    app.include_router(router)
    index = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(index)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        msgs = [f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": msgs})

    @app.exception_handler(KeyError)
    async def _unknown_model(request: Request, exc: KeyError):
        return JSONResponse(status_code=422, content={"error": "unknown_model", "detail": [f"Unknown model {exc}. See GET /models."]})

    @app.exception_handler(ModelNotReady)
    async def _nm(request: Request, exc: ModelNotReady):
        return JSONResponse(status_code=503, content={"error": "model_not_ready", "detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        ERRORS.inc()
        log.exception("unhandled error on %s", request.url.path)  # stack trace stays in server logs only
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Unexpected server error."})

    return app


app = create_app()
