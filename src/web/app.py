"""
FastAPI web interface for running the HERMES Agentic pipeline locally.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fastapi import Response

from orchestration.pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="HERMES Agentic Web", version="1.0.0")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PipelineRequest(BaseModel):
    folder_path: str = Field(..., min_length=1, description="Absolute path to the folder")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/run_pipeline")
def run_pipeline_endpoint(request: PipelineRequest) -> dict:
    folder_path = request.folder_path.strip()
    if not folder_path:
        raise HTTPException(status_code=400, detail="folder_path cannot be empty.")

    try:
        results = run_pipeline(folder_path)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    payload = {
        "hierarchy": results.hierarchy.model_dump(),
        "categorizations": [item.model_dump() for item in results.categorizations],
        "reviewer": results.review.model_dump() if results.review else None,
    }

    return {"status": "success", "data": payload}


@app.get("/status")
def get_status() -> Response:
    status_path = BASE_DIR.parent.parent / "logs" / "pipeline_status.txt"
    if not status_path.exists():
        return Response(content="En attente", media_type="text/plain")
    with open(status_path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/plain")