"""Web UI for the statistical board.

Run:  python -m stat_board.webapp   (defaults to http://127.0.0.1:8643)

Endpoints:
  GET  /                        the single-page UI
  GET  /api/health              credentials + model info
  POST /api/runs                start a run {question, data, group_col?, value_col?, paired?, alpha?, rounds?, dry_run?}
  GET  /api/runs                list runs from this server session
  GET  /api/runs/{id}/events    SSE stream of progress events
  GET  /api/datasets            list CSV/JSON files under sample_data/ and reports/
  GET  /api/reports             list saved reports
  GET  /api/reports/{name}      serve a report (.pdf) or transcript (.json/.md)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from . import config, orchestrator
from . import transcript as transcript_mod

REPORTS_DIR = Path("reports")
UPLOAD_DIR = Path("uploads")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Statistical Board")


@dataclass
class Run:
    id: str
    question: str
    data: str
    group_col: str | None
    value_col: str | None
    paired: bool
    alpha: float
    rounds: int
    dry_run: bool
    status: str = "running"  # running | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    pdf_name: str | None = None
    transcript_name: str | None = None


RUNS: dict[str, Run] = {}


class RunRequest(BaseModel):
    question: str
    data: str
    group_col: str | None = None
    value_col: str | None = None
    paired: bool = False
    alpha: float = 0.05
    rounds: int | None = None
    dry_run: bool = False


class UploadRequest(BaseModel):
    filename: str
    content: str


def _push(run: Run, event: dict[str, Any]) -> None:
    run.events.append(event)
    run.changed.set()


async def _execute(run: Run) -> None:
    try:
        result = await orchestrator.run(
            run.question, run.data, group_col=run.group_col, value_col=run.value_col,
            paired=run.paired, alpha=run.alpha, max_rounds=run.rounds,
            dry_run=run.dry_run, on_event=lambda e: _push(run, e),
        )
        paths = transcript_mod.save(result, REPORTS_DIR)
        run.pdf_name = paths["pdf"].name
        run.transcript_name = paths["json"].name
        run.status = "done"
        _push(run, {"type": "done", "pdf": paths["pdf"].name,
                    "md": paths["md"].name, "transcript": paths["json"].name})
    except Exception as exc:
        run.status = "error"
        message = f"{type(exc).__name__}: {exc}"
        auth_markers = ("Could not resolve authentication", "Config file not found",
                        "authentication_error", "invalid x-api-key")
        if any(m in str(exc) for m in auth_markers):
            message = ("No Anthropic credentials found by the server process. Put "
                       "ANTHROPIC_API_KEY=sk-ant-... in a .env file where you launch the "
                       "server, export it in the shell, or run `ant auth login`, then "
                       "restart. Dry runs work without credentials.")
        _push(run, {"type": "error", "message": message})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"credentials": config.credentials_present(),
            "model": config.MODEL, "effort": config.EFFORT}


@app.get("/api/datasets")
async def datasets() -> list[str]:
    out: list[str] = []
    for d in ("sample_data", "reports", "."):
        base = Path(d)
        if base.is_dir():
            for p in sorted(base.glob("*.csv")) + sorted(base.glob("*.json")):
                if ".transcript" not in p.name:
                    out.append(str(p))
    # de-dupe preserving order
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


@app.post("/api/upload")
async def upload(req: UploadRequest) -> dict[str, str]:
    """Accept a small CSV/JSON as text (no multipart dependency) and save it
    server-side so the existing path-based run flow can use it."""
    name = Path(req.filename).name
    if not (name.endswith(".csv") or name.endswith(".json")):
        raise HTTPException(400, "only .csv or .json files are accepted")
    if len(req.content) > 8_000_000:
        raise HTTPException(400, "file too large (8 MB limit)")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    dest.write_text(req.content, encoding="utf-8")
    return {"path": str(dest)}


@app.post("/api/runs")
async def start_run(req: RunRequest) -> dict[str, str]:
    question = req.question.strip()
    data = req.data.strip()
    if not question:
        raise HTTPException(400, "question is required")
    if not data:
        raise HTTPException(400, "data path is required")
    if not Path(data).is_file():
        raise HTTPException(400, f"data file not found: {data}")
    run = Run(
        id=uuid.uuid4().hex[:12], question=question, data=data,
        group_col=req.group_col or None, value_col=req.value_col or None,
        paired=req.paired, alpha=req.alpha, rounds=req.rounds or config.MAX_ROUNDS,
        dry_run=req.dry_run,
    )
    RUNS[run.id] = run
    asyncio.create_task(_execute(run))
    return {"id": run.id}


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    return [{"id": r.id, "question": r.question, "status": r.status, "dry_run": r.dry_run,
             "pdf": r.pdf_name, "transcript": r.transcript_name} for r in RUNS.values()]


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run id")

    async def gen():
        idx = 0
        while True:
            while idx < len(run.events):
                yield f"data: {json.dumps(run.events[idx], ensure_ascii=False)}\n\n"
                idx += 1
            if run.status != "running":
                return
            run.changed.clear()
            if idx < len(run.events):
                continue
            await run.changed.wait()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/reports")
async def list_reports() -> list[dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for pdf in sorted(REPORTS_DIR.glob("*.pdf"),
                      key=lambda p: p.stat().st_mtime, reverse=True):
        tr = pdf.with_name(pdf.stem + ".transcript.json")
        items.append({"name": pdf.name,
                      "transcript": tr.name if tr.exists() else None,
                      "modified": pdf.stat().st_mtime})
    return items


@app.get("/api/reports/{name}")
async def get_report(name: str):
    safe = Path(name).name
    path = REPORTS_DIR / safe
    if not path.is_file():
        raise HTTPException(404, "no such report")
    if safe.endswith(".pdf"):
        return FileResponse(path, media_type="application/pdf")
    if safe.endswith(".transcript.json") or safe.endswith(".md"):
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    raise HTTPException(400, "only .pdf, .md and .transcript.json files are served")


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="stat_board.webapp",
                                     description="Web UI for the statistical board.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8643)
    args = parser.parse_args()
    print(f"Statistical Board UI → http://{args.host}:{args.port}")
    if not config.credentials_present():
        print("WARNING: no Anthropic credentials found — live runs will fail. "
              "Put ANTHROPIC_API_KEY=... in a .env file here, export it, or run "
              "`ant auth login`. Dry runs work regardless.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
