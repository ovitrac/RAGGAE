# RAGGAE/cli/demo_app.py
"""
Minimal demo API for RAGGAE

FastAPI-based web application providing RESTful endpoints for document
indexing, semantic search, and NLI-based compliance scoring. Includes
static file serving for the web UI.

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Run:
    uvicorn RAGGAE.cli.demo_app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health
    POST /upload         Multipart file upload (single/multi/zip)
    POST /upload-multi   Multiple file upload
    POST /index          JSON: {"key":"...","index_path":"...","model":"...","e5":true}
    POST /search         JSON: {"index_path":"...","query":"...","k":10}
    POST /quickscore     JSON: {"index_path":"...","requirements":[...],"topk":5}
    POST /quickscore/export  Export results as JSON or CSV
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import re, shutil, contextlib, time, zipfile, io, csv, json, logging

# RAGGAE library
from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex
from RAGGAE.core.nli_ollama import NLIClient, NLIConfig
from RAGGAE.core.nli_claude import ClaudeNLIClient, ClaudeNLIConfig, create_nli_client, load_api_key
from RAGGAE.core.scoring import FitScorer, RequirementVerdict
from RAGGAE.io.textloaders import load_blocks_any, TextBlock
from RAGGAE.io.pdf import extract_blocks as extract_pdf_blocks, to_texts_and_meta
from RAGGAE.io.ingest import ingest_any, get_supported_extensions, is_pandoc_available
from RAGGAE.core.worker import (
    get_queue, get_worker, create_indexing_job, get_job_progress,
    cancel_job, list_jobs, IndexJobConfig, JobProgress, JobStatus
)

# Configure the root logger (prints to console)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Create a named logger for your app
logger = logging.getLogger("raggae")

# accepted file formats for indexing (extended to support office formats)
ALLOWED_EXTS = {
    "pdf", "docx", "doc", "odt", "rtf",          # documents
    "pptx", "ppt", "odp",                         # presentations
    "xlsx", "xls", "ods", "csv",                  # spreadsheets
    "txt", "md", "json", "xml", "html",           # text
    "py", "java", "js", "ts", "c", "cpp", "h",    # code
    "yaml", "yml", "toml", "ini", "cfg",          # config
    "zip"                                          # archives
}

# %% App
app = FastAPI(title="RAGGAE demo", version="0.1.2")

# Allow your LAN / specific origins (tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000", "http://YOUR-LAN-IP:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI (serve frontend from FASTapi)
web_dir = Path(__file__).resolve().parent.parent / "web"
app.mount("/app", StaticFiles(directory=web_dir, html=True), name="web")

# Upload storage (create if missing)
# Tip: The endpoint saves files under RAGGAE/uploads/ and returns the absolute path to reuse in /index.
uploads_dir = Path(__file__).resolve().parent.parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

SAFE_NAME = re.compile(r"[^A-Za-z0-9._\-]+")

# helpers
def _sanitize_name(name: str) -> str:
    # Drop path components and scrub suspicious chars
    name = Path(name).name
    return SAFE_NAME.sub("_", name)

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = web_dir / "index.html"
    return index_path.read_text(encoding="utf-8")

def _new_session_folder() -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    p = uploads_dir / ts
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_upload(dest_dir: Path, file: UploadFile) -> Path:
    safe = _sanitize_name(file.filename)
    dest = dest_dir / safe
    with dest.open("wb") as fout:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk: break
            fout.write(chunk)
    return dest

def _resolve_key_to_files(key: str, exts: List[str]) -> List[Path]:
    """key can be 'YYYYmmdd-HHMMSS' (session) or 'YYYYmmdd-HHMMSS/file.pdf' (single)."""
    base = uploads_dir / key
    out: List[Path] = []
    if base.is_dir():
        for p in base.iterdir():
            if p.suffix.lower().lstrip(".") in exts:
                out.append(p)
    elif base.is_file():
        if base.suffix.lower().lstrip(".") in exts:
            out.append(base)
    return out

def _safe_ext(fname: Optional[str]) -> str:
    """Return file extension (lower, no dot) or '' on any issue."""
    if not fname:
        return ""
    s = str(fname).strip()
    if s in {".", "./"}:
        return ""
    try:
        return Path(s).suffix.lstrip(".").lower()
    except Exception:
        return ""


# upload single/multi
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    sess = _new_session_folder()
    try:
        fname = file.filename.lower()
        if fname.endswith(".zip"):
            buf = await file.read()
            zf = zipfile.ZipFile(io.BytesIO(buf))
            count = 0
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                ext = Path(name).suffix.lstrip(".").lower()
                if ext not in ALLOWED_EXTS:
                    continue
                data = zf.read(name)
                dest = sess / _sanitize_name(Path(name).name)
                with dest.open("wb") as f: f.write(data)
                count += 1
            key = f"{sess.relative_to(uploads_dir)}"
            return {"ok": True, "type":"zip", "key": key, "files": count}
        else:
            ext = Path(fname).suffix.lstrip(".")
            if ext not in ALLOWED_EXTS:
                raise HTTPException(400, f"Only {sorted(ALLOWED_EXTS)} or .zip accepted")
            dest = _save_upload(sess, file)
            rel = dest.relative_to(uploads_dir)
            key = str(rel)
            return {"ok": True, "type": ext, "key": key, "size": dest.stat().st_size}
    except Exception as e:
        with contextlib.suppress(Exception):
            sess.rmdir()
        logger.exception("Upload failed: %s", e)
        raise HTTPException(500, f"Upload failed: {e}")

@app.post("/upload-multi")
async def upload_multi(files: List[UploadFile] = File(...)):
    sess = _new_session_folder()
    saved = []
    try:
        for f in files:
            ext = Path(f.filename).suffix.lstrip(".").lower()
            if ext in ALLOWED_EXTS:
                p = _save_upload(sess, f)
                saved.append(str(p.relative_to(uploads_dir)))
        if not saved:
            raise HTTPException(400, f"No allowed files uploaded; allowed: {sorted(ALLOWED_EXTS)}")
        return {"ok": True, "key": f"{sess.relative_to(uploads_dir)}", "files": saved}
    except Exception as e:
        with contextlib.suppress(Exception):
            for s in saved:
                (uploads_dir / s).unlink(missing_ok=True)
            sess.rmdir()
        logger.exception("Multi upload failed: %s", e)
        raise HTTPException(500, f"Multi upload failed: {e}")

@app.get("/list-uploads")
def list_uploads():
    sessions = []
    for sess in sorted(uploads_dir.iterdir()):
        if not sess.is_dir():
            continue
        files = [str(p.relative_to(uploads_dir))
                 for p in sess.iterdir()
                 if p.suffix.lstrip(".").lower() in ALLOWED_EXTS]
        sessions.append({"session": str(sess.name), "files": files})
    return {"ok": True, "sessions": sessions}

# --------- Schemas ---------
# Update IndexRequest schema to accept key or pdf_path + extensions
class IndexRequest(BaseModel):
    pdf_path: Optional[str] = None
    key: Optional[str] = None
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    min_chars: int = 40
    extensions: List[str] = ["pdf","docx","txt","odt","md"]

class SearchRequest(BaseModel):
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    query: str
    k: int = 10
    context: int = 0  # Number of surrounding blocks to include (0 = no context, like grep -A/-B)

class QuickScoreRequest(BaseModel):
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    requirements: List[str]
    topk: int = 5
    # NLI backend configuration
    nli_backend: str = "ollama"     # "ollama" (default, local) | "claude" (API)
    ollama_model: str = "mistral"   # For ollama backend
    claude_model: str = "claude-sonnet-4-20250514"  # For claude backend
    anthropic_api_key: Optional[str] = None  # Required if nli_backend="claude"
    nli_lang: str = "auto"          # "auto" | "en" | "fr"

class QuickScoreExportRequest(BaseModel):
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    requirements: List[str] = Field(..., min_length=1)
    topk: int = 5
    # NLI backend configuration
    nli_backend: str = "ollama"     # "ollama" (default, local) | "claude" (API)
    ollama_model: str = "mistral"   # For ollama backend
    claude_model: str = "claude-sonnet-4-20250514"  # For claude backend
    anthropic_api_key: Optional[str] = None  # Required if nli_backend="claude"
    nli_lang: str = "auto"          # optional hint
    format: str = "json"            # "json" | "csv"

def _create_nli_client(req):
    """Create NLI client based on request parameters.

    For Claude backend, API key is resolved in order:
    1. anthropic_api_key in request
    2. ANTHROPIC_API_KEY environment variable
    3. Config file (~/.config/raggae/config.json)
    """
    backend = getattr(req, "nli_backend", "ollama").lower()
    lang = getattr(req, "nli_lang", "auto")

    if backend == "claude":
        # API key from request (optional, will fallback to env/config)
        explicit_key = getattr(req, "anthropic_api_key", None)
        resolved_key = load_api_key(explicit_key)

        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. Provide via:\n"
                "  - anthropic_api_key in request body\n"
                "  - ANTHROPIC_API_KEY environment variable\n"
                "  - Config file: ~/.config/raggae/config.json"
            )

        claude_model = getattr(req, "claude_model", "claude-sonnet-4-20250514")
        return ClaudeNLIClient(
            api_key=resolved_key,
            config=ClaudeNLIConfig(model=claude_model, lang=lang, temperature=0.0)
        )
    else:
        # Default: Ollama (local, sovereign)
        ollama_model = getattr(req, "ollama_model", "mistral")
        return NLIClient(NLIConfig(model=ollama_model, temperature=0.0, lang=lang))


def _run_quickscore(req) -> dict:
    """Core quickscore routine used by /quickscore and /quickscore/export."""
    enc = STBiEncoder(req.model, prefix_mode="e5" if req.e5 else "none")
    idx = FaissIndex.load(req.index_path)
    nli = _create_nli_client(req)

    # Check if we're using Claude (supports batch optimization)
    use_batch = isinstance(nli, ClaudeNLIClient)

    verdicts = []
    details = []

    for r in req.requirements:
        qv = enc.embed_query(r).astype("float32")[None, :]
        D, I, recs = idx.search(qv, req.topk)

        chosen = {"label":"No","rationale":"No positive evidence found.",
                  "file":"","ext":"","page":None,"block":None,"snippet":"", "score":0.0}
        evaluated = []

        if use_batch and recs and recs[0]:
            # Claude: batch all clause checks for this requirement
            pairs = [(rec.text, r) for rec in recs[0]]
            results = nli.check_batch(pairs)

            for (score, rec), res in zip(zip(D[0], recs[0]), results):
                m = rec.metadata or {}
                fname = m.get("file","")
                ext = Path(fname).suffix.lstrip(".").lower() if fname else ""
                snippet = rec.text[:200].replace("\n"," ")
                row = {
                    "label": res.label, "rationale": res.rationale,
                    "file": fname, "ext": ext, "page": m.get("page"), "block": m.get("block"),
                    "score": float(score), "snippet": snippet
                }
                evaluated.append(row)
                if res.label == "Yes" and chosen["label"] != "Yes":
                    chosen = row
                elif res.label == "Partial" and chosen["label"] == "No":
                    chosen = row
        else:
            # Ollama: sequential checks with early stopping
            for score, rec in zip(D[0], recs[0]):
                m = rec.metadata or {}
                fname = m.get("file","")
                ext = Path(fname).suffix.lstrip(".").lower() if fname else ""
                snippet = rec.text[:200].replace("\n"," ")
                res = nli.check(rec.text, r)
                row = {
                    "label": res.label, "rationale": res.rationale,
                    "file": fname, "ext": ext, "page": m.get("page"), "block": m.get("block"),
                    "score": float(score), "snippet": snippet
                }
                evaluated.append(row)
                if res.label == "Yes":
                    chosen = row; break
                if res.label == "Partial" and chosen["label"] != "Yes":
                    chosen = row

        verdicts.append(RequirementVerdict(requirement=r, label=chosen["label"], rationale=chosen["rationale"], weight=1.0))
        details.append({
            "requirement": r,
            "verdict": chosen["label"],
            "rationale": chosen["rationale"],
            "evidence": {k: chosen[k] for k in ("file","ext","page","block","snippet","score")},
            "evaluated": evaluated
        })

    scorer = FitScorer()
    s = scorer.fit_score(verdicts)

    # Add backend info to response
    backend_info = {
        "nli_backend": getattr(req, "nli_backend", "ollama"),
        "nli_model": getattr(req, "claude_model", None) if getattr(req, "nli_backend", "ollama") == "claude" else getattr(req, "ollama_model", "mistral")
    }

    return {
        "fit_score": scorer.to_percent(s),
        "verdicts": details,
        "summary": [{"requirement": v.requirement, "label": v.label} for v in verdicts],
        "backend": backend_info
    }

@app.post("/quickscore/export")
def quickscore_export(req: QuickScoreExportRequest):
    try:
        result = _run_quickscore(req)
        fmt = (req.format or "json").lower()

        if fmt == "json":
            buf = io.BytesIO(json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
            return StreamingResponse(buf, media_type="application/json",
                                     headers={"Content-Disposition":"attachment; filename=quickscore.json"})

        elif fmt == "csv":
            # Flatten into rows: one row per evaluated clause per requirement
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["requirement","final_verdict","fit_score",
                             "label","rationale","file","ext","page","block","faiss_score","snippet"])
            fit = result.get("fit_score", 0)
            for item in result.get("verdicts", []):
                reqtxt = item["requirement"]
                final_v = item["verdict"]
                for ev in item.get("evaluated", []):
                    writer.writerow([
                        reqtxt, final_v, fit,
                        ev.get("label",""), ev.get("rationale","").replace("\n"," "),
                        ev.get("file",""), ev.get("ext",""),
                        ev.get("page",""), ev.get("block",""),
                        ev.get("score",""), ev.get("snippet","").replace("\n"," ")
                    ])
            buf = io.BytesIO(out.getvalue().encode("utf-8"))
            return StreamingResponse(buf, media_type="text/csv",
                                     headers={"Content-Disposition":"attachment; filename=quickscore.csv"})
        else:
            raise HTTPException(400, "Unsupported format. Use 'json' or 'csv'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export failed: %s", e)
        raise HTTPException(500, f"Export failed: {e}")



# --------- Routes ---------
@app.get("/health")
def health():
    return {"ok": True, "service": "raggae", "version": app.version}

@app.post("/index")
def index_doc(req: IndexRequest):
    try:
        files: List[Path] = []
        if req.key:
            files = _resolve_key_to_files(req.key, [e.lower() for e in req.extensions])
            if not files:
                raise HTTPException(400, f"No files matched in key={req.key} with {req.extensions}")
        elif req.pdf_path:
            p = Path(req.pdf_path)
            if not p.exists(): raise HTTPException(400, "pdf_path not found")
            files = [p]
        else:
            raise HTTPException(400, "Provide either 'key' (recommended) or 'pdf_path'")

        # Aggregate blocks from all files (pdf + text formats)
        all_texts, all_metas = [], []
        for fp in files:
            ext = fp.suffix.lower().lstrip(".")
            if ext == "pdf":
                blocks = extract_pdf_blocks(fp, min_chars=req.min_chars, keep_headers=True)
                t, m = to_texts_and_meta(blocks)
            else:
                # generic loaders return TextBlock; adapt to meta shape
                gblocks = load_blocks_any(fp, min_chars=req.min_chars)
                t = [b.text for b in gblocks]
                m = [b.as_metadata() for b in gblocks]
            # add file provenance
            for md in m: md["file"] = fp.name
            all_texts.extend(t); all_metas.extend(m)

        if not all_texts:
            raise HTTPException(400, "No text found across selected files")

        from RAGGAE.core.embeddings import STBiEncoder
        from RAGGAE.core.index_faiss import FaissIndex
        enc = STBiEncoder(req.model, prefix_mode="e5" if req.e5 else "none")
        vecs = enc.embed_texts(all_texts).astype("float32")
        idx = FaissIndex(dim=vecs.shape[1])
        idx.add(vecs, all_texts, all_metas)
        idx.save(req.index_path)
        return {"indexed": len(all_texts), "files": [f.name for f in files], "index_path": req.index_path, "encoder": str(enc.info)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Indexing failed: %s", e)
        raise HTTPException(500, f"Indexing failed: {e}")

def _get_context_blocks(idx: FaissIndex, hit_file: str, hit_page: int, hit_block: int, context: int) -> dict:
    """
    Get surrounding blocks for context (like grep -A/-B).

    Returns dict with 'before' and 'after' lists of block texts.
    """
    if context <= 0:
        return {"before": [], "after": []}

    before = []
    after = []

    # Find blocks from same file and page with nearby block numbers
    for rec in idx._records:
        m = rec.metadata
        if m.get("file") != hit_file or m.get("page") != hit_page:
            continue

        block_num = m.get("block")
        if block_num is None:
            continue

        # Check if this block is before the hit
        if hit_block - context <= block_num < hit_block:
            before.append({
                "block": block_num,
                "text": rec.text,
                "distance": hit_block - block_num
            })
        # Check if this block is after the hit
        elif hit_block < block_num <= hit_block + context:
            after.append({
                "block": block_num,
                "text": rec.text,
                "distance": block_num - hit_block
            })

    # Sort by block number
    before.sort(key=lambda x: x["block"])
    after.sort(key=lambda x: x["block"])

    return {
        "before": [b["text"] for b in before],
        "after": [a["text"] for a in after]
    }


@app.post("/search")
def search(req: SearchRequest):
    try:
        enc = STBiEncoder(req.model, prefix_mode="e5" if req.e5 else "none")
        idx = FaissIndex.load(req.index_path)
        qv = enc.embed_query(req.query).astype("float32")[None, :]
        D, I, recs = idx.search(qv, req.k)
        out = []
        for score, rec in zip(D[0], recs[0]):
            m = rec.metadata
            fname = m.get("file") or ""
            ext = Path(fname).suffix.lstrip(".").lower() if fname else ""
            page = m.get("page")
            block = m.get("block")

            hit = {
                "score": float(score),
                "page": page,
                "block": block,
                "file": fname,
                "ext": ext,
                "snippet": rec.text,  # Full text, not truncated
            }

            # Add context if requested
            if req.context > 0 and page is not None and block is not None:
                ctx = _get_context_blocks(idx, fname, page, block, req.context)
                hit["context_before"] = ctx["before"]
                hit["context_after"] = ctx["after"]

            out.append(hit)

        return {"query": req.query, "k": req.k, "context": req.context, "hits": out}
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(500, f"Search failed: {e}")

@app.post("/quickscore")
def quickscore(req: QuickScoreRequest):
    """
    NLI-based compliance scoring.

    Supports two NLI backends:
    - "ollama" (default): Local LLM via Ollama (sovereign, no external API)
    - "claude": Anthropic Claude API (requires anthropic_api_key)

    For Claude backend, batch optimization is automatically enabled.
    """
    if not req.requirements:
        raise HTTPException(400, "requirements[] cannot be empty")
    try:
        # Validate Claude backend requirements
        if req.nli_backend == "claude" and not req.anthropic_api_key:
            raise HTTPException(400, "anthropic_api_key is required when nli_backend='claude'")

        return _run_quickscore(req)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Quickscore failed: %s", e)
        raise HTTPException(status_code=500, detail=f"quickscore crashed: {e.__class__.__name__}: {e}")


# ---------------------------------------------------------------------------
# Async Indexing Endpoints (Job Queue)
# ---------------------------------------------------------------------------

class AsyncIndexRequest(BaseModel):
    """Request schema for async (background) indexing."""
    key: Optional[str] = None          # Upload session key
    files: Optional[List[str]] = None  # Or explicit file paths
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    min_chars: int = 40
    max_workers: int = 4
    prefer_pandoc: bool = True


class JobStatusResponse(BaseModel):
    """Response schema for job status."""
    job_id: str
    status: str
    files_total: int
    files_processed: int
    files_failed: int
    files_skipped: int
    chunks_indexed: int
    progress_percent: float
    current_file: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0


@app.post("/index-async")
async def index_async(req: AsyncIndexRequest):
    """
    Start background indexing job.

    This endpoint returns immediately with a job_id.
    Use /job/{job_id} to poll progress.

    Returns:
        {"ok": true, "job_id": "abc123", "files_queued": N}
    """
    try:
        # Resolve files
        if req.key:
            files = _resolve_key_to_files(req.key, list(ALLOWED_EXTS - {"zip"}))
            file_paths = [str(f) for f in files]
        elif req.files:
            file_paths = req.files
        else:
            raise HTTPException(400, "Either 'key' or 'files' must be provided")

        if not file_paths:
            raise HTTPException(400, "No files found to index")

        # Create job config
        config = IndexJobConfig(
            index_path=req.index_path,
            model=req.model,
            e5=req.e5,
            min_chars=req.min_chars,
            max_workers=req.max_workers,
            prefer_pandoc=req.prefer_pandoc
        )

        # Create and start job
        job_id, worker = create_indexing_job(
            files=file_paths,
            index_path=req.index_path,
            config=config,
            start_immediately=True
        )

        return {
            "ok": True,
            "job_id": job_id,
            "files_queued": len(file_paths),
            "index_path": req.index_path
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Async index failed: %s", e)
        raise HTTPException(500, f"Failed to start indexing job: {e}")


@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """
    Get status of an indexing job.

    Returns:
        Job progress including files processed, chunks indexed, etc.
    """
    progress = get_job_progress(job_id)
    if not progress:
        raise HTTPException(404, f"Job not found: {job_id}")

    return progress.to_dict()


@app.post("/job/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    """
    Cancel a running job.

    Returns:
        {"ok": true, "cancelled": true/false}
    """
    success = cancel_job(job_id)
    return {"ok": True, "job_id": job_id, "cancelled": success}


@app.get("/jobs")
def list_jobs_endpoint(status: Optional[str] = None, limit: int = 50):
    """
    List indexing jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Max jobs to return

    Returns:
        {"ok": true, "jobs": [...]}
    """
    try:
        jobs = list_jobs(status=status, limit=limit)
        return {"ok": True, "jobs": jobs}
    except Exception as e:
        logger.exception("Failed to list jobs: %s", e)
        raise HTTPException(500, str(e))


@app.get("/system-info")
def system_info():
    """
    Get system information for debugging.

    Returns:
        Pandoc availability, supported extensions, CUDA status, etc.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "ok": True,
        "pandoc_available": is_pandoc_available(),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "supported_extensions": get_supported_extensions(),
        "allowed_upload_extensions": sorted(ALLOWED_EXTS),
        "uploads_dir": str(uploads_dir),
        "version": "0.2.0"
    }


# ---------------------------------------------------------------------------
# Enhanced Upload Endpoints (Drop Zone Support)
# ---------------------------------------------------------------------------

@app.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Batch upload endpoint for drop zone.

    Accepts any number of files of any supported type.
    Automatically extracts ZIP archives.

    Returns:
        {"ok": true, "key": "session_id", "files": [...], "stats": {...}}
    """
    sess = _new_session_folder()
    saved = []
    stats = {"total": 0, "accepted": 0, "rejected": 0, "extracted": 0}

    try:
        for f in files:
            stats["total"] += 1
            fname = f.filename.lower()
            ext = Path(fname).suffix.lstrip(".")

            if fname.endswith(".zip"):
                # Extract ZIP
                try:
                    buf = await f.read()
                    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
                        for name in zf.namelist():
                            if name.endswith("/"):
                                continue
                            inner_ext = Path(name).suffix.lstrip(".").lower()
                            if inner_ext in (ALLOWED_EXTS - {"zip"}):
                                data = zf.read(name)
                                dest = sess / _sanitize_name(Path(name).name)
                                # Avoid overwrites
                                counter = 1
                                while dest.exists():
                                    stem = Path(name).stem
                                    dest = sess / f"{stem}_{counter}.{inner_ext}"
                                    counter += 1
                                with dest.open("wb") as out:
                                    out.write(data)
                                saved.append({
                                    "name": dest.name,
                                    "path": str(dest.relative_to(uploads_dir)),
                                    "size": len(data),
                                    "ext": inner_ext,
                                    "source": f.filename
                                })
                                stats["extracted"] += 1
                except Exception as e:
                    logger.warning(f"Failed to extract ZIP {f.filename}: {e}")
                    stats["rejected"] += 1

            elif ext in (ALLOWED_EXTS - {"zip"}):
                # Regular file
                dest = _save_upload(sess, f)
                saved.append({
                    "name": dest.name,
                    "path": str(dest.relative_to(uploads_dir)),
                    "size": dest.stat().st_size,
                    "ext": ext,
                    "source": f.filename
                })
                stats["accepted"] += 1

            else:
                stats["rejected"] += 1
                logger.info(f"Rejected unsupported file: {f.filename}")

        if not saved:
            # Clean up empty session
            with contextlib.suppress(Exception):
                sess.rmdir()
            raise HTTPException(400, f"No supported files uploaded. Allowed: {sorted(ALLOWED_EXTS)}")

        return {
            "ok": True,
            "key": str(sess.relative_to(uploads_dir)),
            "files": saved,
            "stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        with contextlib.suppress(Exception):
            shutil.rmtree(sess)
        logger.exception("Batch upload failed: %s", e)
        raise HTTPException(500, f"Batch upload failed: {e}")
