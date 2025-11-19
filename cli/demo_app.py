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
from RAGGAE.core.scoring import FitScorer, RequirementVerdict
from RAGGAE.io.textloaders import load_blocks_any, TextBlock
from RAGGAE.io.pdf import extract_blocks as extract_pdf_blocks, to_texts_and_meta

# Configure the root logger (prints to console)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Create a named logger for your app
logger = logging.getLogger("raggae")

# accepted file formats for indexing
ALLOWED_EXTS = {"pdf", "docx", "txt", "odt", "md"}

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

class QuickScoreRequest(BaseModel):
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    requirements: List[str]
    topk: int = 5
    ollama_model: str = "mistral"
    nli_lang: str = "auto"  # "auto" | "en" | "fr"

class QuickScoreExportRequest(BaseModel):
    index_path: str
    model: str = "intfloat/multilingual-e5-small"
    e5: bool = True
    requirements: List[str] = Field(..., min_items=1)
    topk: int = 5
    ollama_model: str = "mistral"
    nli_lang: str = "auto"        # optional hint
    format: str = "json"          # "json" | "csv"

def _run_quickscore(req) -> dict:
    """Core quickscore routine used by /quickscore and /quickscore/export."""
    enc = STBiEncoder(req.model, prefix_mode="e5" if req.e5 else "none")
    idx = FaissIndex.load(req.index_path)
    nli = NLIClient(NLIConfig(model=req.ollama_model, temperature=0.0, lang=getattr(req, "nli_lang", "auto")))

    verdicts = []
    details = []

    for r in req.requirements:
        qv = enc.embed_query(r).astype("float32")[None, :]
        D, I, recs = idx.search(qv, req.topk)

        chosen = {"label":"No","rationale":"No positive evidence found.",
                  "file":"","ext":"","page":None,"block":None,"snippet":"", "score":0.0}
        evaluated = []

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
    return {
        "fit_score": scorer.to_percent(s),
        "verdicts": details,
        "summary": [{"requirement": v.requirement, "label": v.label} for v in verdicts]
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
            out.append({
                  "score": float(score),
                  "page": m.get("page"),
                  "block": m.get("block"),
                  "file": fname,
                  "ext": ext,
                  "snippet": rec.text[:200].replace("\n"," ")
            })
        return {"query": req.query, "k": req.k, "hits": out}
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(500, f"Search failed: {e}")

@app.post("/quickscore")
def quickscore(req: QuickScoreRequest):
    if not req.requirements:
        raise HTTPException(400, "requirements[] cannot be empty")
    try:
        enc = STBiEncoder(req.model, prefix_mode="e5" if req.e5 else "none")
        idx = FaissIndex.load(req.index_path)

        # Use getattr (Pydantic model is not a dict)
        lang_hint = getattr(req, "nli_lang", "auto")
        nli = NLIClient(NLIConfig(model=req.ollama_model, temperature=0.0, lang=lang_hint))

        verdicts = []
        details = []  # enriched per requirement (chosen evidence + evaluated list)

        for r in req.requirements:
            qv = enc.embed_query(r).astype("float32")[None, :]
            D, I, recs = idx.search(qv, req.topk)

            # Guard empty retrieval
            if not recs or not recs[0]:
                verdicts.append(RequirementVerdict(requirement=r, label="No",
                                                   rationale="No candidate clauses retrieved.", weight=1.0))
                details.append({
                    "requirement": r,
                    "verdict": "No",
                    "rationale": "No candidate clauses retrieved.",
                    "evidence": {"file":"", "ext":"", "page":None, "block":None, "snippet":"", "score":0.0},
                    "evaluated": []
                })
                continue

            chosen = {"label":"No","rationale":"No positive evidence found.",
                      "file":"","ext":"","page":None,"block":None,"snippet":"", "score":0.0}
            evaluated = []

            for score, rec in zip(D[0], recs[0]):
                m = rec.metadata or {}
                fname = m.get("file", "")
                ext = _safe_ext(fname)
                snippet = rec.text[:200].replace("\n"," ")
                res = nli.check(rec.text, r)

                row = {
                    "label": res.label,
                    "rationale": res.rationale,
                    "file": fname or "",
                    "ext": ext,
                    "page": m.get("page"),
                    "block": m.get("block"),
                    "score": float(score),
                    "snippet": snippet
                }
                evaluated.append(row)

                # pick best: Yes > Partial > No
                if res.label == "Yes":
                    chosen = row
                    break
                if res.label == "Partial" and chosen["label"] != "Yes":
                    chosen = row

            verdicts.append(RequirementVerdict(requirement=r, label=chosen["label"],
                                               rationale=chosen["rationale"], weight=1.0))
            details.append({
                "requirement": r,
                "verdict": chosen["label"],
                "rationale": chosen["rationale"],
                "evidence": {k: chosen[k] for k in ("file","ext","page","block","snippet","score")},
                "evaluated": evaluated
            })

        scorer = FitScorer()
        s = scorer.fit_score(verdicts)
        return {
            "fit_score": scorer.to_percent(s),
            "verdicts": details,
            "summary": [{"requirement": v.requirement, "label": v.label} for v in verdicts]
        }
    except Exception as e:
        logger.exception("Quickscore failed: %s", e)
        raise HTTPException(status_code=500, detail=f"quickscore crashed: {e.__class__.__name__}: {e}")
