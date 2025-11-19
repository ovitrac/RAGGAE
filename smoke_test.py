# -*- coding: utf-8 -*-
"""
Smoke test for RAGGAE (use )

Adservio | 2025-10-27
"""

#%% 0a) Environment check (GPU, versions)
import torch, sys, platform, time
print("Python:", platform.python_version(), "| Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| Torch CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| Compute:", torch.cuda.get_device_capability(0))

#%% 1) Imports
from sentence_transformers import SentenceTransformer
import numpy as np, faiss
from rank_bm25 import BM25Okapi

# Optional PDF parsing
PDF_PATH = ""  # set to a local tender PDF path, e.g., "/home/olivi/Documents/tender.pdf"
try:
    import fitz  # PyMuPDF
except Exception as e:
    fitz = None
    print("PyMuPDF not available:", e)

#%% 2) Tiny corpus + optional PDF chunks
def parse_pdf_blocks(path, min_chars=40, max_blocks=300):
    """Return list[{'text','page','block'}] from a PDF, keeping simple text blocks."""
    out = []
    doc = fitz.open(path)
    for pno in range(len(doc)):
        page = doc[pno]
        for bi, blk in enumerate(page.get_text("blocks")):
            # blk: (x0, y0, x1, y1, text, block_no, block_type)
            txt = (blk[4] or "").strip()
            if len(txt) >= min_chars:
                out.append({"text": txt, "page": pno+1, "block": bi})
                if len(out) >= max_blocks:
                    return out
    return out

seed_chunks = [
    {"text": "Adservio propose une offre MLOps fondée sur MLflow et Kubernetes (K8s).", "page": 0, "block": 0},
    {"text": "Exigence ISO 27001 et hébergement des données en Union Européenne.", "page": 0, "block": 1},
    {"text": "DevOps CI/CD avec GitLab, ArgoCD, Helm et GitOps pour déploiement cloud.", "page": 0, "block": 2},
    {"text": "SLA attendu 99.9%, RPO 15 minutes, RTO 1 heure. Support 24/7 requis.", "page": 0, "block": 3},
]

if PDF_PATH and fitz:
    try:
        pdf_chunks = parse_pdf_blocks(PDF_PATH)
        print(f"Parsed {len(pdf_chunks)} blocks from PDF")
        chunks = pdf_chunks or seed_chunks
    except Exception as e:
        print("PDF parse failed, using seed chunks:", e)
        chunks = seed_chunks
else:
    if PDF_PATH and not fitz:
        print("PyMuPDF missing; set PDF_PATH='' or install it in the env.")
    chunks = seed_chunks

texts = [c["text"] for c in chunks]
print(f"Corpus size: {len(texts)}")

#%% 3) Load embedding model (GPU if available)
MODEL = "intfloat/multilingual-e5-small"  # FR/EN good starter
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)
print("Embedding model loaded on:", device)

#%% 4) Build embeddings (timed)
t0 = time.time()
with torch.inference_mode():
    passages = [f"passage: {t}" for t in texts]  # E5-style prefix
    E = model.encode(passages, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
print("Emb shape:", E.shape, "| secs:", round(time.time()-t0, 2))

#%% 5) FAISS index (inner product / cosine with normalized vecs)
index = faiss.IndexFlatIP(E.shape[1])
index.add(E.astype("float32"))
print("FAISS indexed:", index.ntotal, "vectors")

#%% 6) BM25 on same corpus
bm25 = BM25Okapi([t.split() for t in texts])

def _minmax(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.ptp() + 1e-9)

#%% 7) Hybrid search
def hybrid_search(query, k_dense=100, k=10, alpha=0.6):
    # dense
    qv = model.encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = index.search(qv, min(k_dense, len(texts)))
    dense_scores = {int(i): float(s) for i, s in zip(I[0], D[0])}
    # bm25
    bm = bm25.get_scores(query.split())
    bm = _minmax(bm)  # normalize to [0,1]
    # fuse
    fused = []
    for i, ds in dense_scores.items():
        fs = alpha*ds + (1-alpha)*float(bm[i])
        fused.append((i, fs))
    fused.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i, s in fused[:k]:
        out.append(chunks[i] | {"score": round(s, 4)})
    return out

#%% 8) Run a query
query = "Plateforme MLOps avec MLflow sur Kubernetes, exigences ISO 27001 et GitOps"
hits = hybrid_search(query, k=5)
print("\nQuery:", query)
for h in hits:
    loc = f"(p.{h['page']}, block {h['block']})" if h.get("page") else ""
    print(f"- score={h['score']:.4f} {loc} :: {h['text'][:110]}…")

#%% 9) Optional: quick provenance pretty-print
def show_hit(h, max_len=400):
    print(f"\n[score={h['score']}] page={h.get('page','?')} block={h.get('block','?')}")
    print(h["text"][:max_len] + ("…" if len(h["text"])>max_len else ""))

if hits:
    show_hit(hits[0])

#%% 10) (Optional) NLI/compliance check with Ollama (requires daemon running)
# Uncomment to test. Example: does the clause satisfy an ISO 27001 requirement?
import ollama, json
clause = hits[0]["text"] if hits else "Le prestataire dispose d’une certification ISO 27001."
prompt = f'''You are a compliance checker.
Clause: "{clause}"
Requirement: "Provider must be ISO 27001 certified."
Answer JSON with keys: label in ["Yes","No","Partial"], rationale (short).
'''
res = ollama.chat(model="mistral", messages=[{"role":"user","content":prompt}])
print("\nNLI result:", res["message"]["content"])

# %% 10b) More sophisticated RAG: Hardened NLI helper (deterministic + JSON-safe + FR/EN)
import ollama, json, re

NLI_SYS = (
  "You are a strict compliance checker. "
  "Return ONLY compact JSON with keys: label, rationale. "
  "label ∈ ['Yes','No','Partial']."
)

def parse_json_loose(s: str):
    # strip code fences and grab the first {...}
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I|re.M)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def nli_check(clause: str, requirement: str, lang="auto"):
    prompt = (
      f"Language: {lang}. "
      f'Clause: "{clause}"\n'
      f'Requirement: "{requirement}"\n'
      'Respond as JSON: {"label":"Yes|No|Partial","rationale":"..."}'
    )
    r = ollama.chat(
        model="mistral",
        options={"temperature": 0, "num_ctx": 4096},
        messages=[{"role":"system","content":NLI_SYS},
                  {"role":"user","content":prompt}]
    )
    out = parse_json_loose(r["message"]["content"]) or {"label":"No","rationale":"Invalid or non-JSON output"}
    # normalize label
    lbl = out.get("label","").strip().title()
    if lbl not in {"Yes","No","Partial"}: lbl = "No"
    out["label"] = lbl
    return out

# quick test (use your best clause from hits)
clause = hits[0]["text"] if hits else "Le prestataire dispose d’une certification ISO 27001."
print(nli_check(clause, "Provider must be ISO 27001 certified"))

# %% 10c) Batch matrix (requirements x top-k clauses)
import pandas as pd

REQUIREMENTS = [
  "Provider must be ISO 27001 certified",
  "Platform uses MLflow for MLOps",
  "Deployments on Kubernetes with GitOps",
  "Data hosted in the European Union"
]

def requirement_matrix(hits, requirements=REQUIREMENTS, topk=5):
    rows = []
    for req in requirements:
        for i, h in enumerate(hits[:topk]):
            res = nli_check(h["text"], req)
            rows.append({
                "requirement": req,
                "hit_rank": i+1,
                "label": res["label"],
                "rationale": res["rationale"],
                "page": h.get("page"),
                "block": h.get("block"),
                "snippet": h["text"][:160].replace("\n"," ")
            })
    df = pd.DataFrame(rows)
    # simple per-requirement verdict: first Yes > Partial > No
    order = {"Yes":2, "Partial":1, "No":0}
    verdict = (df.assign(score=df["label"].map(order))
                 .groupby("requirement")["score"].max()
                 .map({2:"Yes",1:"Partial",0:"No"}))
    return df, verdict

df_checks, verdict = requirement_matrix(hits, REQUIREMENTS, topk=5)
print("\nOverall verdict per requirement:\n", verdict)
print("\nSample rows:\n", df_checks.head(6))


# %% 10d) Fit score from NLI labels (0..100)
label_w = {"Yes": 1.0, "Partial": 0.5, "No": 0.0}
fit_score = round(100 * verdict.map({"Yes":1.0,"Partial":0.5,"No":0.0}).mean(), 1)
print(f"\nTender fit score (NLI): {fit_score}/100")
