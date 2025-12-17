# RAGGAE CLI Usage Guide

Complete command-line interface reference for document indexing, semantic search, and NLI-based compliance scoring.

## Quick Start

```bash
# 1. Index a document
python -m RAGGAE.cli.index_doc --input tender.pdf --out tender --e5

# 2. Search the index
python -m RAGGAE.cli.search --index tender --e5 --query "ISO 27001"

# 3. Run compliance scoring
python -m RAGGAE.cli.quickscore --index tender --e5 \
    --req "Provider must be ISO 27001 certified" \
    --req "Platform uses MLflow for MLOps"
```

---

## 1. Document Indexing (`index_doc`)

Build a FAISS vector index from documents.

### Basic Usage

```bash
# Single PDF
python -m RAGGAE.cli.index_doc --input document.pdf --out my_index --e5

# Multiple files
python -m RAGGAE.cli.index_doc \
    --input doc1.pdf \
    --input doc2.docx \
    --input notes.txt \
    --out combined_index --e5

# Entire directory
python -m RAGGAE.cli.index_doc --input ./documents/ --out corpus --e5
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | Required | Input file or directory (repeat for multiple) |
| `--out`, `-o` | Required | Output index prefix (.faiss + .jsonl created) |
| `--model` | `intfloat/multilingual-e5-small` | Embedding model |
| `--e5` | false | Use E5-style prefixes |
| `--min-chars` | 40 | Minimum characters per chunk |
| `--no-recursive` | false | Don't recurse into subdirectories |
| `--format` | text | Output format: `text` or `json` |
| `--verbose`, `-v` | false | Show detailed progress |

### Supported Formats

- **Documents**: PDF, DOCX, PPTX, XLSX, ODT, ODP, ODS, RTF
- **Text**: TXT, MD, CSV, JSON, XML, HTML
- **Code**: PY, JAVA, JS, TS, C, CPP, H, GO, RS

### JSON Output (for automation)

```bash
python -m RAGGAE.cli.index_doc --input docs/ --out index --e5 --format json

# Output:
{
  "index": "index",
  "model": "intfloat/multilingual-e5-small",
  "e5": true,
  "files_processed": 5,
  "files_failed": 0,
  "total_chunks": 847,
  "dimension": 384
}
```

---

## 2. Semantic Search (`search`)

Search indexed documents using natural language queries.

### Basic Usage

```bash
# Simple search
python -m RAGGAE.cli.search --index tender --e5 --query "ISO 27001 certification"

# More results
python -m RAGGAE.cli.search --index tender --e5 --k 20 --query "Kubernetes deployment"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | Required | Path to FAISS index (prefix) |
| `--query` | Required | Search query |
| `--model` | `intfloat/multilingual-e5-small` | Embedding model |
| `--e5` | false | Use E5-style prefixes |
| `--k` | 10 | Number of results |
| `--format` | text | Output format: `text` or `json` |
| `--verbose`, `-v` | false | Show full text instead of snippets |

### JSON Output (for automation)

```bash
python -m RAGGAE.cli.search --index tender --e5 --format json --query "MLflow"

# Output:
{
  "query": "MLflow",
  "index": "tender",
  "count": 10,
  "results": [
    {
      "rank": 1,
      "score": 0.8542,
      "text": "Our MLOps platform includes MLflow for...",
      "file": "tender.pdf",
      "page": 12,
      "block": 3
    },
    ...
  ]
}
```

---

## 3. Compliance Scoring (`quickscore`)

NLI-based requirement satisfaction checking using local (Ollama) or cloud (Claude) LLMs.

### Basic Usage (Ollama - Local, Sovereign)

```bash
python -m RAGGAE.cli.quickscore --index tender --e5 \
    --req "Provider must be ISO 27001 certified" \
    --req "Platform uses MLflow for MLOps" \
    --req "Deployments on Kubernetes with GitOps"
```

### Claude Backend (for users without GPU)

```bash
# With environment variable
export ANTHROPIC_API_KEY="sk-ant-..."
python -m RAGGAE.cli.quickscore --index tender --e5 \
    --backend claude \
    --req "Provider must be ISO 27001 certified"

# With explicit key
python -m RAGGAE.cli.quickscore --index tender --e5 \
    --backend claude \
    --api-key "sk-ant-..." \
    --req "Provider must be ISO 27001 certified"

# With config file (~/.config/raggae/config.json)
python -m RAGGAE.cli.quickscore --index tender --e5 \
    --backend claude \
    --req "Provider must be ISO 27001 certified"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | Required | Path to FAISS index (prefix) |
| `--req` | Required | Requirement string (repeat for multiple) |
| `--model` | `intfloat/multilingual-e5-small` | Embedding model |
| `--e5` | false | Use E5-style prefixes |
| `--backend` | `ollama` | NLI backend: `ollama` or `claude` |
| `--ollama-model` | `mistral` | Ollama model |
| `--claude-model` | `claude-sonnet-4-20250514` | Claude model |
| `--api-key` | None | Anthropic API key (optional) |
| `--nli-lang` | `auto` | Language hint: `auto`, `en`, `fr` |
| `--topk` | 5 | Candidate clauses per requirement |
| `--format` | text | Output format: `text`, `json`, `csv` |
| `--verbose`, `-v` | false | Show rationales and evidence |

### Claude Models

| Model | Description |
|-------|-------------|
| `claude-sonnet-4-20250514` | Recommended (fast + capable) |
| `claude-opus-4-20250514` | Highest quality |
| `claude-haiku-3-5-20241022` | Fastest, lowest cost |

### JSON Output (for automation)

```bash
python -m RAGGAE.cli.quickscore --index tender --e5 --format json \
    --req "ISO 27001" --req "MLflow"

# Output:
{
  "score": 75,
  "backend": "ollama",
  "model": "mistral",
  "requirements": [
    {
      "requirement": "ISO 27001",
      "verdict": "Yes",
      "rationale": "Provider certified by Bureau Veritas"
    },
    {
      "requirement": "MLflow",
      "verdict": "Partial",
      "rationale": "MLflow mentioned but details unclear"
    }
  ],
  "summary": {
    "total": 2,
    "yes": 1,
    "partial": 1,
    "no": 0
  }
}
```

### CSV Output (for spreadsheets)

```bash
python -m RAGGAE.cli.quickscore --index tender --e5 --format csv \
    --req "ISO 27001" --req "MLflow" > results.csv
```

---

## 4. Automation Examples

### Bash Pipeline

```bash
#!/bin/bash
# Full pipeline: index → search → score

INDEX_NAME="tender_$(date +%Y%m%d)"
INPUT_DIR="./tender_documents"

# Step 1: Index
python -m RAGGAE.cli.index_doc \
    --input "$INPUT_DIR" \
    --out "$INDEX_NAME" \
    --e5 \
    --format json > index_result.json

# Step 2: Search
python -m RAGGAE.cli.search \
    --index "$INDEX_NAME" \
    --e5 \
    --format json \
    --query "security certifications" > search_result.json

# Step 3: Score (read requirements from file)
REQUIREMENTS=(
    "Provider must be ISO 27001 certified"
    "Platform uses MLflow for MLOps"
    "Data stored in EU region"
)

REQ_ARGS=""
for req in "${REQUIREMENTS[@]}"; do
    REQ_ARGS="$REQ_ARGS --req \"$req\""
done

eval python -m RAGGAE.cli.quickscore \
    --index "$INDEX_NAME" \
    --e5 \
    --backend ollama \
    --format json \
    $REQ_ARGS > score_result.json

echo "Pipeline complete. Score: $(jq .score score_result.json)"
```

### Python Automation

```python
import subprocess
import json

def run_index(input_path, index_name):
    """Index documents."""
    result = subprocess.run([
        "python", "-m", "RAGGAE.cli.index_doc",
        "--input", input_path,
        "--out", index_name,
        "--e5",
        "--format", "json"
    ], capture_output=True, text=True)
    return json.loads(result.stdout)

def run_search(index_name, query, k=10):
    """Search index."""
    result = subprocess.run([
        "python", "-m", "RAGGAE.cli.search",
        "--index", index_name,
        "--e5",
        "--format", "json",
        "--k", str(k),
        "--query", query
    ], capture_output=True, text=True)
    return json.loads(result.stdout)

def run_quickscore(index_name, requirements, backend="ollama"):
    """Run compliance scoring."""
    cmd = [
        "python", "-m", "RAGGAE.cli.quickscore",
        "--index", index_name,
        "--e5",
        "--backend", backend,
        "--format", "json"
    ]
    for req in requirements:
        cmd.extend(["--req", req])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

# Usage
index_result = run_index("./documents", "my_index")
print(f"Indexed {index_result['total_chunks']} chunks")

search_result = run_search("my_index", "ISO 27001")
print(f"Found {search_result['count']} results")

score_result = run_quickscore("my_index", [
    "Provider must be ISO 27001 certified",
    "Platform uses MLflow for MLOps"
])
print(f"Compliance score: {score_result['score']}/100")
```

---

## 5. API Key Configuration

### Option 1: Environment Variable

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option 2: Config File

```bash
# Create config directory
mkdir -p ~/.config/raggae

# Create config file
echo '{"anthropic_api_key": "sk-ant-..."}' > ~/.config/raggae/config.json

# Secure permissions
chmod 600 ~/.config/raggae/config.json
```

### Option 3: Python Helper

```python
from RAGGAE.core.nli_claude import save_api_key, load_api_key

# Save
save_api_key("sk-ant-...")

# Verify
key = load_api_key()
print(f"Key configured: {key[:12]}...")
```

---

## 6. Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (file not found, index load failed, etc.) |

---

## 7. Troubleshooting

### "RAGGAE not found"
```bash
pip install -e .  # Install in editable mode
```

### "Ollama not running"
```bash
ollama serve  # Start Ollama
ollama pull mistral  # Download model
```

### "Claude API key not found"
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Or create ~/.config/raggae/config.json
```

### "Index not found"
```bash
# Check index files exist
ls -la my_index.faiss my_index.jsonl
```
