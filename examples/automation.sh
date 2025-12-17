#!/bin/bash
# =============================================================================
# RAGGAE CLI Automation Examples
#
# Complete workflow examples for document indexing, semantic search, and
# NLI-based compliance scoring using command-line tools.
#
# Author: Dr. Olivier Vitrac, PhD, HDR
# Organization: Adservio
# Date: December 17, 2025
# License: MIT
#
# Usage:
#   chmod +x examples/automation.sh
#   ./examples/automation.sh
#
# Requirements:
#   - Python 3.12+ with RAGGAE installed
#   - Ollama running locally (for NLI, unless using Claude)
#   - Optional: ANTHROPIC_API_KEY for Claude backend
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
INDEX_DIR="${PROJECT_ROOT}/indexes"
OUTPUT_DIR="${PROJECT_ROOT}/output"

# Default settings
MODEL="intfloat/multilingual-e5-small"
OLLAMA_MODEL="mistral"
TOPK=5

# Create directories
mkdir -p "$INDEX_DIR" "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.12+"
        exit 1
    fi

    # Check RAGGAE is importable
    if ! python -c "import RAGGAE" 2>/dev/null; then
        log_error "RAGGAE not installed. Run: pip install -e ."
        exit 1
    fi

    # Check Ollama (optional)
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama is running"
        OLLAMA_AVAILABLE=true
    else
        log_warn "Ollama not running. NLI with Ollama backend will fail."
        OLLAMA_AVAILABLE=false
    fi

    # Check Claude API key (optional)
    if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
        log_success "ANTHROPIC_API_KEY is set"
        CLAUDE_AVAILABLE=true
    elif [[ -f "$HOME/.config/raggae/config.json" ]]; then
        log_success "Claude config file found"
        CLAUDE_AVAILABLE=true
    else
        log_warn "No Claude API key found. Claude backend will not be available."
        CLAUDE_AVAILABLE=false
    fi

    log_success "Prerequisites check complete"
}

# -----------------------------------------------------------------------------
# 1. Document Indexing
# -----------------------------------------------------------------------------

index_single_file() {
    local input_file="$1"
    local index_name="$2"

    log_info "Indexing single file: $input_file"

    python -m RAGGAE.cli.index_doc \
        --input "$input_file" \
        --out "${INDEX_DIR}/${index_name}" \
        --model "$MODEL" \
        --e5 \
        --format json | tee "${OUTPUT_DIR}/${index_name}_index.json"

    log_success "Index created: ${INDEX_DIR}/${index_name}"
}

index_directory() {
    local input_dir="$1"
    local index_name="$2"

    log_info "Indexing directory: $input_dir"

    python -m RAGGAE.cli.index_doc \
        --input "$input_dir" \
        --out "${INDEX_DIR}/${index_name}" \
        --model "$MODEL" \
        --e5 \
        --verbose \
        --format json | tee "${OUTPUT_DIR}/${index_name}_index.json"

    log_success "Index created: ${INDEX_DIR}/${index_name}"
}

index_multiple_files() {
    local index_name="$1"
    shift
    local files=("$@")

    log_info "Indexing multiple files..."

    local input_args=""
    for f in "${files[@]}"; do
        input_args="$input_args --input $f"
    done

    python -m RAGGAE.cli.index_doc \
        $input_args \
        --out "${INDEX_DIR}/${index_name}" \
        --model "$MODEL" \
        --e5 \
        --format json | tee "${OUTPUT_DIR}/${index_name}_index.json"

    log_success "Index created: ${INDEX_DIR}/${index_name}"
}

# -----------------------------------------------------------------------------
# 2. Semantic Search
# -----------------------------------------------------------------------------

search_index() {
    local index_name="$1"
    local query="$2"
    local k="${3:-10}"

    log_info "Searching index: $index_name"
    log_info "Query: $query"

    python -m RAGGAE.cli.search \
        --index "${INDEX_DIR}/${index_name}" \
        --query "$query" \
        --model "$MODEL" \
        --e5 \
        --k "$k" \
        --format json | tee "${OUTPUT_DIR}/search_results.json"

    log_success "Search complete"
}

search_index_text() {
    local index_name="$1"
    local query="$2"

    python -m RAGGAE.cli.search \
        --index "${INDEX_DIR}/${index_name}" \
        --query "$query" \
        --model "$MODEL" \
        --e5 \
        --k 5
}

# -----------------------------------------------------------------------------
# 3. NLI-Based Compliance Scoring
# -----------------------------------------------------------------------------

quickscore_ollama() {
    local index_name="$1"
    shift
    local requirements=("$@")

    if [[ "$OLLAMA_AVAILABLE" != "true" ]]; then
        log_error "Ollama not available. Cannot run quickscore with Ollama backend."
        return 1
    fi

    log_info "Running quickscore with Ollama backend..."

    local req_args=""
    for r in "${requirements[@]}"; do
        req_args="$req_args --req \"$r\""
    done

    eval python -m RAGGAE.cli.quickscore \
        --index "${INDEX_DIR}/${index_name}" \
        --model "$MODEL" \
        --e5 \
        --backend ollama \
        --ollama-model "$OLLAMA_MODEL" \
        --topk "$TOPK" \
        $req_args \
        --format json | tee "${OUTPUT_DIR}/quickscore_ollama.json"

    log_success "Quickscore (Ollama) complete"
}

quickscore_claude() {
    local index_name="$1"
    local claude_model="${2:-claude-sonnet-4-20250514}"
    shift 2
    local requirements=("$@")

    if [[ "$CLAUDE_AVAILABLE" != "true" ]]; then
        log_error "Claude API key not available. Cannot run quickscore with Claude backend."
        return 1
    fi

    log_info "Running quickscore with Claude backend ($claude_model)..."

    local req_args=""
    for r in "${requirements[@]}"; do
        req_args="$req_args --req \"$r\""
    done

    eval python -m RAGGAE.cli.quickscore \
        --index "${INDEX_DIR}/${index_name}" \
        --model "$MODEL" \
        --e5 \
        --backend claude \
        --claude-model "$claude_model" \
        --topk "$TOPK" \
        $req_args \
        --format json | tee "${OUTPUT_DIR}/quickscore_claude.json"

    log_success "Quickscore (Claude) complete"
}

# -----------------------------------------------------------------------------
# 4. Full Pipeline Example
# -----------------------------------------------------------------------------

run_full_pipeline() {
    local input_path="$1"
    local index_name="$2"
    local query="$3"
    shift 3
    local requirements=("$@")

    log_info "=========================================="
    log_info "Running full RAGGAE pipeline"
    log_info "=========================================="

    # Step 1: Index
    log_info "Step 1/3: Indexing documents..."
    if [[ -d "$input_path" ]]; then
        index_directory "$input_path" "$index_name"
    else
        index_single_file "$input_path" "$index_name"
    fi

    # Step 2: Search
    log_info "Step 2/3: Semantic search..."
    search_index "$index_name" "$query" 10

    # Step 3: Quickscore
    log_info "Step 3/3: NLI compliance scoring..."
    if [[ "$OLLAMA_AVAILABLE" == "true" ]]; then
        quickscore_ollama "$index_name" "${requirements[@]}"
    elif [[ "$CLAUDE_AVAILABLE" == "true" ]]; then
        quickscore_claude "$index_name" "claude-haiku-3-5-20241022" "${requirements[@]}"
    else
        log_warn "No NLI backend available. Skipping quickscore."
    fi

    log_success "=========================================="
    log_success "Pipeline complete!"
    log_success "Results in: $OUTPUT_DIR"
    log_success "=========================================="
}

# -----------------------------------------------------------------------------
# 5. Batch Processing Example
# -----------------------------------------------------------------------------

batch_process_requirements() {
    local index_name="$1"
    local requirements_file="$2"
    local backend="${3:-ollama}"

    log_info "Batch processing requirements from: $requirements_file"

    # Read requirements from file (one per line)
    local req_args=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        req_args="$req_args --req \"$line\""
    done < "$requirements_file"

    if [[ "$backend" == "claude" ]]; then
        eval python -m RAGGAE.cli.quickscore \
            --index "${INDEX_DIR}/${index_name}" \
            --model "$MODEL" \
            --e5 \
            --backend claude \
            --topk "$TOPK" \
            $req_args \
            --format json
    else
        eval python -m RAGGAE.cli.quickscore \
            --index "${INDEX_DIR}/${index_name}" \
            --model "$MODEL" \
            --e5 \
            --backend ollama \
            --ollama-model "$OLLAMA_MODEL" \
            --topk "$TOPK" \
            $req_args \
            --format json
    fi
}

# -----------------------------------------------------------------------------
# 6. Export Results
# -----------------------------------------------------------------------------

export_to_csv() {
    local index_name="$1"
    shift
    local requirements=("$@")

    log_info "Exporting quickscore results to CSV..."

    local req_args=""
    for r in "${requirements[@]}"; do
        req_args="$req_args --req \"$r\""
    done

    eval python -m RAGGAE.cli.quickscore \
        --index "${INDEX_DIR}/${index_name}" \
        --model "$MODEL" \
        --e5 \
        --backend ollama \
        --topk "$TOPK" \
        $req_args \
        --format csv > "${OUTPUT_DIR}/results.csv"

    log_success "Results exported to: ${OUTPUT_DIR}/results.csv"
}

# -----------------------------------------------------------------------------
# 7. API Key Management
# -----------------------------------------------------------------------------

setup_claude_key() {
    log_info "Setting up Claude API key..."

    read -sp "Enter your Anthropic API key (sk-ant-...): " api_key
    echo

    python -c "
from RAGGAE.core.nli_claude import save_api_key
path = save_api_key('$api_key')
print(f'API key saved to: {path}')
"

    log_success "Claude API key configured"
}

check_claude_key() {
    python -c "
from RAGGAE.core.nli_claude import load_api_key
key = load_api_key()
if key:
    print(f'API key found: {key[:12]}...{key[-4:]}')
else:
    print('No API key found')
"
}

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

show_help() {
    cat << EOF
RAGGAE CLI Automation Examples

Usage: $0 <command> [options]

Commands:
  check           Check prerequisites (Python, RAGGAE, Ollama, Claude)
  index           Index documents (file or directory)
  search          Semantic search
  quickscore      NLI compliance scoring
  pipeline        Run full pipeline (index → search → quickscore)
  setup-claude    Configure Claude API key
  demo            Run demo with sample data

Examples:
  # Check prerequisites
  $0 check

  # Index a PDF
  $0 index tender.pdf tender_index

  # Search an index
  $0 search tender_index "ISO 27001 certification"

  # Run quickscore with Ollama
  $0 quickscore tender_index "Provider must be ISO 27001 certified"

  # Run full pipeline
  $0 pipeline ./documents tender_corpus "MLflow" "ISO 27001" "Kubernetes"

  # Setup Claude API key
  $0 setup-claude

EOF
}

demo() {
    log_info "Running RAGGAE demo..."

    # Create sample data
    local sample_dir="${DATA_DIR}/demo"
    mkdir -p "$sample_dir"

    # Create a sample text file
    cat > "${sample_dir}/sample_tender.txt" << 'SAMPLE'
TENDER RESPONSE: Cloud Infrastructure Services

1. SECURITY CERTIFICATIONS

Our organization holds the following certifications:
- ISO/IEC 27001:2022 certified by Bureau Veritas
- SOC 2 Type II compliant
- GDPR compliant data processing

2. TECHNICAL CAPABILITIES

2.1 Container Orchestration
We provide enterprise Kubernetes clusters with:
- Auto-scaling and self-healing
- GitOps-based deployments using ArgoCD
- Service mesh with Istio

2.2 MLOps Platform
Our MLOps infrastructure includes:
- MLflow for experiment tracking and model registry
- Kubeflow for ML pipelines
- GPU support for training workloads

3. DATA SOVEREIGNTY

All data is processed and stored within EU data centers:
- Primary: Paris, France
- Backup: Frankfurt, Germany
- No data transfer outside EU without explicit consent
SAMPLE

    log_success "Sample data created: ${sample_dir}/sample_tender.txt"

    # Run pipeline
    run_full_pipeline \
        "$sample_dir" \
        "demo_index" \
        "MLOps platform capabilities" \
        "Provider must be ISO 27001 certified" \
        "Platform uses MLflow for MLOps" \
        "Deployments on Kubernetes with GitOps"
}

# Parse command
case "${1:-help}" in
    check)
        check_prerequisites
        ;;
    index)
        check_prerequisites
        if [[ -d "$2" ]]; then
            index_directory "$2" "${3:-corpus}"
        else
            index_single_file "$2" "${3:-index}"
        fi
        ;;
    search)
        check_prerequisites
        search_index_text "$2" "$3"
        ;;
    quickscore)
        check_prerequisites
        index_name="$2"
        shift 2
        if [[ "$OLLAMA_AVAILABLE" == "true" ]]; then
            quickscore_ollama "$index_name" "$@"
        else
            quickscore_claude "$index_name" "claude-haiku-3-5-20241022" "$@"
        fi
        ;;
    pipeline)
        check_prerequisites
        input="$2"
        index_name="$3"
        query="$4"
        shift 4
        run_full_pipeline "$input" "$index_name" "$query" "$@"
        ;;
    setup-claude)
        setup_claude_key
        ;;
    demo)
        check_prerequisites
        demo
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
