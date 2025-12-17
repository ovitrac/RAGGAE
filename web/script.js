/**
 * RAGGAE Web UI v2.0 - Client-side JavaScript
 *
 * Enhanced interface with batch upload, async indexing, job queue, and progress tracking.
 *
 * Features:
 * - Drag-and-drop batch upload (all document types, ZIP, folders)
 * - Background indexing with real-time progress polling
 * - Job queue management with cancel/refresh
 * - Semantic search with provenance display
 * - NLI-based compliance scoring with audit trail
 * - Export functionality (JSON, CSV)
 *
 * @author Dr. Olivier Vitrac, PhD, HDR
 * @email olivier.vitrac@adservio.com
 * @organization Adservio
 * @date December 17, 2025
 * @license MIT
 */

const API = location.origin;
const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

// =============================================================================
// Supported Extensions
// =============================================================================

const ALLOWED_EXTS = new Set([
  // Documents
  "pdf", "docx", "doc", "odt", "rtf",
  // Presentations
  "pptx", "ppt", "odp",
  // Spreadsheets
  "xlsx", "xls", "ods", "csv",
  // Text
  "txt", "md", "json", "xml", "html",
  // Code
  "py", "java", "js", "ts", "c", "cpp", "h",
  // Config
  "yaml", "yml", "toml", "ini", "cfg",
  // Archives
  "zip"
]);

// =============================================================================
// State
// =============================================================================

let pickedFiles = [];
let currentJobId = null;
let progressPollInterval = null;
let jobsPollInterval = null;

// =============================================================================
// Profiles (Preconfigured requirement sets)
// =============================================================================

const PROFILES = {
  custom: {
    name: "Custom",
    requirements: []
  },
  tender: {
    name: "Tender Evaluation",
    requirements: [
      "Provider must be ISO 27001 certified",
      "Solution must support Kubernetes deployment",
      "Platform must include MLOps capabilities (MLflow or similar)",
      "Service must provide 99.9% SLA availability",
      "Data must be stored within EU (GDPR compliance)",
      "Solution must support multi-tenant architecture",
      "Provider must offer 24/7 technical support"
    ]
  },
  cv: {
    name: "CV Screening",
    requirements: [
      "Candidate has experience with Python programming",
      "Candidate has experience with machine learning frameworks",
      "Candidate has cloud platform experience (AWS, GCP, or Azure)",
      "Candidate has worked with Docker and containerization",
      "Candidate has experience with CI/CD pipelines",
      "Candidate has database management experience (SQL or NoSQL)",
      "Candidate demonstrates leadership or project management experience"
    ]
  },
  compliance: {
    name: "Compliance Check",
    requirements: [
      "Organization has documented security policies",
      "Data encryption is implemented at rest and in transit",
      "Access control mechanisms are in place",
      "Regular security audits are conducted",
      "Incident response procedures are documented",
      "Data retention policies are defined",
      "Privacy impact assessments are performed"
    ]
  }
};

// =============================================================================
// Global Settings
// =============================================================================

const DEFAULT_SETTINGS = {
  model: "intfloat/multilingual-e5-small",
  e5: true,
  nliBackend: "ollama",
  ollamaModel: "mistral",
  claudeModel: "claude-sonnet-4-20250514",
  apiKey: "",
  indexName: "tender",
  workers: 4,
  pandoc: true,
  minChars: 40
};

let globalSettings = { ...DEFAULT_SETTINGS };

function loadSettings() {
  try {
    const saved = localStorage.getItem("raggae-settings");
    if (saved) {
      globalSettings = { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
    }
  } catch (e) {
    console.warn("Could not load settings:", e);
  }
  applySettingsToUI();
}

function saveSettings() {
  try {
    localStorage.setItem("raggae-settings", JSON.stringify(globalSettings));
  } catch (e) {
    console.warn("Could not save settings:", e);
  }
}

function applySettingsToUI() {
  // Modal inputs
  $("#global-model").value = globalSettings.model;
  $("#global-e5").checked = globalSettings.e5;
  $("#global-nli-backend").value = globalSettings.nliBackend;
  $("#global-ollama-model").value = globalSettings.ollamaModel;
  $("#global-claude-model").value = globalSettings.claudeModel;
  $("#global-api-key").value = globalSettings.apiKey;
  $("#global-index-name").value = globalSettings.indexName;
  $("#global-workers").value = globalSettings.workers;
  $("#global-pandoc").checked = globalSettings.pandoc;
  $("#global-min-chars").value = globalSettings.minChars;

  // Toggle Claude options visibility
  const claudeOpts = $("#global-claude-options");
  if (claudeOpts) {
    claudeOpts.style.display = globalSettings.nliBackend === "claude" ? "block" : "none";
  }

  // Apply to Index tab
  const indexNameInput = $("#index-name");
  if (indexNameInput && !indexNameInput.value) {
    indexNameInput.value = globalSettings.indexName;
  }
}

function collectSettingsFromUI() {
  globalSettings.model = $("#global-model").value;
  globalSettings.e5 = $("#global-e5").checked;
  globalSettings.nliBackend = $("#global-nli-backend").value;
  globalSettings.ollamaModel = $("#global-ollama-model").value;
  globalSettings.claudeModel = $("#global-claude-model").value;
  globalSettings.apiKey = $("#global-api-key").value;
  globalSettings.indexName = $("#global-index-name").value;
  globalSettings.workers = parseInt($("#global-workers").value) || 4;
  globalSettings.pandoc = $("#global-pandoc").checked;
  globalSettings.minChars = parseInt($("#global-min-chars").value) || 40;
}

// =============================================================================
// Utilities
// =============================================================================

function escapeHtml(s) {
  if (!s) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function collapseEmptyLines(s) {
  if (!s) return "";
  // Replace 2+ consecutive empty lines (or lines with only whitespace) with a single empty line
  return String(s).replace(/(\n\s*){2,}/g, '\n\n').trim();
}

function getExtension(filename) {
  const parts = filename.split(".");
  return parts.length > 1 ? parts.pop().toLowerCase() : "";
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function extEmoji(ext) {
  const map = {
    "pdf": "📄", "docx": "📝", "doc": "📝", "odt": "📝", "rtf": "📝",
    "pptx": "📊", "ppt": "📊", "odp": "📊",
    "xlsx": "📈", "xls": "📈", "ods": "📈", "csv": "📈",
    "txt": "🧾", "md": "🗒️", "json": "📋", "xml": "📋", "html": "🌐",
    "py": "🐍", "java": "☕", "js": "📜", "ts": "📜",
    "c": "⚙️", "cpp": "⚙️", "h": "⚙️",
    "yaml": "⚙️", "yml": "⚙️", "toml": "⚙️", "ini": "⚙️", "cfg": "⚙️",
    "zip": "📦"
  };
  return map[ext] || "📄";
}

// =============================================================================
// Health Check
// =============================================================================

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const data = await r.json();
    $("#health-dot").classList.remove("dot-off");
    $("#health-dot").classList.add("dot-on");
    $("#health-text").textContent = "online";

    // Check for pandoc
    if (data.pandoc_available !== undefined) {
      const pandocEl = $("#pandoc-status");
      if (data.pandoc_available) {
        pandocEl.textContent = "pandoc ✓";
        pandocEl.classList.remove("muted");
        pandocEl.classList.add("badge-success");
      } else {
        pandocEl.textContent = "pandoc ✗";
      }
    }
  } catch {
    $("#health-dot").classList.remove("dot-on");
    $("#health-dot").classList.add("dot-off");
    $("#health-text").textContent = "offline";
  }
}

// =============================================================================
// Tab Navigation
// =============================================================================

$$(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach(t => t.classList.remove("active"));
    $$(".tabpanel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");

    // Refresh jobs when switching to jobs tab
    if (btn.dataset.tab === "jobs") {
      refreshJobs();
    }
  });
});

// =============================================================================
// Settings Modal
// =============================================================================

function openSettingsModal() {
  applySettingsToUI();
  $("#settings-modal").classList.add("active");
}

function closeSettingsModal() {
  $("#settings-modal").classList.remove("active");
}

$("#btn-settings")?.addEventListener("click", openSettingsModal);
$("#btn-close-settings")?.addEventListener("click", closeSettingsModal);

// Close modal on overlay click
$("#settings-modal")?.addEventListener("click", (e) => {
  if (e.target === $("#settings-modal")) {
    closeSettingsModal();
  }
});

// Close modal on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $("#settings-modal").classList.contains("active")) {
    closeSettingsModal();
  }
});

// Toggle Claude options in modal
$("#global-nli-backend")?.addEventListener("change", (e) => {
  const claudeOpts = $("#global-claude-options");
  if (claudeOpts) {
    claudeOpts.style.display = e.target.value === "claude" ? "block" : "none";
  }
});

// Save settings
$("#btn-save-settings")?.addEventListener("click", () => {
  collectSettingsFromUI();
  saveSettings();
  closeSettingsModal();
});

// =============================================================================
// Profile Selection
// =============================================================================

$("#profile-select")?.addEventListener("change", (e) => {
  const profile = PROFILES[e.target.value];
  if (profile && profile.requirements.length > 0) {
    const textarea = $("#form-quickscore textarea[name='requirements']");
    if (textarea) {
      textarea.value = profile.requirements.join("\n");
    }
  }
});

// Clear requirements button
$("#btn-clear-reqs")?.addEventListener("click", () => {
  const textarea = $("#form-quickscore textarea[name='requirements']");
  if (textarea) {
    textarea.value = "";
  }
  $("#profile-select").value = "custom";
});

// =============================================================================
// Drop Zone & File Selection
// =============================================================================

function isAllowedFile(file) {
  const ext = getExtension(file.name);
  return ALLOWED_EXTS.has(ext);
}

function addFiles(fileList) {
  for (const file of fileList) {
    if (isAllowedFile(file)) {
      // Avoid duplicates
      const exists = pickedFiles.some(f => f.name === file.name && f.size === file.size);
      if (!exists) {
        pickedFiles.push(file);
      }
    }
  }
  updateFileList();
}

function updateFileList() {
  const container = $("#file-list-container");
  const list = $("#file-list");
  const count = $("#file-count");
  const uploadBtn = $("#btn-upload-index");

  if (pickedFiles.length === 0) {
    container.style.display = "none";
    uploadBtn.disabled = true;
    return;
  }

  container.style.display = "block";
  uploadBtn.disabled = false;
  count.textContent = `${pickedFiles.length} file(s) selected`;

  list.innerHTML = pickedFiles.map((f, i) => {
    const ext = getExtension(f.name);
    return `
      <div class="file-item" data-index="${i}">
        <span class="file-icon">${extEmoji(ext)}</span>
        <span class="file-name">${escapeHtml(f.name)}</span>
        <span class="file-size muted">${formatFileSize(f.size)}</span>
        <button class="btn-remove" data-index="${i}" title="Remove">×</button>
      </div>
    `;
  }).join("");

  // Add remove handlers
  list.querySelectorAll(".btn-remove").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.index);
      pickedFiles.splice(idx, 1);
      updateFileList();
    });
  });
}

function clearFiles() {
  pickedFiles = [];
  updateFileList();
  $("#pick-files").value = "";
  $("#pick-folder").value = "";
}

// File pickers
$("#pick-files")?.addEventListener("change", e => addFiles(e.target.files));
$("#pick-folder")?.addEventListener("change", e => addFiles(e.target.files));
$("#btn-clear-files")?.addEventListener("click", clearFiles);

// Drop zone
const dropzone = $("#dropzone");
const overlay = $("#dropzone-overlay");

dropzone?.addEventListener("dragenter", e => {
  e.preventDefault();
  overlay.classList.add("active");
});

dropzone?.addEventListener("dragover", e => {
  e.preventDefault();
  overlay.classList.add("active");
});

dropzone?.addEventListener("dragleave", e => {
  e.preventDefault();
  if (!dropzone.contains(e.relatedTarget)) {
    overlay.classList.remove("active");
  }
});

dropzone?.addEventListener("drop", e => {
  e.preventDefault();
  overlay.classList.remove("active");

  // Handle dropped items
  if (e.dataTransfer.items) {
    const items = Array.from(e.dataTransfer.items);
    items.forEach(item => {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) addFiles([file]);
      }
    });
  } else {
    addFiles(e.dataTransfer.files);
  }
});

// =============================================================================
// Batch Upload & Async Indexing
// =============================================================================

async function uploadAndIndex() {
  if (pickedFiles.length === 0) return;

  const indexName = $("#index-name").value.trim() || globalSettings.indexName || "tender";
  const model = globalSettings.model;
  const workers = globalSettings.workers;
  const useE5 = globalSettings.e5;
  const usePandoc = globalSettings.pandoc;
  const minChars = globalSettings.minChars;

  // Show progress section
  showProgress("Uploading files...", 0);

  try {
    // Step 1: Upload files
    const formData = new FormData();
    for (const file of pickedFiles) {
      formData.append("files", file);
    }

    const uploadResp = await fetch(`${API}/upload-batch`, {
      method: "POST",
      body: formData
    });

    if (!uploadResp.ok) {
      const err = await uploadResp.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || `HTTP ${uploadResp.status}`);
    }

    const uploadData = await uploadResp.json();
    if (!uploadData.ok) {
      throw new Error(uploadData.detail || "Upload failed");
    }

    updateProgress("Starting indexing...", 5);

    // Step 2: Start async indexing
    const indexResp = await fetch(`${API}/index-async`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        key: uploadData.key,
        index_path: indexName,
        model: model,
        e5: useE5,
        prefer_pandoc: usePandoc,
        min_chars: minChars,
        workers: workers
      })
    });

    if (!indexResp.ok) {
      const err = await indexResp.json().catch(() => ({ detail: "Indexing failed" }));
      throw new Error(err.detail || `HTTP ${indexResp.status}`);
    }

    const indexData = await indexResp.json();
    currentJobId = indexData.job_id;

    // Step 3: Start polling for progress
    startProgressPolling();

    // Clear file list
    clearFiles();

  } catch (err) {
    showError(err.message || err.toString());
  }
}

$("#btn-upload-index")?.addEventListener("click", uploadAndIndex);

// =============================================================================
// Progress Tracking
// =============================================================================

function showProgress(title, percent) {
  const section = $("#progress-section");
  section.style.display = "block";
  $("#progress-title").textContent = title;
  $("#progress-bar").style.width = `${percent}%`;
  $("#progress-percent").textContent = `${Math.round(percent)}%`;
  $("#progress-error").style.display = "none";
}

function updateProgress(title, percent, files = null, chunks = null, current = null) {
  $("#progress-title").textContent = title;
  $("#progress-bar").style.width = `${percent}%`;
  $("#progress-percent").textContent = `${Math.round(percent)}%`;

  if (files !== null) {
    $("#progress-files").textContent = files;
  }
  if (chunks !== null) {
    $("#progress-chunks").textContent = `${chunks} chunks`;
  }
  if (current !== null) {
    $("#progress-current").textContent = current;
  }
}

function showError(message) {
  const errEl = $("#progress-error");
  errEl.textContent = `❌ ${message}`;
  errEl.style.display = "block";
  stopProgressPolling();
}

function hideProgress() {
  $("#progress-section").style.display = "none";
  stopProgressPolling();
}

function startProgressPolling() {
  if (progressPollInterval) {
    clearInterval(progressPollInterval);
  }
  progressPollInterval = setInterval(pollJobProgress, 1000);
}

function stopProgressPolling() {
  if (progressPollInterval) {
    clearInterval(progressPollInterval);
    progressPollInterval = null;
  }
  currentJobId = null;
}

async function pollJobProgress() {
  if (!currentJobId) return;

  try {
    const resp = await fetch(`${API}/job/${currentJobId}`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const job = await resp.json();

    // Backend returns flat structure, not nested under 'progress'
    const percent = job.progress_percent || 0;
    const processed = job.files_processed || 0;
    const total = job.files_total || 0;
    const chunks = job.chunks_indexed || 0;
    const currentFile = job.current_file || "";
    const duration = job.elapsed_seconds || 0;

    let title = "Indexing...";
    if (job.status === "completed") {
      title = "Completed!";
      updateProgress(title, 100, `${total} / ${total} files`, chunks, "");

      // Show result as nice card
      $("#out-index").innerHTML = `
        <div class="index-success">
          <div class="success-icon">✓</div>
          <div class="success-details">
            <div class="success-title">Index Created Successfully</div>
            <div class="success-stats">
              <span><strong>${total}</strong> files</span>
              <span><strong>${chunks}</strong> chunks</span>
              <span><strong>${duration.toFixed(1)}s</strong> duration</span>
            </div>
            <div class="success-path">📁 ${escapeHtml(job.index_path || "index")}</div>
          </div>
        </div>
      `;

      stopProgressPolling();
      setTimeout(() => { hideProgress(); }, 3000);
      return;
    } else if (job.status === "failed") {
      showError(job.error || "Job failed");
      return;
    } else if (job.status === "cancelled") {
      showError("Job cancelled");
      return;
    }

    updateProgress(title, percent, `${processed} / ${total} files`, chunks, currentFile);

  } catch (err) {
    console.error("Progress poll error:", err);
  }
}

// Cancel job
$("#btn-cancel-job")?.addEventListener("click", async () => {
  if (!currentJobId) return;

  try {
    const resp = await fetch(`${API}/job/${currentJobId}/cancel`, {
      method: "POST"
    });
    if (resp.ok) {
      showError("Job cancelled");
    }
  } catch (err) {
    console.error("Cancel error:", err);
  }
});

// =============================================================================
// Jobs Tab
// =============================================================================

async function refreshJobs() {
  const list = $("#jobs-list");
  const filter = $("#jobs-filter")?.value || "";

  list.innerHTML = '<div class="muted">Loading...</div>';

  try {
    let url = `${API}/jobs`;
    if (filter) {
      url += `?status=${filter}`;
    }

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const jobs = data.jobs || [];

    if (!jobs || jobs.length === 0) {
      list.innerHTML = '<div class="muted">No jobs found</div>';
      return;
    }

    list.innerHTML = jobs.map(job => {
      const statusClass = `status-${job.status}`;
      // Jobs list returns flat structure from database
      const total = job.files_total || 0;
      const processed = job.files_processed || 0;
      const chunks = job.chunks_indexed || 0;
      const percent = total > 0 ? Math.round((processed / total) * 100) : 0;

      return `
        <div class="job-item ${statusClass}">
          <div class="job-header">
            <span class="job-id">${escapeHtml(job.job_id?.substring(0, 8) || "?")}</span>
            <span class="job-index">${escapeHtml(job.index_path || "")}</span>
            <span class="badge badge-${job.status}">${job.status}</span>
          </div>
          <div class="job-progress">
            <div class="progress-bar-mini">
              <div class="progress-fill" style="width:${percent}%"></div>
            </div>
            <span class="job-stats">${processed}/${total} files • ${chunks} chunks</span>
          </div>
          <div class="job-time muted">
            Created: ${formatTime(job.created_at)}
            ${job.completed_at ? ` • Completed: ${formatTime(job.completed_at)}` : ""}
          </div>
          ${job.error ? `<div class="job-error">Error: ${escapeHtml(job.error)}</div>` : ""}
        </div>
      `;
    }).join("");

  } catch (err) {
    list.innerHTML = `<div class="muted">Error: ${escapeHtml(err.message)}</div>`;
  }
}

function formatTime(isoString) {
  if (!isoString) return "—";
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

$("#btn-refresh-jobs")?.addEventListener("click", refreshJobs);
$("#jobs-filter")?.addEventListener("change", refreshJobs);

// =============================================================================
// Search Tab
// =============================================================================

$("#btn-search")?.addEventListener("click", async () => {
  const form = $("#form-search");
  const payload = {
    index_path: form.index_path.value.trim() || "tender",
    model: globalSettings.model,
    e5: globalSettings.e5,
    query: form.query.value.trim(),
    k: parseInt(form.k.value) || 10,
    context: parseInt(form.context.value) || 0
  };

  if (!payload.query) {
    $("#hits").innerHTML = '<div class="muted">Please enter a search query</div>';
    return;
  }

  $("#hits").innerHTML = '<div class="muted">Searching...</div>';

  try {
    const r = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const data = await r.json();
    const hits = data.hits || [];

    if (hits.length === 0) {
      $("#hits").innerHTML = '<div class="muted">No results found</div>';
      return;
    }

    $("#hits").innerHTML = hits.map((h, i) => {
      const hasContext = (h.context_before && h.context_before.length > 0) ||
                         (h.context_after && h.context_after.length > 0);

      // Build grep-like context display
      let contextHtml = '';
      if (hasContext) {
        // Context before (grayed)
        if (h.context_before && h.context_before.length > 0) {
          contextHtml += h.context_before.map(txt =>
            `<div class="ctx-line ctx-before">${escapeHtml(collapseEmptyLines(txt))}</div>`
          ).join("");
        }

        // The matched block (highlighted)
        contextHtml += `<div class="ctx-line ctx-match">${escapeHtml(collapseEmptyLines(h.snippet || ""))}</div>`;

        // Context after (grayed)
        if (h.context_after && h.context_after.length > 0) {
          contextHtml += h.context_after.map(txt =>
            `<div class="ctx-line ctx-after">${escapeHtml(collapseEmptyLines(txt))}</div>`
          ).join("");
        }
      } else {
        // No context - just show the snippet
        contextHtml = `<div class="ctx-line ctx-match">${escapeHtml(collapseEmptyLines(h.snippet || ""))}</div>`;
      }

      return `
        <div class="search-result${hasContext ? ' with-context' : ''}">
          <div class="result-header">
            <span class="result-rank">${i + 1}</span>
            <span class="result-file">${extEmoji(h.ext)} ${escapeHtml(h.file || "unknown")}</span>
            <span class="result-loc">p.${h.page} b${h.block}</span>
            <span class="result-score">${(h.score || 0).toFixed(4)}</span>
          </div>
          <div class="result-context">
            ${contextHtml}
          </div>
        </div>
      `;
    }).join("");

  } catch (err) {
    $("#hits").innerHTML = `<div class="muted">Error: ${escapeHtml(err.message)}</div>`;
  }
});

// =============================================================================
// Quickscore Tab
// =============================================================================

$("#btn-quickscore")?.addEventListener("click", async () => {
  const form = $("#form-quickscore");
  const reqs = form.requirements.value.split("\n").map(s => s.trim()).filter(Boolean);

  if (reqs.length === 0) {
    $("#quickscore-out").innerHTML = '<div class="muted">Please enter at least one requirement</div>';
    return;
  }

  const nliBackend = globalSettings.nliBackend;

  const payload = {
    index_path: form.index_path.value.trim() || "tender",
    model: globalSettings.model,
    e5: globalSettings.e5,
    requirements: reqs,
    topk: parseInt(form.topk.value) || 5,
    ollama_model: globalSettings.ollamaModel,
    nli_lang: form.nli_lang.value || "auto",
    nli_backend: nliBackend
  };

  // Add Claude-specific parameters if using Claude backend
  if (nliBackend === "claude") {
    const apiKey = globalSettings.apiKey;
    if (!apiKey) {
      $("#quickscore-out").innerHTML = '<div class="muted">Anthropic API key is required. Set it in ⚙️ Settings.</div>';
      return;
    }
    payload.anthropic_api_key = apiKey;
    payload.claude_model = globalSettings.claudeModel;
  }

  $("#quickscore-out").innerHTML = '<div class="muted">Computing...</div>';

  try {
    const r = await fetch(`${API}/quickscore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const data = await r.json();
    const score = Number(data.fit_score || 0).toFixed(0);
    const scoreClass = score >= 70 ? "score-high" : score >= 40 ? "score-mid" : "score-low";

    const head = `
      <div class="qs-header">
        <div class="qs-score ${scoreClass}">${score}<span>/100</span></div>
        <div class="qs-summary">
          <span class="qs-yes">✓ ${(data.verdicts || []).filter(v => v.verdict === "Yes").length}</span>
          <span class="qs-partial">~ ${(data.verdicts || []).filter(v => v.verdict === "Partial").length}</span>
          <span class="qs-no">✗ ${(data.verdicts || []).filter(v => v.verdict === "No").length}</span>
        </div>
      </div>
    `;

    const rows = (data.verdicts || []).map(v => {
      const e = v.evidence || {};
      const warn = (v.rationale || "").toLowerCase().includes("invalid") ||
                   (v.rationale || "").toLowerCase().includes("wrong language");
      const verdictClass = v.verdict === "Yes" ? "verdict-yes" : v.verdict === "Partial" ? "verdict-partial" : "verdict-no";
      const verdictIcon = v.verdict === "Yes" ? "✓" : v.verdict === "Partial" ? "~" : "✗";

      const evals = (v.evaluated || []).map(ev => `
        <div class="eval-row">
          <span class="eval-label label-${ev.label}">${ev.label}</span>
          <span class="eval-file">${extEmoji(ev.ext)} ${escapeHtml(ev.file || "?")} p.${ev.page ?? "?"}</span>
          <span class="eval-score">${(ev.score ?? 0).toFixed(3)}</span>
        </div>
      `).join("");

      return `
        <div class="qs-item ${verdictClass}">
          <div class="qs-item-header">
            <span class="qs-verdict">${verdictIcon}</span>
            <span class="qs-req">${escapeHtml(v.requirement)}</span>
            <span class="qs-evidence">${extEmoji(e.ext)} ${escapeHtml(e.file || "")} p.${e.page ?? "?"}</span>
          </div>
          <div class="qs-rationale">${escapeHtml(v.rationale || "—")}${warn ? ' <span class="qs-warn">⚠</span>' : ''}</div>
          ${(v.evaluated || []).length > 0 ? `
            <details class="qs-details">
              <summary>Evaluated (${(v.evaluated || []).length})</summary>
              <div class="qs-evals">${evals}</div>
            </details>
          ` : ''}
        </div>
      `;
    }).join("");

    $("#quickscore-out").innerHTML = head + rows;

  } catch (err) {
    $("#quickscore-out").innerHTML = `<div class="muted">Error: ${escapeHtml(err.message)}</div>`;
  }
});

// Export
$("#btn-export")?.addEventListener("click", async () => {
  const form = $("#form-quickscore");
  const reqs = form.requirements.value.split("\n").map(s => s.trim()).filter(Boolean);
  const nliBackend = globalSettings.nliBackend;

  const payload = {
    index_path: form.index_path.value.trim() || "tender",
    model: globalSettings.model,
    e5: globalSettings.e5,
    requirements: reqs,
    topk: parseInt(form.topk.value) || 5,
    ollama_model: globalSettings.ollamaModel,
    nli_lang: form.nli_lang.value || "auto",
    nli_backend: nliBackend,
    format: $("#export-format")?.value || "json"
  };

  // Add Claude-specific parameters if using Claude backend
  if (nliBackend === "claude") {
    const apiKey = globalSettings.apiKey;
    if (!apiKey) {
      alert("Anthropic API key is required. Set it in Settings.");
      return;
    }
    payload.anthropic_api_key = apiKey;
    payload.claude_model = globalSettings.claudeModel;
  }

  try {
    const r = await fetch(`${API}/quickscore/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `quickscore.${payload.format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

  } catch (err) {
    alert("Export failed: " + err.message);
  }
});

// =============================================================================
// Initialize
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  $("#year").textContent = new Date().getFullYear();

  // Load saved settings
  loadSettings();

  checkHealth();
  setInterval(checkHealth, 30000); // Re-check every 30 seconds
});
