/**
 * RAGGAE Web UI - Client-side JavaScript
 *
 * Handles file uploads, API communication, and UI interactions for the
 * RAGGAE semantic search and compliance scoring system.
 *
 * Features:
 * - Drag-and-drop file upload (PDF, DOCX, TXT, ODT, MD, ZIP)
 * - Semantic search with provenance display
 * - NLI-based compliance scoring with audit trail
 * - Export functionality (JSON, CSV)
 *
 * @author Dr. Olivier Vitrac, PhD, HDR
 * @email olivier.vitrac@adservio.com
 * @organization Adservio
 * @date October 31, 2025
 * @license MIT
 */

const API = location.origin; // same host/port as FastAPI
const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

// --- Drop zone & pickers ---
let pickedFiles = [];   // File objects waiting to upload

function addFiles(list){
  for (const f of list){
    const name = f.name.toLowerCase();
    const ok = name.endsWith(".pdf") || name.endsWith(".zip") ||
               name.endsWith(".docx") || name.endsWith(".txt") ||
               name.endsWith(".odt") || name.endsWith(".md");
    if (ok) pickedFiles.push(f);
  }
  $("#upload-status").textContent = `${pickedFiles.length} file(s) selected`;
}


$("#pick-files").addEventListener("change", (e)=> addFiles(e.target.files));
$("#pick-folder").addEventListener("change", (e)=> addFiles(e.target.files));

const dz = $("#dropzone");
dz.addEventListener("dragover", (e)=>{ e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", ()=> dz.classList.remove("dragover"));
dz.addEventListener("drop", (e)=>{
  e.preventDefault(); dz.classList.remove("dragover");
  addFiles(e.dataTransfer.files);
});

// Upload (single or multi). If exactly one .zip, use /upload; otherwise /upload-multi
$("#btn-upload").addEventListener("click", async ()=>{
  if (!pickedFiles.length){ $("#upload-status").textContent = "Pick or drop files first."; return; }
  $("#upload-status").textContent = "Uploading…";
  try{
    let data;
    if (pickedFiles.length === 1 && pickedFiles[0].name.toLowerCase().endsWith(".zip")){
      const fd = new FormData();
      fd.append("file", pickedFiles[0]);
      const r = await fetch(`${API}/upload`, { method:"POST", body: fd });
      data = await r.json();
      if (!r.ok || !data.ok) throw new Error(data.detail || "Upload failed");
      $("#upload-status").textContent = `✅ Unzipped: ${data.files} PDF(s). key=${data.key}`;
      $("#form-index").key.value = data.key;
    } else {
      const fd = new FormData();
      for (const f of pickedFiles) fd.append("files", f);
      const r = await fetch(`${API}/upload-multi`, { method:"POST", body: fd });
      data = await r.json();
      if (!r.ok || !data.ok) throw new Error(data.detail || "Multi-upload failed");
      $("#upload-status").textContent = `✅ Uploaded ${data.files.length} PDF(s). key=${data.key}`;
      $("#form-index").key.value = data.key;
    }
    // reset local buffer
    pickedFiles = [];
  }catch(e){
    $("#upload-status").textContent = "❌ " + e.toString();
  }
});

// Health
async function ping() {
  try {
    const r = await fetch(`${API}/health`);
    const data = await r.json();
    $("#health-dot").classList.remove("dot-off");
    $("#health-dot").classList.add("dot-on");
    $("#health-text").textContent = "online";
  } catch {
    $("#health-dot").classList.remove("dot-on");
    $("#health-dot").classList.add("dot-off");
    $("#health-text").textContent = "offline";
  }
}

// Tabs
$$(".tab").forEach(b => {
  b.addEventListener("click", () => {
    $$(".tab").forEach(t=>t.classList.remove("active"));
    $$(".tabpanel").forEach(p=>p.classList.remove("active"));
    b.classList.add("active");
    $(`#tab-${b.dataset.tab}`).classList.add("active");
  });
});

// Index
function selectedExts(){
  const opts = Array.from($("#exts").options);
  return opts.filter(o => o.selected).map(o => o.value);
}

$("#btn-index").addEventListener("click", async () => {
  const f = $("#form-index");
  const payload = {
    key: f.key.value.trim(),                // prefer key from upload
    index_path: f.index_path.value.trim(),
    model: f.model.value.trim(),
    e5: f.e5.checked,
    min_chars: Number(f.min_chars.value || 40),
    extensions: selectedExts()              // <- multi-extension array
  };
  $("#out-index").textContent = "Indexing…";
  try {
    const r = await fetch(`${API}/index`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    $("#out-index").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("#out-index").textContent = "Error: " + e.toString();
  }
});

// Search
function extEmoji(ext){
  switch((ext||"").toLowerCase()){
    case "pdf": return "📄";
    case "docx": return "📝";
    case "odt": return "📝";
    case "md": return "🗒️";
    case "txt": return "🧾";
    default: return "📄";
  }
}

$("#btn-search").addEventListener("click", async () => {
  const f = $("#form-search");
  const payload = {
    index_path: f.index_path.value.trim(),
    model: f.model.value.trim(),
    e5: f.e5.checked,
    query: f.query.value.trim(),
    k: Number(f.k.value || 10)
  };
  $("#hits").innerHTML = "<div class='muted'>Searching…</div>";
  try {
    const r = await fetch(`${API}/search`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    const list = (data.hits || []).map(h => `
      <div class="hit">
        <div class="fileline">
          <span class="badge"><span class="emoji">${extEmoji(h.ext)}</span> ${h.ext || "file"}</span>
          <span class="filename">${escapeHtml(h.file || "(unknown)")}</span>
          <span class="muted">p.${h.page} b${h.block}</span>
          <span class="muted" style="margin-left:auto">score: ${h.score.toFixed(4)}</span>
        </div>
        <div>${escapeHtml(h.snippet)}…</div>
      </div>
    `).join("") || "<div class='muted'>No hits</div>";
    $("#hits").innerHTML = list;
  } catch (e) {
    $("#hits").innerHTML = `<div class='muted'>Error: ${e}</div>`;
  }
});

// Quickscore
$("#btn-quickscore").addEventListener("click", async () => {
  const f = $("#form-quickscore");
  const reqs = f.requirements.value.split("\n").map(s=>s.trim()).filter(Boolean);
  const payload = {
    index_path: f.index_path.value.trim(),
    model: f.model.value.trim(),
    e5: f.e5.checked,
    requirements: reqs,
    topk: Number(f.topk.value || 5),
    ollama_model: f.ollama_model.value.trim(),
    nli_lang: f.nli_lang.value || "auto"
  };
  $("#quickscore-out").innerHTML = "<div class='muted'>Computing…</div>";
  try {
    const r = await fetch(`${API}/quickscore`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await r.json();

    const head = `
      <div class="hit"><div><strong>Fit score:</strong> ${Number(data.fit_score).toFixed(1)} / 100</div></div>
    `;

    const rows = (data.verdicts || []).map(v => {
      const e = v.evidence || {};
      const warn = (v.rationale || "").toLowerCase().includes("invalid") || (v.rationale || "").toLowerCase().includes("wrong language");

      // Build evaluated list
      const evals = (v.evaluated || []).map(ev => `
        <div class="eval-item">
          <div class="eval-head">
            <span class="badge label-${ev.label}">${ev.label}</span>
            <span class="badge"><span class="emoji">${extEmoji(ev.ext)}</span> ${ev.ext || "file"}</span>
            <span class="filename">${escapeHtml(ev.file || "(unknown)")}</span>
            <span class="muted">p.${ev.page ?? "?"} b${ev.block ?? "?"}</span>
            <span class="muted" style="margin-left:auto">score: ${(ev.score ?? 0).toFixed(4)}</span>
          </div>
          <div class="eval-rationale">${escapeHtml(ev.rationale || "—")}</div>
          <div class="muted" style="margin-top:4px">${escapeHtml(ev.snippet || "")}…</div>
        </div>
      `).join("");

      return `
        <div class="hit">
          <div><strong>${escapeHtml(v.requirement)}</strong></div>
          <div class="fileline">
            <span class="badge"><span class="emoji">${extEmoji(e.ext)}</span> ${e.ext || "file"}</span>
            <span class="filename">${escapeHtml(e.file || "(unknown)")}</span>
            <span class="muted">p.${e.page ?? "?"} b${e.block ?? "?"}</span>
            <span class="muted" style="margin-left:auto">score: ${(e.score ?? 0).toFixed(4)}</span>
          </div>
          <div>Verdict: <strong>${v.verdict}</strong> ${warn ? '<span class="badge warn">LLM output sanitized</span>' : ''}</div>
          <div class="rationale">${escapeHtml(v.rationale || "—")}</div>

          <details class="eval">
            <summary>Show evaluated clauses (${(v.evaluated||[]).length})</summary>
            <div class="eval-list">
              ${evals || '<div class="muted">No evaluated clauses recorded.</div>'}
            </div>
          </details>
        </div>
      `;
    }).join("");

    $("#quickscore-out").innerHTML = head + rows;

  } catch (e) {
    $("#quickscore-out").innerHTML = `<div class='muted'>Error: ${e}</div>`;
  }
});

function escapeHtml(s){
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

$("#year").textContent = new Date().getFullYear();
ping();

// export
function quickscorePayload(){
  const f = $("#form-quickscore");
  const reqs = f.requirements.value.split("\n").map(s=>s.trim()).filter(Boolean);
  return {
    index_path: f.index_path.value.trim(),
    model: f.model.value.trim(),
    e5: f.e5.checked,
    requirements: reqs,
    topk: Number(f.topk.value || 5),
    ollama_model: f.ollama_model.value.trim(),
    nli_lang: f.nli_lang.value || "auto"
  };
}

$("#btn-export").addEventListener("click", async ()=>{
  const payload = quickscorePayload();
  const fmt = $("#export-format").value || "json";
  payload.format = fmt;

  try{
    const r = await fetch(`${API}/quickscore/export`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    if (!r.ok) {
      const err = await r.json().catch(()=>({detail:`HTTP ${r.status}`}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `quickscore.${fmt}`;
    document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
  }catch(e){
    alert("Export failed: " + e.toString());
  }
});

