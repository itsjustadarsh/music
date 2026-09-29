const API_BASE =
  new URLSearchParams(window.location.search).get("api") ||
  window.MUSIC_AGENT_API_BASE ||
  "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

// ── Demo files ──────────────────────────────────────────────

let DEMO_FILES = [];

async function initDemoFiles() {
  const container = $("demo-file-list");

  // Try to discover available demo files from the API
  try {
    const resp = await fetch(`${API_BASE}/demo-files-list`);
    if (resp.ok) {
      const data = await resp.json();
      DEMO_FILES = data.files;
    }
  } catch {}

  // Fallback to hardcoded synthetic files if API doesn't respond
  if (DEMO_FILES.length === 0) {
    DEMO_FILES = [
      { file: "electronic_synth_demo.wav", label: "Electronic", type: "synthetic" },
      { file: "hiphop_beat_demo.wav", label: "Hip-Hop", type: "synthetic" },
      { file: "rock_powerchord_demo.wav", label: "Rock", type: "synthetic" },
      { file: "folk_arpeggio_demo.wav", label: "Folk", type: "synthetic" },
      { file: "pop_melody_demo.wav", label: "Pop", type: "synthetic" },
      { file: "experimental_glitch_demo.wav", label: "Experimental", type: "synthetic" },
      { file: "instrumental_piano_demo.wav", label: "Instrumental", type: "synthetic" },
      { file: "international_drone_demo.wav", label: "International", type: "synthetic" },
    ];
  }

  const chips = document.createElement("div");
  chips.className = "demo-chips";

  for (const demo of DEMO_FILES) {
    const chip = document.createElement("button");
    chip.className = "demo-chip";
    const icon = demo.type === "fma"
      ? `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>`
      : `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    chip.innerHTML = `${icon}${demo.label}`;
    chip.addEventListener("click", () => loadDemoFile(demo, chip));
    chips.appendChild(chip);
  }
  container.appendChild(chips);
}

async function loadDemoFile(demo, chip) {
  document.querySelectorAll(".demo-chip").forEach((c) => c.classList.remove("active"));
  chip.classList.add("active");

  const path = `${API_BASE.replace(/\/+$/, "")}/demo-files/${demo.file}`;
  try {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`Could not load ${demo.file}`);
    const blob = await resp.blob();
    const file = new File([blob], demo.file, { type: "audio/wav" });
    currentFile = file;
    showSelectedFile(file);
    $("classify-btn").disabled = false;
  } catch (e) {
    $("classify-status").textContent = `Could not load demo file: ${e.message}`;
    $("classify-status").className = "status error";
  }
}

// ── File handling ───────────────────────────────────────────

let currentFile = null;

function showSelectedFile(file) {
  $("file-name").textContent = file.name;
  $("selected-file").hidden = false;
  $("upload-zone").style.display = "none";
  $("audio-player").src = URL.createObjectURL(file);
}

function clearFile() {
  currentFile = null;
  $("selected-file").hidden = true;
  $("upload-zone").style.display = "";
  $("audio-file").value = "";
  $("classify-btn").disabled = true;
  $("all-models-results").hidden = true;
  $("classify-status").textContent = "";
  document.querySelectorAll(".demo-chip").forEach((c) => c.classList.remove("active"));
}

const uploadZone = $("upload-zone");
uploadZone.addEventListener("click", () => $("audio-file").click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("audio/")) {
    currentFile = file;
    showSelectedFile(file);
    $("classify-btn").disabled = false;
    document.querySelectorAll(".demo-chip").forEach((c) => c.classList.remove("active"));
  }
});
$("audio-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    currentFile = file;
    showSelectedFile(file);
    $("classify-btn").disabled = false;
    document.querySelectorAll(".demo-chip").forEach((c) => c.classList.remove("active"));
  }
});
$("clear-file").addEventListener("click", clearFile);

// ── Classification (all models) ─────────────────────────────

$("classify-btn").addEventListener("click", async () => {
  if (!currentFile) return;

  $("classify-status").textContent = `Analysing ${currentFile.name} with all models...`;
  $("classify-status").className = "status loading";
  $("classify-btn").disabled = true;
  $("all-models-results").hidden = true;

  try {
    const form = new FormData();
    form.append("file", currentFile);
    const res = await fetch(`${API_BASE}/classify-all`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderAllModels(data.results);
    $("classify-status").textContent = `${data.results.length} model(s) compared`;
    $("classify-status").className = "status";
  } catch (e) {
    $("classify-status").textContent = `Error: ${e.message}`;
    $("classify-status").className = "status error";
  } finally {
    $("classify-btn").disabled = false;
  }
});

function renderAllModels(results) {
  const container = $("all-models-results");
  container.innerHTML = "";

  // Sort by macro_f1 descending (best model first)
  results.sort((a, b) => (b.macro_f1 || 0) - (a.macro_f1 || 0));
  const bestName = results[0]?.run_name;

  for (const r of results) {
    const isBest = r.run_name === bestName;
    const card = document.createElement("div");
    card.className = `model-card${isBest ? " best" : ""}`;

    const scores = Object.entries(r.all_scores).sort((a, b) => b[1] - a[1]);
    const topGenre = scores[0][0];

    card.innerHTML = `
      <div class="model-card-header">
        <div class="model-name">
          ${escapeHtml(r.run_name)}
          ${isBest ? '<span class="best-badge">Best Model</span>' : ""}
        </div>
        <div class="model-prediction">
          <span class="model-pred-genre">${escapeHtml(r.genre)}</span>
          <span class="model-pred-conf">${(r.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="model-eval-stats">
        ${r.accuracy != null ? `<span class="eval-stat">Accuracy: <strong>${(r.accuracy * 100).toFixed(1)}%</strong></span>` : ""}
        ${r.macro_f1 != null ? `<span class="eval-stat">Macro-F1: <strong>${(r.macro_f1 * 100).toFixed(1)}%</strong></span>` : ""}
      </div>
      <ul class="model-bars">${scores
        .map(
          ([genre, score]) => `
        <li>
          <span class="bar-label">${escapeHtml(genre)}</span>
          <span class="bar-track"><span class="bar-fill ${genre === topGenre ? "top" : ""}" data-width="${(score * 100).toFixed(1)}"></span></span>
          <span class="bar-value">${(score * 100).toFixed(1)}%</span>
        </li>`
        )
        .join("")}
      </ul>
      <div style="margin-top: 14px; display: flex; gap: 8px;">
        <button class="btn-use-genre" data-genre="${escapeHtml(r.genre)}">
          Use "${escapeHtml(r.genre)}" in Step 2
        </button>
      </div>
    `;
    container.appendChild(card);
  }

  container.hidden = false;

  // Animate bars
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll(".bar-fill").forEach((fill) => {
        fill.style.width = fill.dataset.width + "%";
      });
    });
  });

  // Wire up "Use genre" buttons
  container.querySelectorAll(".btn-use-genre").forEach((btn) => {
    btn.addEventListener("click", () => {
      const genre = btn.dataset.genre;
      $("genre").value = genre;
      $("prompt").value = `${genre} music`;
      $("step-playlist").scrollIntoView({ behavior: "smooth", block: "start" });
      // Flash the genre input
      $("genre").style.borderColor = "var(--accent)";
      $("genre").style.boxShadow = "0 0 0 3px var(--accent-dim)";
      setTimeout(() => {
        $("genre").style.borderColor = "";
        $("genre").style.boxShadow = "";
      }, 1500);
    });
  });
}

// ── Playlist Generation ─────────────────────────────────────

let lastRequest = null;

$("generate-btn").addEventListener("click", async () => {
  const prompt = $("prompt").value.trim();
  const genre = $("genre").value.trim() || null;
  const mood = $("mood").value || null;
  const n = parseInt($("count").value, 10) || 15;

  if (!prompt && !genre && !mood) {
    $("generate-status").textContent = "Enter a prompt, genre, or mood to get started.";
    $("generate-status").className = "status error";
    return;
  }

  lastRequest = { prompt: prompt || null, genre, mood, n };

  $("generate-status").textContent = "Generating playlist...";
  $("generate-status").className = "status loading";
  $("generate-btn").disabled = true;

  try {
    const res = await fetch(`${API_BASE}/generate-playlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastRequest),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderPlaylist(data);
    $("generate-status").textContent = "";
    $("generate-status").className = "status";
  } catch (e) {
    $("generate-status").textContent = `Error: ${e.message}`;
    $("generate-status").className = "status error";
  } finally {
    $("generate-btn").disabled = false;
  }
});

function renderPlaylist(data) {
  const meta = $("playlist-meta");
  meta.innerHTML = "";
  if (data.resolved_genre) {
    meta.innerHTML += `<span class="meta-tag">Genre: ${escapeHtml(data.resolved_genre)}</span>`;
  }
  meta.innerHTML += `<span class="meta-tag">Mood: ${escapeHtml(data.resolved_mood)}</span>`;
  meta.innerHTML += `<span class="meta-tag secondary">${data.tracks.length} tracks</span>`;

  const tbody = $("track-tbody");
  tbody.innerHTML = "";
  for (let i = 0; i < data.tracks.length; i++) {
    const t = data.tracks[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="td-num">${i + 1}</td>
      <td class="td-track" title="${escapeHtml(t.name)}">${escapeHtml(t.name)}</td>
      <td class="td-artist" title="${escapeHtml(t.artist)}">${escapeHtml(t.artist)}</td>
      <td class="td-genre">${escapeHtml(t.genre)}</td>
      <td class="td-num">${Math.round(t.tempo)}</td>
      <td class="td-num">${t.key}</td>
      <td class="td-num">${t.valence != null ? t.valence.toFixed(2) : "-"}</td>
      <td class="td-num">${t.energy != null ? t.energy.toFixed(2) : "-"}</td>
      <td class="td-num td-score">${t.score.toFixed(2)}</td>`;
    tbody.appendChild(tr);
  }

  $("step-results").hidden = false;
  $("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Push to Spotify ─────────────────────────────────────────

$("push-btn").addEventListener("click", async () => {
  const playlistName = $("playlist-name").value.trim() || "Music Agent Playlist";
  if (!lastRequest) return;

  $("push-status").textContent = "Pushing to Spotify (a browser window may open for login)...";
  $("push-status").className = "status loading";
  $("push-btn").disabled = true;

  try {
    const res = await fetch(`${API_BASE}/push-playlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...lastRequest, playlist_name: playlistName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    $("push-status").innerHTML =
      `Created "<strong>${escapeHtml(playlistName)}</strong>" with ${data.added_count} tracks. ` +
      (data.skipped_tracks.length ? `${data.skipped_tracks.length} skipped. ` : "") +
      `<a href="${data.playlist_url}" target="_blank" style="color:var(--accent)">Open in Spotify</a>`;
    $("push-status").className = "status";
  } catch (e) {
    $("push-status").textContent = `Error: ${e.message}`;
    $("push-status").className = "status error";
  } finally {
    $("push-btn").disabled = false;
  }
});

// ── Mood Explorer with Circumplex Canvas ────────────────────

const MOOD_COLORS = {
  happy: "#facc15",
  excited: "#f97316",
  tense: "#ef4444",
  angry: "#dc2626",
  sad: "#6366f1",
  tired: "#8b5cf6",
  relaxed: "#06b6d4",
  content: "#22c55e",
  neutral: "#6b7280",
};

const MOOD_ORDER = ["happy", "excited", "tense", "angry", "sad", "tired", "relaxed", "content"];

function drawCircumplex(valence, energy) {
  const canvas = $("circumplex-canvas");
  const dpr = window.devicePixelRatio || 1;
  const size = 300;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 20;

  // Background
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, size, size);

  // Draw octant sectors
  for (let i = 0; i < 8; i++) {
    const startAngle = (i * 45 - 90) * (Math.PI / 180); // -90 to rotate so 0deg is right
    const endAngle = ((i + 1) * 45 - 90) * (Math.PI / 180);
    // Map: canvas angles go clockwise, circumplex goes counterclockwise
    // Octant 0 (happy) is 0-45deg from positive x = top-right in our canvas
    const cStartAngle = -(i + 1) * 45 * (Math.PI / 180) + Math.PI / 2;
    const cEndAngle = -i * 45 * (Math.PI / 180) + Math.PI / 2;

    const mood = MOOD_ORDER[i];
    const color = MOOD_COLORS[mood];

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, cStartAngle, cEndAngle);
    ctx.closePath();
    ctx.fillStyle = color + "18";
    ctx.fill();
    ctx.strokeStyle = color + "30";
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Label
    const midAngle = (cStartAngle + cEndAngle) / 2;
    const labelR = r * 0.72;
    const lx = cx + Math.cos(midAngle) * labelR;
    const ly = cy + Math.sin(midAngle) * labelR;
    ctx.fillStyle = color + "cc";
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(mood.charAt(0).toUpperCase() + mood.slice(1), lx, ly);
  }

  // Neutral circle
  const neutralR = 0.15 * r;
  ctx.beginPath();
  ctx.arc(cx, cy, neutralR, 0, Math.PI * 2);
  ctx.fillStyle = "#6b728015";
  ctx.fill();
  ctx.strokeStyle = "#6b728040";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#6b7280aa";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Neutral", cx, cy + neutralR + 12);

  // Axis lines
  ctx.strokeStyle = "#ffffff15";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "#ffffff55";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Valence +", cx + r - 20, cy + 14);
  ctx.fillText("Valence -", cx - r + 20, cy + 14);
  ctx.textAlign = "center";
  ctx.fillText("Energy +", cx, cy - r + 12);
  ctx.fillText("Energy -", cx, cy + r - 6);

  // Current point
  const v = Math.max(0, Math.min(1, valence)) * 2 - 1;
  const a = Math.max(0, Math.min(1, energy)) * 2 - 1;
  const px = cx + v * r;
  const py = cy - a * r; // y is inverted in canvas

  // Determine current mood for glow color
  const radius = Math.hypot(v, a);
  let currentMood = "neutral";
  if (radius >= 0.15) {
    const angle = ((Math.atan2(a, v) * 180) / Math.PI + 360) % 360;
    currentMood = MOOD_ORDER[Math.floor(angle / 45) % 8];
  }
  const pointColor = MOOD_COLORS[currentMood];

  // Glow
  const glow = ctx.createRadialGradient(px, py, 0, px, py, 20);
  glow.addColorStop(0, pointColor + "60");
  glow.addColorStop(1, pointColor + "00");
  ctx.fillStyle = glow;
  ctx.fillRect(px - 20, py - 20, 40, 40);

  // Point
  ctx.beginPath();
  ctx.arc(px, py, 7, 0, Math.PI * 2);
  ctx.fillStyle = pointColor;
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Crosshair lines to point
  ctx.strokeStyle = pointColor + "40";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(px, cy);
  ctx.lineTo(px, py);
  ctx.moveTo(cx, py);
  ctx.lineTo(px, py);
  ctx.stroke();
  ctx.setLineDash([]);
}

function updateMood() {
  const valence = parseFloat($("mood-valence").value);
  const energy = parseFloat($("mood-energy").value);

  $("val-display").textContent = valence.toFixed(2);
  $("eng-display").textContent = energy.toFixed(2);

  drawCircumplex(valence, energy);

  // Compute mood locally (fast, no API lag)
  const v = Math.max(0, Math.min(1, valence)) * 2 - 1;
  const a = Math.max(0, Math.min(1, energy)) * 2 - 1;
  const radius = Math.hypot(v, a);

  if (radius < 0.15) {
    $("mood-label").textContent = "neutral";
    $("mood-label").style.color = MOOD_COLORS.neutral;
  } else {
    const angle = ((Math.atan2(a, v) * 180) / Math.PI + 360) % 360;
    const mood = MOOD_ORDER[Math.floor(angle / 45) % 8];
    $("mood-label").textContent = mood;
    $("mood-label").style.color = MOOD_COLORS[mood];
    $("mood-angle").textContent = `angle: ${angle.toFixed(0)}deg`;
  }
  $("mood-radius").textContent = `radius: ${radius.toFixed(3)}`;
}

// Click on canvas to set mood
$("circumplex-canvas").addEventListener("click", (e) => {
  const canvas = $("circumplex-canvas");
  const rect = canvas.getBoundingClientRect();
  const size = 300;
  const r = size / 2 - 20;
  const cx = size / 2;
  const cy = size / 2;

  const x = ((e.clientX - rect.left) / rect.width) * size;
  const y = ((e.clientY - rect.top) / rect.height) * size;

  // Convert canvas coords to valence/energy 0..1
  const v = Math.max(0, Math.min(1, ((x - cx) / r + 1) / 2));
  const a = Math.max(0, Math.min(1, (-(y - cy) / r + 1) / 2));

  $("mood-valence").value = v;
  $("mood-energy").value = a;
  updateMood();
});

let moodTimeout = null;
function debouncedMoodUpdate() {
  $("val-display").textContent = parseFloat($("mood-valence").value).toFixed(2);
  $("eng-display").textContent = parseFloat($("mood-energy").value).toFixed(2);
  clearTimeout(moodTimeout);
  moodTimeout = setTimeout(updateMood, 30);
}

$("mood-valence").addEventListener("input", debouncedMoodUpdate);
$("mood-energy").addEventListener("input", debouncedMoodUpdate);

// ── Health Check ────────────────────────────────────────────

async function checkHealth() {
  const dot = $("health-dot");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    dot.className = "health-dot online";
    dot.title = `API online | Model: ${data.model_loaded ? "loaded" : "not loaded"} | Curator: ${data.curator_loaded ? "loaded" : "not loaded"}`;
  } catch {
    dot.className = "health-dot offline";
    dot.title = "API offline - start with: uvicorn src.api.main:app --port 8000";
  }
}

// ── Utils ───────────────────────────────────────────────────

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ── Init ────────────────────────────────────────────────────

initDemoFiles();
checkHealth();
updateMood();
setInterval(checkHealth, 30000);
