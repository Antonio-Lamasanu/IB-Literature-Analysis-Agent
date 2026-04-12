import React, { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getFileStem(filename) {
  if (!filename) return "extracted";
  const lastDot = filename.lastIndexOf(".");
  return lastDot > 0 ? filename.slice(0, lastDot) : filename;
}

function parseDownloadFilename(contentDispositionHeader, fallbackName) {
  if (!contentDispositionHeader) return fallbackName;
  const utf8Match = contentDispositionHeader.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) return decodeURIComponent(utf8Match[1]);
  const basicMatch = contentDispositionHeader.match(/filename="?([^";]+)"?/i);
  return basicMatch?.[1] || fallbackName;
}

async function parseErrorDetail(response, fallback = "Request failed.") {
  const ct = response.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    try {
      const data = await response.clone().json();
      return data?.detail || fallback;
    } catch {
      return (await response.text()) || fallback;
    }
  }
  return (await response.text()) || fallback;
}

function formatSeconds(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0.000s";
  return `${n.toFixed(3)}s`;
}

function toApiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path}`;
}

function getModelTier(filename) {
  if (!filename) return null;
  const lower = filename.toLowerCase();
  if (lower.includes("12b")) return "12b";
  if (lower.includes("9b")) return "9b";
  if (lower.includes("4b")) return "4b";
  return null;
}

function shortModelName(filename) {
  if (!filename) return "No model";
  let name = filename.replace(/\.gguf$/i, "");
  name = name.replace(/-Q\d+_K_M$/i, "");
  name = name.replace(/-SPPO-Iter\d+$/i, "");
  name = name.replace(/-[Ii]t$/i, "");
  return name.replace(/-/g, " ").replace(/\b(\w)/g, (c) => c.toUpperCase());
}

// ─── Router Debug Panel ───────────────────────────────────────────────────────

function ScoreBar({ label, value, threshold, color = "#6366f1" }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  const thPct = threshold != null ? Math.max(0, Math.min(1, threshold)) * 100 : null;
  return (
    <div className="rdbg-row">
      <span className="rdbg-label">{label}</span>
      <div className="rdbg-bar-wrap" style={{ position: "relative" }}>
        <div className="rdbg-bar" style={{ width: `${pct}%`, background: color }} />
        {thPct != null && (
          <div className="rdbg-threshold-mark" style={{ left: `${thPct}%` }} title={`threshold: ${threshold}`} />
        )}
      </div>
      <span className="rdbg-val">{value != null ? value.toFixed(3) : "n/a"}</span>
    </div>
  );
}

function RouterDebugPanel({ debug }) {
  if (!debug) return null;
  const isRag = debug.mode === "rag";
  const modeClass = isRag ? "rdbg-mode-rag" : "rdbg-mode-bk";
  const modeLabel = isRag ? "RAG" : "Base Knowledge";
  const reasonMap = {
    context_need_high: "context need ↑",
    known_work_confidence_high: "known work confidence ↑",
    has_conversation_history: "conversation history",
    top_semantic_score_high: "semantic score ↑ (legacy)",
    default: "default",
    override: "manual override",
  };
  return (
    <details className="rdbg-details">
      <summary>
        ▸ Router Debug &nbsp;
        <span className={`rdbg-mode-badge ${modeClass}`}>{modeLabel}</span>
        &nbsp;·&nbsp;{reasonMap[debug.reason] ?? debug.reason}
      </summary>
      <div className="rdbg-body">
        <ScoreBar
          label="Context need score"
          value={debug.context_need_score}
          threshold={debug.context_need_threshold}
        />
        <ScoreBar
          label="Known work confidence"
          value={debug.known_work_confidence}
          threshold={debug.confidence_threshold}
          color="#8b5cf6"
        />
        <ScoreBar
          label="Top semantic score"
          value={debug.top_semantic_score}
          threshold={debug.semantic_threshold}
          color="#06b6d4"
        />
        {(debug.top_k_matches ?? []).length > 0 && (
          <div className="rdbg-matches">
            <div className="dbg-section" style={{ fontSize: "0.73rem", margin: "8px 0 6px" }}>
              Top matched questions (K={debug.top_k_matches.length})
            </div>
            {debug.top_k_matches.map((m, i) => (
              <div key={i} className="rdbg-match">
                <span className="rdbg-match-sim">{m.similarity?.toFixed(3)}</span>
                <span className="rdbg-match-score">→ {m.score?.toFixed(2)}</span>
                <span className="rdbg-match-q">
                  {m.question}
                  <span className="rdbg-cat">[{m.category}]</span>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}

// ─── CSS ──────────────────────────────────────────────────────────────────────

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f0f4f8;
  --surface: #ffffff;
  --surface-2: #f8fafc;
  --surface-3: #f1f5f9;
  --text: #0f172a;
  --text-2: #334155;
  --muted: #64748b;
  --accent: #4f46e5;
  --accent-hover: #4338ca;
  --accent-soft: #eef2ff;
  --accent-border: #c7d2fe;
  --border: #e2e8f0;
  --shadow: 0 4px 20px rgba(15,23,42,.08);
  --shadow-sm: 0 1px 4px rgba(15,23,42,.05);
  --shadow-lg: 0 8px 32px rgba(15,23,42,.13);
  --error: #dc2626;
  --error-bg: #fef2f2;
  --success: #16a34a;
  --success-bg: #f0fdf4;
  --success-border: #bbf7d0;
  --warning: #b45309;
  --warning-bg: #fffbeb;
  --warning-border: #fde68a;
  --radius: 12px;
  --radius-sm: 8px;
  --radius-lg: 16px;
  --header-h: 62px;
  font-size: 16px;
}

[data-theme="dark"] {
  --bg: #0c1020;
  --surface: #161e30;
  --surface-2: #1a2438;
  --surface-3: #1e2a42;
  --text: #e2e8f0;
  --text-2: #94a3b8;
  --muted: #64748b;
  --accent: #818cf8;
  --accent-hover: #a5b4fc;
  --accent-soft: #1e1b4b;
  --accent-border: #3730a3;
  --border: #1e293b;
  --shadow: 0 4px 20px rgba(0,0,0,.5);
  --shadow-sm: 0 1px 4px rgba(0,0,0,.4);
  --shadow-lg: 0 8px 32px rgba(0,0,0,.65);
  --error: #f87171;
  --error-bg: #1f1010;
  --success: #4ade80;
  --success-bg: #0a1a10;
  --success-border: #166534;
  --warning: #fbbf24;
  --warning-bg: #1c1408;
  --warning-border: #92400e;
}

html, body { min-height: 100vh; background: var(--bg); }
body {
  font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', system-ui, sans-serif;
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

/* ── Layout ── */
.layout { display: flex; flex-direction: column; min-height: 100vh; }

/* ── Header ── */
.app-header {
  position: sticky; top: 0; z-index: 100;
  height: var(--header-h);
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center;
  padding: 0 24px; gap: 12px;
  box-shadow: var(--shadow-sm);
}
.header-left { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }
.header-right { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }

.back-btn {
  display: inline-flex; align-items: center; gap: 5px;
  background: none; border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 6px 12px;
  font: 500 0.82rem 'Inter', sans-serif; color: var(--text-2);
  cursor: pointer; transition: border-color .15s, color .15s;
  white-space: nowrap;
}
.back-btn:hover { border-color: var(--accent); color: var(--accent); }

.header-title { font-size: 1rem; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Theme toggle */
.theme-toggle {
  display: flex; align-items: center; gap: 2px;
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 3px;
}
.theme-opt {
  background: none; border: none; border-radius: 6px;
  padding: 4px 9px; font: 500 0.75rem 'Inter', sans-serif;
  color: var(--muted); cursor: pointer; transition: all .15s;
  white-space: nowrap;
}
.theme-opt.active { background: var(--surface); color: var(--text); box-shadow: var(--shadow-sm); }

/* Model selector */
.model-wrap { display: flex; align-items: center; gap: 7px; }
.model-select {
  padding: 5px 10px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface-2);
  color: var(--text); font: inherit; font-size: 0.78rem;
  cursor: pointer; max-width: 200px; transition: border-color .15s;
}
.model-select:focus { outline: none; border-color: var(--accent); }
.model-select:disabled { opacity: .6; cursor: not-allowed; }
.model-status { font-size: 0.75rem; color: var(--muted); white-space: nowrap; }
.model-err { font-size: 0.75rem; color: var(--error); white-space: nowrap; }

/* ── Page ── */
.page { flex: 1; max-width: 960px; width: 100%; margin: 0 auto; padding: 32px 24px; }

/* ── Typography ── */
.page-title { font-size: 1.8rem; font-weight: 800; letter-spacing: -.4px; color: var(--text); margin-bottom: 6px; }
.page-sub { font-size: 0.95rem; color: var(--muted); }
.section-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 14px; }

/* ── Cards ── */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 24px;
  box-shadow: var(--shadow-sm);
}
.card + .card { margin-top: 16px; }

/* ── Buttons ── */
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  border: none; border-radius: var(--radius-sm);
  padding: 9px 18px; font: 600 0.875rem 'Inter', sans-serif;
  cursor: pointer; background: var(--accent); color: #fff;
  transition: background .15s, opacity .15s;
  white-space: nowrap;
}
.btn:hover:not(:disabled) { background: var(--accent-hover); }
.btn:disabled { opacity: .5; cursor: not-allowed; }
.btn-sm { padding: 6px 12px; font-size: 0.8rem; }
.btn-ghost {
  background: none; color: var(--text-2);
  border: 1px solid var(--border);
}
.btn-ghost:hover:not(:disabled) { background: var(--surface-2); border-color: var(--muted); }
.btn-danger { background: none; color: var(--error); border: 1px solid var(--border); }
.btn-danger:hover:not(:disabled) { background: var(--error-bg); border-color: var(--error); }

/* ── Form elements ── */
.file-input {
  flex: 1; min-width: 220px;
  padding: 9px 12px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface-2);
  color: var(--text); font: inherit; font-size: 0.875rem;
}
.file-input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }

.input, .select {
  padding: 9px 12px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--surface-2);
  color: var(--text); font: inherit; font-size: 0.875rem;
  width: 100%; transition: border-color .15s;
}
.input:focus, .select:focus { outline: none; border-color: var(--accent); }

/* ── Status / progress ── */
.status-bar {
  display: flex; justify-content: space-between;
  font-size: 0.82rem; color: var(--muted); margin-bottom: 8px;
}
.progress-track {
  height: 5px; border-radius: 999px;
  background: var(--border); overflow: hidden; margin-bottom: 12px;
}
.progress-fill { height: 100%; border-radius: 999px; background: var(--accent); }
.progress-fill.idle { width: 0; }
.progress-fill.running { width: 40%; animation: pslide 1.2s ease-in-out infinite; }
.progress-fill.done { width: 100%; }
@keyframes pslide {
  0% { transform: translateX(-120%); }
  100% { transform: translateX(280%); }
}

/* ── Messages ── */
.msg { padding: 10px 14px; border-radius: var(--radius-sm); font-size: 0.875rem; margin-top: 10px; }
.msg-error { background: var(--error-bg); color: var(--error); font-weight: 600; }
.msg-success { background: var(--success-bg); color: var(--success); }

/* ── Upload result ── */
.result-card {
  margin-top: 14px; padding: 14px 16px;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--surface-2); font-size: 0.82rem;
}
.result-meta { color: var(--text-2); margin-bottom: 3px; }
.result-meta strong { color: var(--text); }
.result-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.dl-link {
  display: inline-flex; align-items: center; gap: 5px;
  text-decoration: none; background: var(--accent); color: #fff;
  border-radius: var(--radius-sm); padding: 7px 14px;
  font: 600 0.82rem 'Inter', sans-serif; transition: background .15s;
}
.dl-link:hover { background: var(--accent-hover); }

/* ── Dashboard ── */
.dash-hero { margin-bottom: 24px; }

.sysinfo-bar {
  display: flex; flex-wrap: wrap; gap: 8px 20px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px 16px;
  margin-bottom: 16px; font-size: 0.8rem; color: var(--text-2);
}
.sysinfo-chip { display: flex; align-items: center; gap: 5px; }
.sysinfo-lbl { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em; }

.recommend-card {
  display: flex; align-items: flex-start; gap: 14px;
  background: var(--accent-soft); border: 1px solid var(--accent-border);
  border-radius: var(--radius); padding: 16px 20px; margin-bottom: 24px;
}
.recommend-icon { font-size: 1.6rem; flex-shrink: 0; margin-top: 1px; }
.recommend-body { flex: 1; min-width: 0; }
.recommend-label {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .07em; color: var(--accent); margin-bottom: 4px;
}
.recommend-name { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 3px; }
.recommend-reason { font-size: 0.8rem; color: var(--muted); }

.nav-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
}
@media (max-width: 640px) { .nav-grid { grid-template-columns: 1fr; } }

.nav-card {
  background: var(--surface); border: 2px solid var(--border);
  border-radius: var(--radius-lg); padding: 28px 22px;
  cursor: pointer; text-align: left;
  transition: box-shadow .2s, transform .2s, border-color .2s;
  position: relative; overflow: hidden;
}
.nav-card:hover:not(.nav-card--off) {
  box-shadow: var(--shadow-lg); transform: translateY(-3px); border-color: var(--accent);
}
.nav-card--off { opacity: .45; cursor: not-allowed; }
.nav-card-icon { font-size: 2rem; margin-bottom: 12px; display: block; }
.nav-card-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.nav-card-desc { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }
.nav-card-badge {
  position: absolute; top: 14px; right: 14px;
  background: var(--accent-soft); color: var(--accent);
  border: 1px solid var(--accent-border); border-radius: 20px;
  padding: 2px 10px; font: 600 0.72rem 'Inter', sans-serif;
}

/* ── Library ── */
.upload-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }

.doc-list { display: flex; flex-direction: column; gap: 10px; }
.doc-item {
  display: flex; align-items: center; gap: 14px;
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px;
}
.doc-item-icon { font-size: 1.4rem; flex-shrink: 0; }
.doc-item-body { flex: 1; min-width: 0; }
.doc-item-name { font-weight: 600; font-size: 0.9rem; color: var(--text); margin-bottom: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-item-meta { font-size: 0.75rem; color: var(--muted); }
.doc-item-actions { display: flex; gap: 6px; flex-shrink: 0; }
.doc-list-empty { text-align: center; padding: 32px 0; color: var(--muted); font-size: 0.875rem; }

/* ── Learn ── */
.doc-pick-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 12px;
}
.doc-pick-card {
  border: 2px solid var(--border); border-radius: var(--radius);
  padding: 14px; cursor: pointer; background: var(--surface-2); transition: all .15s;
}
.doc-pick-card:hover:not(.disabled) { border-color: var(--accent); }
.doc-pick-card.selected { border-color: var(--accent); background: var(--accent-soft); }
.doc-pick-card.disabled { opacity: .4; cursor: not-allowed; }
.doc-pick-name { font-weight: 600; font-size: 0.875rem; color: var(--text); margin-bottom: 4px; }
.doc-pick-meta { font-size: 0.75rem; color: var(--muted); }

.chat-panel { display: flex; flex-direction: column; gap: 12px; }
.chat-list {
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 12px; max-height: 420px; overflow-y: auto;
  background: var(--surface-2); display: flex; flex-direction: column; gap: 10px;
}
.chat-empty { text-align: center; color: var(--muted); font-size: 0.875rem; padding: 24px 0; }
.chat-msg-block {}
.chat-timing { font-size: 0.72rem; color: var(--muted); text-align: center; margin-bottom: 3px; }
.chat-msg {
  padding: 10px 14px; border-radius: var(--radius-sm);
  font-size: 0.875rem; line-height: 1.55; white-space: pre-wrap;
}
.chat-msg.user { background: var(--accent-soft); border: 1px solid var(--accent-border); margin-left: 28px; }
.chat-msg.assistant { background: var(--surface); border: 1px solid var(--border); margin-right: 28px; }

.chat-form { display: flex; gap: 8px; align-items: flex-end; }
.chat-textarea {
  flex: 1; min-height: 72px; resize: vertical;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 10px 12px; font: inherit; font-size: 0.875rem;
  background: var(--surface-2); color: var(--text); transition: border-color .15s;
}
.chat-textarea:focus { outline: none; border-color: var(--accent); }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.cursor-blink { display: inline-block; margin-left:1px; animation: blink .8s step-start infinite; }
.cursor-block { display:inline-block; width:.5em; height:.9em; background:currentColor; animation:blink 1s step-end infinite; vertical-align:middle; }

.pill-row {
  display: flex; background: var(--surface-2);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 3px; gap: 2px; width: fit-content;
}
.pill {
  background: none; border: none; border-radius: 6px;
  padding: 4px 11px; font: 500 0.78rem 'Inter', sans-serif;
  color: var(--muted); cursor: pointer; transition: all .15s;
}
.pill.active { background: var(--surface); color: var(--text); box-shadow: var(--shadow-sm); }
.pill:disabled { opacity:.5; cursor:not-allowed; }
.pill-hint { font-size: 0.75rem; color: var(--muted); font-style: italic; margin-top: 5px; }

/* ── Exam ── */
.paper-tabs {
  display: flex; gap: 8px; border-bottom: 1px solid var(--border);
  padding-bottom: 0; margin-bottom: 20px;
}
.paper-tab {
  border: none; border-bottom: 2px solid transparent;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  padding: 8px 18px; font: 600 0.875rem 'Inter', sans-serif;
  cursor: pointer; background: none; color: var(--muted);
  transition: all .15s; margin-bottom: -1px;
}
.paper-tab:hover:not(.active) { color: var(--text); }
.paper-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.paper-tab:disabled { opacity: .4; cursor: not-allowed; }

.passage-box {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 18px; margin-bottom: 14px;
  max-height: 260px; overflow-y: auto; white-space: pre-wrap;
  font-size: 0.875rem; line-height: 1.65; color: var(--text-2);
}
.question-box {
  background: var(--warning-bg); border: 1px solid var(--warning-border);
  border-radius: var(--radius); padding: 14px 16px; margin-bottom: 14px;
  font-size: 1rem; line-height: 1.55; color: var(--text); white-space: pre-wrap;
}
.exam-textarea {
  width: 100%; min-height: 200px; resize: vertical;
  border: 1px solid var(--border); border-radius: var(--radius);
  padding: 12px; font: inherit; font-size: 0.875rem;
  background: var(--surface-2); color: var(--text); margin-bottom: 12px; transition: border-color .15s;
}
.exam-textarea:focus { outline: none; border-color: var(--accent); }
.exam-textarea:disabled { opacity: .6; }

.exam-results {
  background: var(--success-bg); border: 1px solid var(--success-border);
  border-radius: var(--radius); padding: 20px; margin-top: 16px;
}
.score-heading { font-size: 1.4rem; font-weight: 800; color: var(--success); margin-bottom: 14px; }

.criterion-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 14px 16px; margin-bottom: 10px;
}
.criterion-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.criterion-name { font-weight: 600; font-size: 0.875rem; color: var(--text); }
.criterion-badge {
  background: var(--accent); color: #fff; border-radius: 20px;
  padding: 2px 12px; font: 700 0.72rem 'Inter', sans-serif;
}
.criterion-note { font-size: 0.875rem; color: var(--text-2); line-height: 1.5; margin-bottom: 6px; }
.criterion-label { font-size: 0.72rem; font-variant: small-caps; color: var(--muted); margin-right: 6px; }

.overall-box {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 14px 16px; margin-top: 14px;
}
.overall-title { font-size: 0.875rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.overall-text { font-size: 0.875rem; color: var(--text-2); line-height: 1.5; }

.tags-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.tag {
  display: inline-flex; align-items: center;
  background: var(--accent-soft); color: var(--accent);
  border: 1px solid var(--accent-border); border-radius: 20px;
  padding: 3px 12px; font: 600 0.75rem 'Inter', sans-serif;
}

.radio-group { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.radio-label { display: flex; align-items: center; gap: 5px; font-size: 0.875rem; cursor: pointer; color: var(--text-2); }

.context-badge { font-size: 0.75rem; color: var(--muted); margin-bottom: 12px; }

/* Doc picker grid (exam P2) */
.doc-pick-grid-exam { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 10px; margin: 12px 0; }

/* ── Debug ── */
.dbg-details {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); overflow: hidden; margin-top: 14px;
}
.dbg-details > summary {
  padding: 8px 14px; cursor: pointer; list-style: none;
  font: 700 0.72rem 'Inter', sans-serif; text-transform: uppercase;
  letter-spacing: .05em; color: var(--muted);
  display: flex; align-items: center; gap: 6px; user-select: none;
}
.dbg-details > summary::-webkit-details-marker { display: none; }
.dbg-details[open] > summary { border-bottom: 1px solid var(--border); color: var(--text-2); }
.dbg-body { padding: 14px; }
.dbg-pre {
  background: #0f172a; color: #e2e8f0;
  border-radius: var(--radius-sm); padding: 12px;
  font-size: 0.76rem; line-height: 1.45; white-space: pre-wrap;
  max-height: 320px; overflow: auto;
}
.dbg-line { font-size: 0.78rem; color: var(--text-2); margin-bottom: 4px; }
.dbg-section { font-size: 0.875rem; font-weight: 600; color: var(--text); margin: 14px 0 8px; }
.dbg-excerpt {
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--surface); padding: 10px 12px; margin-bottom: 8px;
}
.dbg-excerpt-meta { font-size: 0.75rem; color: var(--text-2); margin-bottom: 6px; }
.dbg-excerpt-text { font-size: 0.75rem; color: var(--text-2); white-space: pre-wrap; line-height: 1.45; }

/* ── Router Debug Panel ── */
.rdbg-details {
  background: #f8f9ff; border: 1px solid #c7d2fe;
  border-radius: var(--radius-sm); overflow: hidden; margin-top: 10px;
}
.rdbg-details > summary {
  padding: 7px 12px; cursor: pointer; list-style: none;
  font: 700 0.70rem 'Inter', sans-serif; text-transform: uppercase;
  letter-spacing: .05em; color: #6366f1;
  display: flex; align-items: center; gap: 6px; user-select: none;
}
.rdbg-details > summary::-webkit-details-marker { display: none; }
.rdbg-details[open] > summary { border-bottom: 1px solid #c7d2fe; }
.rdbg-body { padding: 12px 14px; font-family: 'Inter', monospace; font-size: 0.76rem; color: var(--text-2); }
.rdbg-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.rdbg-label { width: 170px; flex-shrink: 0; color: var(--muted); font-size: 0.73rem; }
.rdbg-bar-wrap { flex: 1; background: #e0e7ff; border-radius: 99px; height: 8px; overflow: hidden; }
.rdbg-bar { height: 100%; border-radius: 99px; background: #6366f1; transition: width .3s; }
.rdbg-val { width: 38px; text-align: right; font-variant-numeric: tabular-nums; font-size: 0.73rem; }
.rdbg-mode-badge {
  display: inline-block; padding: 1px 8px; border-radius: 99px; font-size: 0.72rem; font-weight: 700;
  letter-spacing: .03em; text-transform: uppercase;
}
.rdbg-mode-rag { background: #dcfce7; color: #15803d; }
.rdbg-mode-bk  { background: #fef9c3; color: #854d0e; }
.rdbg-matches { margin-top: 10px; }
.rdbg-match { display: flex; gap: 8px; align-items: baseline; margin-bottom: 4px; font-size: 0.73rem; }
.rdbg-match-sim { width: 40px; flex-shrink: 0; color: #6366f1; font-variant-numeric: tabular-nums; }
.rdbg-match-score { width: 34px; flex-shrink: 0; color: var(--muted); }
.rdbg-match-q { color: var(--text-2); }
.rdbg-cat { font-size: 0.66rem; color: var(--muted); margin-left: 4px; }
.rdbg-threshold-line { position: relative; height: 100%; }
.rdbg-threshold-mark {
  position: absolute; top: -2px; height: 12px; width: 2px;
  background: #f59e0b; border-radius: 1px;
}

/* ── Misc ── */
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.spacer { margin-top: 14px; }

/* ── Sessions sidebar ── */
.learn-with-sessions { display: grid; grid-template-columns: 210px 1fr; gap: 16px; align-items: start; }
.sessions-sidebar {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px;
  position: sticky; top: 80px;
}
.sessions-sidebar-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 10px; font-size: 0.75rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
}
.sessions-list { display: flex; flex-direction: column; gap: 2px; }
.session-item {
  display: flex; align-items: center; gap: 5px;
  padding: 6px 8px; border-radius: var(--radius-sm);
  cursor: pointer; font-size: 0.82rem; color: var(--text-2);
  transition: background 0.1s;
}
.session-item:hover { background: var(--surface-2); }
.session-item.active { background: var(--accent); color: #fff; }
.session-item-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-turn-badge {
  font-size: 0.7rem; background: var(--surface-3);
  border-radius: 10px; padding: 1px 6px; flex-shrink: 0;
}
.session-item.active .session-turn-badge { background: rgba(255,255,255,.25); }
@keyframes session-pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
.session-loading-dot { font-size:0.55rem; color:var(--accent); animation:session-pulse 1s ease-in-out infinite; flex-shrink:0; margin-right:2px; }
.session-actions { display: flex; gap: 2px; flex-shrink: 0; }
.session-icon-btn {
  background: none; border: none; cursor: pointer; padding: 2px 4px;
  font-size: 0.75rem; color: inherit; opacity: 0.6; border-radius: 3px;
}
.session-icon-btn:hover { opacity: 1; background: rgba(0,0,0,0.08); }
.session-item.active .session-icon-btn:hover { background: rgba(255,255,255,0.2); }
.session-rename-row { display: flex; gap: 4px; padding: 4px 0; }
.session-rename-input {
  flex: 1; font-size: 0.82rem; padding: 3px 6px;
  border: 1px solid var(--accent); border-radius: var(--radius-sm);
  background: var(--surface); color: var(--text);
}
.msg-warning {
  background: #fef9c3; color: #78350f;
  border: 1px solid #fde68a; border-radius: var(--radius-sm);
  padding: 8px 12px; font-size: 0.84rem;
}
[data-theme="dark"] .msg-warning { background: #422006; color: #fde68a; border-color: #78350f; }
.msg-info {
  background: var(--accent-soft); color: var(--accent);
  border: 1px solid var(--accent-border); border-radius: var(--radius-sm);
  padding: 10px 14px; font-size: 0.875rem;
}
[data-theme="dark"] .msg-info { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-border); }

@media (max-width: 760px) {
  .learn-with-sessions { grid-template-columns: 1fr; }
  .sessions-sidebar { position: static; }
}

@media (max-width: 640px) {
  .page { padding: 20px 16px; }
  .app-header { padding: 0 14px; }
  .chat-form { flex-direction: column; align-items: stretch; }
  .theme-toggle { display: none; }
  .model-wrap { display: none; }
  .header-title { font-size: 0.95rem; }
}
`;

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  // ── User ID (persistent, browser-local) ───────────────────────────────────
  const [userId] = useState(() => {
    let id = localStorage.getItem("ib_user_id");
    if (!id) { id = crypto.randomUUID(); localStorage.setItem("ib_user_id", id); }
    return id;
  });

  // ── Navigation & Theme ────────────────────────────────────────────────────
  const [activePage, setActivePage] = useState("dashboard");
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "auto");

  useEffect(() => {
    const html = document.documentElement;
    if (theme === "auto") {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      html.setAttribute("data-theme", mq.matches ? "dark" : "light");
      const handler = (e) => html.setAttribute("data-theme", e.matches ? "dark" : "light");
      mq.addEventListener("change", handler);
      return () => mq.removeEventListener("change", handler);
    }
    html.setAttribute("data-theme", theme);
  }, [theme]);

  const setThemePref = (t) => { localStorage.setItem("theme", t); setTheme(t); };

  // ── Upload / PDF State ────────────────────────────────────────────────────
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle|processing|chunking|success|error
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [uploadError, setUploadError] = useState("");
  const [result, setResult] = useState(null);
  const [uploadQualityDebug, setUploadQualityDebug] = useState(null);

  // ── Docs & LLM ───────────────────────────────────────────────────────────
  const [availableDocs, setAvailableDocs] = useState([]);
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [selectedLearnDocId, setSelectedLearnDocId] = useState(null);

  // ── System & Models ───────────────────────────────────────────────────────
  const [systemInfo, setSystemInfo] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [modelError, setModelError] = useState("");
  const [sessionModels, setSessionModels] = useState(() => new Map());

  // ── Chat State (per-session maps) ─────────────────────────────────────────
  // Keys are session_id strings or "__nosession__" for no-session chats.
  const [sessionChats, setSessionChats] = useState(() => new Map());
  const [sessionStatuses, setSessionStatuses] = useState(() => new Map());
  const [sessionErrors, setSessionErrors] = useState(() => new Map());
  const [sessionDebugs, setSessionDebugs] = useState(() => new Map());
  const [sessionRoutingDebugs, setSessionRoutingDebugs] = useState(() => new Map());
  const [chatInput, setChatInput] = useState("");
  const [promptFormat, setPromptFormat] = useState(null);

  // ── Session State ─────────────────────────────────────────────────────────
  const [learnSessions, setLearnSessions] = useState([]);
  const [activeLearnSessionId, setActiveLearnSessionId] = useState(null);
  const [renamingSessionId, setRenamingSessionId] = useState(null);
  const [renameValue, setRenameValue] = useState("");

  // ── Exam Session State ────────────────────────────────────────────────────
  const [examSessions, setExamSessions] = useState([]);
  const [activeExamSessionId, setActiveExamSessionId] = useState(null);
  const [examSessionData, setExamSessionData] = useState(() => new Map());

  // ── Exam State ────────────────────────────────────────────────────────────
  const [examPaperType, setExamPaperType] = useState("paper1");
  const [examStep, setExamStep] = useState("setup");
  const [examPassage, setExamPassage] = useState("");
  const [examQuestionText, setExamQuestionText] = useState("");
  const [examAllDocs, setExamAllDocs] = useState([]);
  const [examDocsStatus, setExamDocsStatus] = useState("idle");
  const [examDocsError, setExamDocsError] = useState("");
  const [examSelectedDocIds, setExamSelectedDocIds] = useState([]);
  const [examAnswer, setExamAnswer] = useState("");
  const [examP2Questions, setExamP2Questions] = useState([]);
  const [examContextMode, setExamContextMode] = useState("chunks");
  const [examGrading, setExamGrading] = useState(null);
  const [examGradingStatus, setExamGradingStatus] = useState("idle");
  const [examGradingError, setExamGradingError] = useState("");
  const [examDebug, setExamDebug] = useState(null);
  const [examFeedbacks, setExamFeedbacks] = useState({});
  const [examFeedbackStatus, setExamFeedbackStatus] = useState("idle");
  const [examTokenDebug, setExamTokenDebug] = useState(null);

  // ── Delete state ──────────────────────────────────────────────────────────
  const [deleteDocId, setDeleteDocId] = useState(null);
  const [deleteError, setDeleteError] = useState("");

  // ── Exam grading timer & abort ────────────────────────────────────────────
  const [examGradingElapsed, setExamGradingElapsed] = useState(0);
  const examGradingIntervalRef = useRef(null);
  const examGradingAbortRef = useRef(null);
  const examFeedbackAbortRef = useRef(null);

  // ── Refs ──────────────────────────────────────────────────────────────────
  const startedAtRef = useRef(null);
  const intervalRef = useRef(null);
  const downloadUrlRef = useRef(null);
  const sessionCreatingRef = useRef(false);
  const sessionAbortRefs = useRef(new Map());

  // ── Derived ───────────────────────────────────────────────────────────────
  const isBusy = uploadStatus === "processing" || uploadStatus === "chunking";
  const isProcessing = uploadStatus === "processing";
  const isChunking = uploadStatus === "chunking";
  const activeChatDocId = result?.documentId || selectedLearnDocId;
  const hasChatDocument = Boolean(activeChatDocId);
  const chatAvailable = llmAvailable && Boolean(activeChatDocId);
  const chunksAvailable = Boolean(result?.chunksAvailable);
  const modesAvailable = availableDocs.length > 0 || Boolean(result?.documentId);

  // Per-session chat accessors
  const _sessionKey = activeLearnSessionId ?? "__nosession__";
  const chatMessages = sessionChats.get(_sessionKey) ?? [];
  const chatStatus = sessionStatuses.get(_sessionKey) ?? "idle";
  const chatError = sessionErrors.get(_sessionKey) ?? "";
  const chatDebug = sessionDebugs.get(_sessionKey) ?? null;
  const chatRoutingDebug = sessionRoutingDebugs.get(_sessionKey) ?? null;
  const isChatSending = chatStatus === "sending";

  // ── On Mount ──────────────────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE_URL}/api/documents`).then((r) => (r.ok ? r.json() : [])),
      fetch(`${API_BASE_URL}/api/status`).then((r) => (r.ok ? r.json() : {})),
      fetch(`${API_BASE_URL}/api/system-info`).then((r) => (r.ok ? r.json() : null)),
      fetch(`${API_BASE_URL}/api/models`).then((r) => (r.ok ? r.json() : {})),
    ])
      .then(([docs, statusData, sysInfo, modelsData]) => {
        setAvailableDocs((docs || []).filter((d) => d.chunks_available));
        setLlmAvailable(Boolean(statusData?.chat_available));
        if (sysInfo) setSystemInfo(sysInfo);
        setAvailableModels(modelsData?.models || []);
        setActiveModel(modelsData?.active || null);
      })
      .catch(() => {});
  }, []);

  // ── Upload timer ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isBusy) return undefined;
    intervalRef.current = window.setInterval(() => {
      if (!startedAtRef.current) return;
      setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 200);
    return () => { if (intervalRef.current) { window.clearInterval(intervalRef.current); intervalRef.current = null; } };
  }, [isBusy]);

  useEffect(() => {
    return () => { if (downloadUrlRef.current) { URL.revokeObjectURL(downloadUrlRef.current); downloadUrlRef.current = null; } };
  }, []);

  // Auto-load (or create) a session when navigating to Learn with a doc pre-selected
  // (e.g. via the "Learn" button in the Library page).
  useEffect(() => {
    if (activePage === "learn" && selectedLearnDocId && !activeLearnSessionId && learnSessions.length === 0) {
      if (sessionCreatingRef.current) return;
      sessionCreatingRef.current = true;
      _loadOrCreateSession(selectedLearnDocId).finally(() => { sessionCreatingRef.current = false; });
    }
  }, [activePage, selectedLearnDocId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Reset helpers ──────────────────────────────────────────────────────────
  const resetExam = () => {
    if (examGradingIntervalRef.current) { clearInterval(examGradingIntervalRef.current); examGradingIntervalRef.current = null; }
    examGradingAbortRef.current?.abort();
    examFeedbackAbortRef.current?.abort();
    setExamStep("setup");
    setExamPassage("");
    setExamQuestionText("");
    setExamAllDocs([]);
    setExamDocsStatus("idle");
    setExamDocsError("");
    setExamSelectedDocIds([]);
    setExamP2Questions([]);
    setExamAnswer("");
    setExamContextMode("chunks");
    setExamGrading(null);
    setExamGradingStatus("idle");
    setExamGradingError("");
    setExamGradingElapsed(0);
    setExamDebug(null);
    setExamFeedbacks({});
    setExamFeedbackStatus("idle");
    setExamTokenDebug(null);
  };

  const resetResult = () => {
    if (downloadUrlRef.current) { URL.revokeObjectURL(downloadUrlRef.current); downloadUrlRef.current = null; }
    setResult(null);
    setUploadQualityDebug(null);
    setSelectedLearnDocId(null);
    resetExam();
  };

  const resetChat = (keyOverride) => {
    const key = keyOverride ?? (_sessionKey);
    setChatInput("");
    setSessionChats((prev) => { const m = new Map(prev); m.delete(key); return m; });
    setSessionStatuses((prev) => { const m = new Map(prev); m.delete(key); return m; });
    setSessionErrors((prev) => { const m = new Map(prev); m.delete(key); return m; });
    setSessionDebugs((prev) => { const m = new Map(prev); m.delete(key); return m; });
  };

  const CHAT_MAX_MESSAGES = 10;

  // ── Session helpers ────────────────────────────────────────────────────────
  const loadLearnSessions = async (documentId) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions?user_id=${encodeURIComponent(userId)}&document_id=${encodeURIComponent(documentId)}&mode=learn`);
      if (!res.ok) return [];
      const sessions = await res.json();
      setLearnSessions(sessions);
      return sessions;
    } catch { return []; }
  };

  const createLearnSession = async (documentId) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, document_id: documentId, mode: "learn" }),
      });
      if (!res.ok) return null;
      const session = await res.json();
      setLearnSessions((prev) => [session, ...prev]);
      setActiveLearnSessionId(session.session_id);
      return session;
    } catch { return null; }
  };

  const loadSessionHistory = async (sessionId) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/turns`);
      if (!res.ok) return;
      const turns = await res.json();
      const msgs = [];
      for (const t of turns) {
        msgs.push({ role: "user", content: t.user_query });
        msgs.push({ role: "assistant", content: t.assistant_answer });
      }
      // Only overwrite if the session has no in-progress or stored messages
      setSessionChats((prev) => {
        if ((prev.get(sessionId) ?? []).length > 0) return prev;
        const m = new Map(prev); m.set(sessionId, msgs); return m;
      });
      setSessionErrors((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      setSessionDebugs((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
    } catch { /* non-critical */ }
  };

  const _loadOrCreateSession = async (docId) => {
    const sessions = await loadLearnSessions(docId);
    if (sessions.length > 0) {
      setActiveLearnSessionId(sessions[0].session_id);
      await loadSessionHistory(sessions[0].session_id);
    } else {
      await createLearnSession(docId);
    }
  };

  const onSelectLearnSession = async (sessionId) => {
    setActiveLearnSessionId(sessionId);
    setRenamingSessionId(null);
    // Don't wipe state for other sessions — just switch the active view.
    // Load history only if we have nothing stored for this session yet.
    await loadSessionHistory(sessionId);
  };

  const onRenameSession = async (sessionId) => {
    const name = renameValue.trim();
    if (!name) { setRenamingSessionId(null); return; }
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) return;
      setLearnSessions((prev) => prev.map((s) => s.session_id === sessionId ? { ...s, name } : s));
    } catch { /* non-critical */ }
    setRenamingSessionId(null);
  };

  const onDeleteSession = async (sessionId) => {
    if (!window.confirm("Delete this session and all its messages?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      if (!res.ok) return;
      setLearnSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
      // Clean up per-session state maps
      setSessionChats((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      setSessionStatuses((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      setSessionErrors((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      setSessionDebugs((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      setSessionModels((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      if (activeLearnSessionId === sessionId) {
        setActiveLearnSessionId(null);
      }
    } catch { /* non-critical */ }
  };

  // ── Exam Sessions ──────────────────────────────────────────────────────────
  const loadExamSessions = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions?user_id=${encodeURIComponent(userId)}&document_id=__exam__&mode=exam`);
      if (!res.ok) return [];
      const sessions = await res.json();
      setExamSessions(sessions);
      return sessions;
    } catch { return []; }
  };

  const createExamSession = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, document_id: "__exam__", mode: "exam", name: "Exam Session" }),
      });
      if (!res.ok) return null;
      const session = await res.json();
      setExamSessions((prev) => [session, ...prev]);
      setActiveExamSessionId(session.session_id);
      return session;
    } catch { return null; }
  };

  const saveExamState = (sessionId) => {
    if (!sessionId) return;
    const snapshot = {
      paperType: examPaperType, step: examStep, passage: examPassage,
      questionText: examQuestionText, selectedDocIds: examSelectedDocIds,
      answer: examAnswer, contextMode: examContextMode, grading: examGrading,
      gradingStatus: examGradingStatus, debug: examDebug,
      feedbacks: examFeedbacks, feedbackStatus: examFeedbackStatus,
      tokenDebug: examTokenDebug,
    };
    setExamSessionData((prev) => { const m = new Map(prev); m.set(sessionId, snapshot); return m; });
    fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/metadata`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata: snapshot }),
    }).catch(() => {});
  };

  const _restoreExamSnapshot = (md) => {
    if (!md) return;
    setExamPaperType(md.paperType || "paper1");
    setExamStep(md.step || "setup");
    setExamPassage(md.passage || "");
    setExamQuestionText(md.questionText || "");
    setExamSelectedDocIds(md.selectedDocIds || []);
    setExamAnswer(md.answer || "");
    setExamContextMode(md.contextMode || "chunks");
    setExamGrading(md.grading || null);
    setExamGradingStatus(md.gradingStatus || "idle");
    setExamDebug(md.debug || null);
    setExamFeedbacks(md.feedbacks || {});
    setExamFeedbackStatus(md.feedbackStatus || "idle");
    setExamTokenDebug(md.tokenDebug || null);
  };

  const onSelectExamSession = (sessionId) => {
    if (activeExamSessionId) saveExamState(activeExamSessionId);
    setActiveExamSessionId(sessionId);
    const stored = examSessionData.get(sessionId);
    if (stored) {
      _restoreExamSnapshot(stored);
    } else {
      resetExam();
      fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`)
        .then((r) => r.ok ? r.json() : null)
        .then((sess) => {
          if (!sess?.metadata) return;
          let md;
          try { md = typeof sess.metadata === "string" ? JSON.parse(sess.metadata) : sess.metadata; }
          catch { return; }
          setExamSessionData((prev) => { const m = new Map(prev); m.set(sessionId, md); return m; });
          _restoreExamSnapshot(md);
        })
        .catch(() => {});
    }
  };

  const onRenameExamSession = async (sessionId) => {
    const name = renameValue.trim();
    if (!name) { setRenamingSessionId(null); return; }
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) return;
      setExamSessions((prev) => prev.map((s) => s.session_id === sessionId ? { ...s, name } : s));
    } catch { /* non-critical */ }
    finally { setRenamingSessionId(null); }
  };

  const onDeleteExamSession = async (sessionId) => {
    if (!window.confirm("Delete this exam session?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      if (!res.ok) return;
      setExamSessions((prev) => {
        const next = prev.filter((s) => s.session_id !== sessionId);
        return next;
      });
      setExamSessionData((prev) => { const m = new Map(prev); m.delete(sessionId); return m; });
      if (activeExamSessionId === sessionId) {
        setActiveExamSessionId(null);
        resetExam();
        // Create a fresh session automatically
        createExamSession();
      }
    } catch { /* non-critical */ }
  };

  // ── Model selection ────────────────────────────────────────────────────────
  const onSelectModel = async (filename) => {
    if (modelSwitching || filename === activeModel) return;
    setModelSwitching(true);
    setModelError("");
    try {
      const res = await fetch(`${API_BASE_URL}/api/models/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_filename: filename }),
      });
      if (!res.ok) { setModelError(await parseErrorDetail(res, "Failed to switch model.")); return; }
      const data = await res.json();
      setActiveModel(data.active);
      setAvailableModels((prev) => prev.map((m) => ({ ...m, active: m.filename === data.active })));
    } catch { setModelError("Network error switching model."); }
    finally { setModelSwitching(false); }
  };

  // ── Upload PDF ─────────────────────────────────────────────────────────────
  const onFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setUploadError("");
    resetResult();
    resetChat();
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setUploadError("Please choose a PDF file first."); return; }
    if (!file.name.toLowerCase().endsWith(".pdf")) { setUploadError("File must end with .pdf"); return; }

    setUploadError("");
    setUploadStatus("processing");
    setElapsedSeconds(0);
    startedAtRef.current = Date.now();
    resetResult();
    resetChat();

    const formData = new FormData();
    formData.append("file", file);
    const fallbackName = `${getFileStem(file.name)}.txt`;

    try {
      const response = await fetch(`${API_BASE_URL}/api/process-pdf`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await parseErrorDetail(response, "Failed to process PDF."));

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      downloadUrlRef.current = downloadUrl;

      const documentId = response.headers.get("X-Document-Id");
      const chatAvailableHeader = response.headers.get("X-Chat-Available");
      setResult({
        mode: response.headers.get("X-Processing-Mode"),
        pages: response.headers.get("X-Pages"),
        chars: response.headers.get("X-Text-Chars"),
        chunksCount: response.headers.get("X-Chunks-Count"),
        chunksAvailable: response.headers.get("X-Chunks-Available") === "1",
        chunksDownloadUrl: response.headers.get("X-Chunks-Download-Url") ? toApiUrl(response.headers.get("X-Chunks-Download-Url")) : "",
        chunkSchemaVersion: response.headers.get("X-Chunk-Schema-Version") || "",
        filename: parseDownloadFilename(response.headers.get("Content-Disposition"), fallbackName),
        downloadUrl,
        documentId,
        documentFilename: response.headers.get("X-Document-Filename"),
        chatAvailable: chatAvailableHeader === "1",
      });
      if (chatAvailableHeader === "1") setLlmAvailable(true);
      const qd = response.headers.get("X-Quality-Decision");
      if (qd) {
        const scoreNewRaw = response.headers.get("X-Quality-Score-New");
        const scoreExistingRaw = response.headers.get("X-Quality-Score-Existing");
        setUploadQualityDebug({
          decision: qd,
          method: response.headers.get("X-Quality-Method") || null,
          scoreNew: scoreNewRaw != null ? parseFloat(scoreNewRaw) : null,
          scoreExisting: scoreExistingRaw != null ? parseFloat(scoreExistingRaw) : null,
        });
      } else {
        setUploadQualityDebug(null);
      }
      fetch(`${API_BASE_URL}/api/documents`)
        .then((r) => (r.ok ? r.json() : []))
        .then((docs) => setAvailableDocs((docs || []).filter((d) => d.chunks_available)))
        .catch(() => {});
      setUploadStatus("success");
    } catch (err) {
      setUploadStatus("error");
      setUploadError(err?.message || "Unexpected error while processing PDF.");
    } finally {
      if (intervalRef.current) { window.clearInterval(intervalRef.current); intervalRef.current = null; }
      if (startedAtRef.current) setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
      startedAtRef.current = null;
    }
  };

  const onGenerateChunks = async () => {
    if (!result?.documentId || isBusy) return;
    setUploadError("");
    setUploadStatus("chunking");
    setElapsedSeconds(0);
    startedAtRef.current = Date.now();
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(result.documentId)}/generate-chunks`, { method: "POST" });
      if (!response.ok) throw new Error(await parseErrorDetail(response, "Failed to generate chunks."));
      const data = await response.json();
      setResult((prev) => prev ? {
        ...prev,
        chunksAvailable: Boolean(data?.chunks_available),
        chunksCount: data?.chunks_count ?? null,
        chunkSchemaVersion: data?.chunk_schema_version || "",
        chunksDownloadUrl: data?.chunks_download_url ? toApiUrl(data.chunks_download_url) : "",
      } : prev);
      setUploadStatus("success");
    } catch (err) {
      setUploadStatus("error");
      setUploadError(err?.message || "Unexpected error while generating chunks.");
    } finally {
      if (intervalRef.current) { window.clearInterval(intervalRef.current); intervalRef.current = null; }
      if (startedAtRef.current) setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
      startedAtRef.current = null;
    }
  };

  // ── Delete document ────────────────────────────────────────────────────────
  const onDeleteDocument = async (docId) => {
    if (!window.confirm("Delete this document? This cannot be undone.")) return;
    setDeleteDocId(docId);
    setDeleteError("");
    try {
      const res = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
      if (!res.ok) { setDeleteError(await parseErrorDetail(res, "Failed to delete.")); return; }
      setAvailableDocs((prev) => prev.filter((d) => d.document_id !== docId));
      if (selectedLearnDocId === docId) setSelectedLearnDocId(null);
      if (result?.documentId === docId) resetResult();
    } catch (err) { setDeleteError(err?.message || "Delete failed."); }
    finally { setDeleteDocId(null); }
  };

  // ── Chat ───────────────────────────────────────────────────────────────────
  // Core streaming function. `userMessages` is the full message list ending with the user turn.
  // The caller is responsible for setting up the placeholder assistant message before calling.
  const _doStreamChat = async (sendKey, sendSessionId, userMessages) => {
    const controller = new AbortController();
    sessionAbortRefs.current.set(sendKey, controller);
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          document_id: activeChatDocId,
          messages: userMessages,
          prompt_format: promptFormat,
          session_id: sendSessionId,
          user_id: userId,
          model: sessionModels.get(sendKey) || null,
        }),
      });
      if (!response.ok) {
        setSessionChats((prev) => { const m = new Map(prev); m.set(sendKey, userMessages); return m; });
        throw new Error(await parseErrorDetail(response, "Failed to get chat response."));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          if (ev.type === "routing_debug") {
            setSessionRoutingDebugs((prev) => { const m = new Map(prev); m.set(sendKey, ev); return m; });
          } else if (ev.type === "token") {
            setSessionChats((prev) => {
              const m = new Map(prev);
              const msgs = [...(m.get(sendKey) ?? [])];
              const last = msgs[msgs.length - 1];
              msgs[msgs.length - 1] = { ...last, content: last.content + ev.text };
              m.set(sendKey, msgs);
              return m;
            });
          } else if (ev.type === "done") {
            setSessionDebugs((prev) => { const m = new Map(prev); m.set(sendKey, ev.debug ? { ...ev.debug, prompt_mode: ev.prompt_mode } : null); return m; });
            setSessionChats((prev) => {
              const m = new Map(prev);
              const msgs = [...(m.get(sendKey) ?? [])];
              const last = msgs[msgs.length - 1];
              msgs[msgs.length - 1] = { ...last, streaming: false, totalSeconds: Number(ev.debug?.total_seconds) || null };
              m.set(sendKey, msgs);
              return m;
            });
            if (sendSessionId) {
              setLearnSessions((prev) => prev.map((s) => s.session_id === sendSessionId ? { ...s, turn_count: (s.turn_count || 0) + 1 } : s));
            }
          } else if (ev.type === "error") {
            setSessionChats((prev) => { const m = new Map(prev); m.set(sendKey, userMessages); return m; });
            throw new Error(ev.detail || "Chat stream error.");
          }
        }
      }
      setSessionStatuses((prev) => { const m = new Map(prev); m.set(sendKey, "idle"); return m; });
    } catch (err) {
      if (err?.name === "AbortError") {
        // Mark the partial assistant message as stopped instead of removing it
        setSessionChats((prev) => {
          const m = new Map(prev);
          const msgs = [...(m.get(sendKey) ?? [])];
          if (msgs.length > 0 && msgs[msgs.length - 1].role === "assistant") {
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], streaming: false, stopped: true };
          }
          m.set(sendKey, msgs);
          return m;
        });
        setSessionStatuses((prev) => { const m = new Map(prev); m.set(sendKey, "idle"); return m; });
      } else {
        setSessionStatuses((prev) => { const m = new Map(prev); m.set(sendKey, "error"); return m; });
        setSessionErrors((prev) => { const m = new Map(prev); m.set(sendKey, err?.message || "Unexpected error while chatting."); return m; });
      }
    } finally {
      sessionAbortRefs.current.delete(sendKey);
    }
  };

  const onSendChat = async (e) => {
    e.preventDefault();
    if (!hasChatDocument || !chatAvailable) return;
    const userText = chatInput.trim();
    if (!userText) return;

    // Capture the key at send-time so closures inside the async function always
    // update the correct session even if the user navigates elsewhere.
    const sendKey = activeLearnSessionId ?? "__nosession__";
    const sendSessionId = activeLearnSessionId;

    const prevMsgs = sessionChats.get(sendKey) ?? [];
    const nextMessages = [...prevMsgs, { role: "user", content: userText }];
    setSessionChats((prev) => { const m = new Map(prev); m.set(sendKey, [...nextMessages, { role: "assistant", content: "", streaming: true }]); return m; });
    setChatInput("");
    setSessionErrors((prev) => { const m = new Map(prev); m.delete(sendKey); return m; });
    setSessionDebugs((prev) => { const m = new Map(prev); m.delete(sendKey); return m; });
    setSessionStatuses((prev) => { const m = new Map(prev); m.set(sendKey, "sending"); return m; });

    await _doStreamChat(sendKey, sendSessionId, nextMessages);
  };

  const regenerateChat = async (key) => {
    const sessionId = key === "__nosession__" ? null : key;
    const msgs = sessionChats.get(key) ?? [];
    const trimmed = [...msgs];
    // Remove last assistant message (the stopped one)
    if (trimmed.length > 0 && trimmed[trimmed.length - 1].role === "assistant") trimmed.pop();
    const lastUser = trimmed[trimmed.length - 1];
    if (!lastUser || lastUser.role !== "user") return;
    setSessionChats((prev) => { const m = new Map(prev); m.set(key, [...trimmed, { role: "assistant", content: "", streaming: true }]); return m; });
    setSessionErrors((prev) => { const m = new Map(prev); m.delete(key); return m; });
    setSessionDebugs((prev) => { const m = new Map(prev); m.delete(key); return m; });
    setSessionStatuses((prev) => { const m = new Map(prev); m.set(key, "sending"); return m; });
    await _doStreamChat(key, sessionId, trimmed);
  };

  // ── Exam ───────────────────────────────────────────────────────────────────
  const onEnterExam = async (paperType) => {
    setExamAnswer("");
    setExamGrading(null);
    setExamGradingStatus("idle");
    setExamGradingError("");
    if (paperType === "paper1") {
      setExamDocsStatus("loading");
      try {
        const res = await fetch(`${API_BASE_URL}/api/exam/paper1`);
        if (!res.ok) throw new Error(await parseErrorDetail(res, "Failed to load Paper 1."));
        const data = await res.json();
        setExamPassage(data.passage || "");
        setExamQuestionText(data.question || "");
        setExamStep("writing");
        setExamDocsStatus("ready");
        setTimeout(() => saveExamState(activeExamSessionId), 0);
      } catch (err) { setExamDocsStatus("error"); setExamDocsError(err?.message || "Failed to load Paper 1."); }
    } else {
      setExamDocsStatus("loading");
      setExamSelectedDocIds([]);
      setExamStep("setup");
      try {
        const [docsRes, qRes] = await Promise.all([fetch(`${API_BASE_URL}/api/documents`), fetch(`${API_BASE_URL}/api/exam/paper2/questions`)]);
        if (!docsRes.ok) throw new Error(await parseErrorDetail(docsRes, "Failed to load documents."));
        if (!qRes.ok) throw new Error(await parseErrorDetail(qRes, "Failed to load questions."));
        setExamAllDocs((await docsRes.json() || []).filter((d) => d.chunks_available));
        setExamP2Questions((await qRes.json()).questions || []);
        setExamDocsStatus("ready");
      } catch (err) { setExamDocsStatus("error"); setExamDocsError(err?.message || "Failed to load data."); }
    }
  };

  const onToggleDocSelection = (docId) => {
    setExamSelectedDocIds((prev) => {
      if (prev.includes(docId)) return prev.filter((id) => id !== docId);
      if (prev.length >= 2) return prev;
      return [...prev, docId];
    });
  };

  const onConfirmDocSelection = () => {
    if (examSelectedDocIds.length !== 2 || examP2Questions.length === 0) return;
    const picked = examP2Questions[Math.floor(Math.random() * examP2Questions.length)];
    setExamQuestionText(picked.text || "");
    setExamStep("writing");
    setTimeout(() => saveExamState(activeExamSessionId), 0);
  };

  // Generate detailed coaching feedback for a single criterion.
  // `gradingCriteria` is the array of criterion objects from the grading result.
  // `studentAnswer`, `questionText`, `paperType`, `passage`, `selectedDocIds`, `contextMode`, `modelKey`
  // are captured from state at call time.
  const generateCriterionFeedback = async (criterion, gradingCriteria, studentAnswer, questionText, paperType, passage, selectedDocIds, contextMode, modelKey) => {
    const c = gradingCriteria.find((x) => x.criterion === criterion);
    if (!c) return;

    setExamFeedbacks((prev) => ({ ...prev, [criterion]: { ...(prev[criterion] || { text: "", error: "", isOpen: false }), status: "streaming", isOpen: true } }));
    setExamFeedbackStatus("streaming");
    setExamTokenDebug((prev) => prev ? { ...prev, calls: prev.calls.map((call) => call.criterion === criterion ? { ...call, status: "streaming" } : call) } : prev);

    const controller = new AbortController();
    examFeedbackAbortRef.current = controller;

    const body = {
      paper_type: paperType, criterion: c.criterion, score: c.score, max_score: c.max_score,
      student_answer: studentAnswer, question: questionText,
      passage_text: paperType === "paper1" ? passage : null,
      document_ids: paperType === "paper2" ? selectedDocIds : [],
      context_mode: paperType === "paper2" ? contextMode : "chunks",
      model: modelKey || null,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/exam/criterion-feedback/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: controller.signal, body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await parseErrorDetail(response, "Feedback request failed."));
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let doneDebug = null;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let ev; try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          if (ev.type === "token") setExamFeedbacks((prev) => ({ ...prev, [criterion]: { ...prev[criterion], text: (prev[criterion]?.text || "") + ev.text } }));
          else if (ev.type === "done") doneDebug = ev.debug;
          else if (ev.type === "error") throw new Error(ev.detail || "Feedback stream error.");
        }
      }
      setExamFeedbacks((prev) => ({ ...prev, [criterion]: { ...prev[criterion], status: "done" } }));
      setExamTokenDebug((prev) => {
        if (!prev || !doneDebug) return prev;
        return {
          ...prev, nCtx: doneDebug.n_ctx ?? prev.nCtx,
          totalTokensEstimated: prev.totalTokensEstimated + (doneDebug.prompt_tokens || 0),
          calls: prev.calls.map((call) => call.criterion === criterion ? { ...call, promptTokens: doneDebug.prompt_tokens, remainingBudget: doneDebug.remaining_budget, inferenceSeconds: doneDebug.inference_seconds, status: "done" } : call),
        };
      });
      return true; // not aborted
    } catch (err) {
      if (err?.name === "AbortError") {
        setExamFeedbacks((prev) => ({ ...prev, [criterion]: { ...prev[criterion], status: "canceled" } }));
        setExamTokenDebug((prev) => prev ? { ...prev, calls: prev.calls.map((call) => call.criterion === criterion ? { ...call, status: "canceled" } : call) } : prev);
        return false; // aborted
      }
      setExamFeedbacks((prev) => ({ ...prev, [criterion]: { ...prev[criterion], status: "error", error: err.message } }));
      setExamTokenDebug((prev) => prev ? { ...prev, calls: prev.calls.map((call) => call.criterion === criterion ? { ...call, status: "error" } : call) } : prev);
      return true; // error but not aborted, continue to next
    }
  };

  // Update overall feedback status based on per-criterion statuses.
  const _updateExamFeedbackStatus = (feedbacks) => {
    const statuses = Object.values(feedbacks).map((f) => f.status);
    if (statuses.every((s) => s === "done" || s === "canceled" || s === "error")) {
      setExamFeedbackStatus("done");
    }
  };

  const startFeedbackStreaming = async (gradingData, studentAnswer, gradingPromptTokens, modelKey) => {
    const criteria = gradingData.criteria || [];
    const initial = {};
    for (const c of criteria) initial[c.criterion] = { status: "pending", text: "", error: "", isOpen: false };
    setExamFeedbacks(initial);
    setExamFeedbackStatus("streaming");
    setExamTokenDebug({
      nCtx: null,
      gradingPromptTokens: gradingPromptTokens || 0,
      calls: criteria.map((c) => ({ criterion: c.criterion, label: c.label, promptTokens: null, remainingBudget: null, inferenceSeconds: null, status: "pending" })),
      totalTokensEstimated: gradingPromptTokens || 0,
    });

    // Capture current exam state for the feedback calls
    const capturedAnswer = studentAnswer;
    const capturedQuestion = examQuestionText;
    const capturedPaperType = examPaperType;
    const capturedPassage = examPassage;
    const capturedDocIds = examSelectedDocIds;
    const capturedContextMode = examContextMode;

    for (const c of criteria) {
      const continued = await generateCriterionFeedback(
        c.criterion, criteria, capturedAnswer, capturedQuestion,
        capturedPaperType, capturedPassage, capturedDocIds, capturedContextMode, modelKey,
      );
      if (!continued) break; // user aborted, stop auto-advancing
    }

    // Set final status based on what happened
    setExamFeedbacks((prev) => {
      const statuses = Object.values(prev).map((f) => f.status);
      if (statuses.every((s) => s === "done" || s === "canceled" || s === "error")) {
        setExamFeedbackStatus("done");
      }
      return prev;
    });
  };

  const onSubmitExamAnswer = async (e) => {
    e.preventDefault();
    if (!examQuestionText || !examAnswer.trim()) return;
    setExamGradingStatus("grading");
    setExamGradingError("");
    setExamFeedbacks({});
    setExamFeedbackStatus("idle");
    setExamTokenDebug(null);
    setExamGradingElapsed(0);
    if (examGradingIntervalRef.current) clearInterval(examGradingIntervalRef.current);
    examGradingIntervalRef.current = setInterval(() => setExamGradingElapsed((s) => s + 1), 1000);
    const controller = new AbortController();
    examGradingAbortRef.current = controller;

    try {
      const studentAnswer = examAnswer.trim();
      const examModelKey = sessionModels.get(activeExamSessionId || "__exam__") || null;
      const body = {
        paper_type: examPaperType, question: examQuestionText, student_answer: studentAnswer,
        document_ids: examPaperType === "paper2" ? examSelectedDocIds : [],
        context_mode: examPaperType === "paper2" ? examContextMode : "chunks",
        passage_text: examPaperType === "paper1" ? examPassage : null,
        model: examModelKey,
      };
      const response = await fetch(`${API_BASE_URL}/api/exam/submit-answer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: controller.signal, body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await parseErrorDetail(response, "Grading failed."));
      const data = await response.json();
      setExamGrading(data);
      setExamDebug(data.debug || null);
      setExamGradingStatus("done");
      setExamStep("done");
      startFeedbackStreaming(data, studentAnswer, data.debug?.prompt_tokens || 0, examModelKey);
      setTimeout(() => saveExamState(activeExamSessionId), 0);
    } catch (err) {
      if (err?.name === "AbortError") {
        setExamGradingStatus("idle");
        setExamGradingError("Grading was cancelled.");
      } else {
        setExamGradingStatus("error");
        setExamGradingError(err?.message || "Unexpected error during grading.");
      }
    } finally {
      if (examGradingIntervalRef.current) { clearInterval(examGradingIntervalRef.current); examGradingIntervalRef.current = null; }
    }
  };

  // ── Navigation helpers ─────────────────────────────────────────────────────
  const goTo = (page) => {
    if (page === "exam") {
      loadExamSessions().then((sessions) => {
        if (sessions.length > 0) {
          onSelectExamSession(sessions[0].session_id);
        } else {
          resetExam();
          createExamSession();
        }
      });
    }
    setActivePage(page);
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER HELPERS
  // ─────────────────────────────────────────────────────────────────────────────

  // Model selector (per-session)
  const recommendedTier = systemInfo?.recommended_model;
  const recommendedModel = availableModels.find((m) => getModelTier(m.filename) === recommendedTier);

  const renderSessionModelSelector = (sessionKey) => {
    if (availableModels.length === 0) return null;
    const current = sessionModels.get(sessionKey) || activeModel || "";
    return (
      <div className="model-wrap" style={{ marginBottom: 8 }}>
        <select
          className="model-select"
          value={current}
          onChange={(e) => {
            const filename = e.target.value;
            setSessionModels((prev) => {
              const m = new Map(prev);
              m.set(sessionKey, filename);
              return m;
            });
          }}
          title="Model for this session"
        >
          {availableModels.map((m) => {
            const isRec = getModelTier(m.filename) === recommendedTier;
            return (
              <option key={m.filename} value={m.filename}>
                {shortModelName(m.filename)}{isRec ? " ★" : ""}
              </option>
            );
          })}
        </select>
      </div>
    );
  };

  // ── Header ─────────────────────────────────────────────────────────────────
  const pageLabels = { dashboard: "IB Literature Tutor", library: "Library", learn: "Learn", exam: "Exam" };

  const renderHeader = () => (
    <header className="app-header">
      <div className="header-left">
        {activePage !== "dashboard" && (
          <button className="back-btn" onClick={() => setActivePage("dashboard")}>← Dashboard</button>
        )}
        <span className="header-title">{pageLabels[activePage]}</span>
      </div>
      <div className="header-right">
        <div className="theme-toggle">
          {["auto", "light", "dark"].map((t) => (
            <button key={t} className={`theme-opt${theme === t ? " active" : ""}`} onClick={() => setThemePref(t)}>
              {t === "auto" ? "Auto" : t === "light" ? "☀ Day" : "☾ Night"}
            </button>
          ))}
        </div>
      </div>
    </header>
  );

  // ── Dashboard Page ─────────────────────────────────────────────────────────
  const renderDashboard = () => {
    const docCount = availableDocs.length + (result?.documentId && !availableDocs.find((d) => d.document_id === result.documentId) ? 1 : 0);

    return (
      <div>
        <div className="dash-hero">
          <h1 className="page-title">IB Literature Tutor</h1>
          <p className="page-sub">Upload literary works, chat with your documents, and practise IB exam essays with instant AI feedback.</p>
        </div>

        {/* System info */}
        {systemInfo && (
          <div className="sysinfo-bar">
            <span className="sysinfo-chip">
              <span className="sysinfo-lbl">RAM</span>
              <strong>{systemInfo.ram_available_gb?.toFixed(1)} / {systemInfo.ram_total_gb?.toFixed(1)} GB</strong>
            </span>
            <span className="sysinfo-chip">
              <span className="sysinfo-lbl">CPU</span>
              <strong>{systemInfo.cpu_cores_physical}c / {systemInfo.cpu_cores_logical}t{systemInfo.cpu_has_avx2 ? " · AVX2" : ""}</strong>
            </span>
            {systemInfo.gpu_name && (
              <span className="sysinfo-chip">
                <span className="sysinfo-lbl">GPU</span>
                <strong>{systemInfo.gpu_name} · {systemInfo.gpu_vram_free_gb?.toFixed(1)}/{systemInfo.gpu_vram_total_gb?.toFixed(1)} GB VRAM</strong>
              </span>
            )}
          </div>
        )}

        {/* Model recommendation */}
        {systemInfo?.recommended_model && (
          <div className="recommend-card">
            <span className="recommend-icon">🎯</span>
            <div className="recommend-body">
              <div className="recommend-label">Recommended for your hardware</div>
              <div className="recommend-name">
                {recommendedModel ? shortModelName(recommendedModel.filename) : `${systemInfo.recommended_model.toUpperCase()} model`}
                {activeModel && getModelTier(activeModel) === systemInfo.recommended_model
                  ? <span style={{ marginLeft: 8, fontSize: "0.78rem", color: "var(--success)" }}>✓ active</span>
                  : null}
              </div>
              {systemInfo.recommendation_reason && (
                <div className="recommend-reason">{systemInfo.recommendation_reason}</div>
              )}
            </div>
          </div>
        )}

        {/* Nav cards */}
        <div className="nav-grid">
          <button className="nav-card" onClick={() => goTo("library")}>
            <span className="nav-card-icon">📚</span>
            <div className="nav-card-title">Library</div>
            <div className="nav-card-desc">Upload and manage your literary texts.</div>
            {docCount > 0 && <span className="nav-card-badge">{docCount} doc{docCount !== 1 ? "s" : ""}</span>}
          </button>

          <button
            className={`nav-card${!modesAvailable ? " nav-card--off" : ""}`}
            onClick={() => modesAvailable && goTo("learn")}
            title={!modesAvailable ? "Upload a document first" : undefined}
          >
            <span className="nav-card-icon">💬</span>
            <div className="nav-card-title">Learn</div>
            <div className="nav-card-desc">Chat with your documents using AI.</div>
            {!modesAvailable && <span className="nav-card-badge">Upload first</span>}
          </button>

          <button
            className={`nav-card${!modesAvailable || !llmAvailable ? " nav-card--off" : ""}`}
            onClick={() => modesAvailable && llmAvailable && goTo("exam")}
            title={!modesAvailable ? "Upload a document first" : !llmAvailable ? "LLM not available" : undefined}
          >
            <span className="nav-card-icon">📝</span>
            <div className="nav-card-title">Exam</div>
            <div className="nav-card-desc">Practise IB essays with instant grading.</div>
            {!llmAvailable && <span className="nav-card-badge">No LLM</span>}
          </button>
        </div>
      </div>
    );
  };

  // ── Library Page ───────────────────────────────────────────────────────────
  const statusText = uploadStatus === "processing" ? "Processing PDF…" : uploadStatus === "chunking" ? "Generating chunks…" : uploadStatus === "success" ? "Done" : uploadStatus === "error" ? "Failed" : "Ready";
  const progressClass = isProcessing || isChunking ? "running" : uploadStatus === "success" ? "done" : "idle";

  const renderLibrary = () => (
    <div>
      {/* Upload */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-title">Upload a PDF</div>
        <form onSubmit={onSubmit}>
          <div className="upload-row">
            <input className="file-input" type="file" accept="application/pdf,.pdf" onChange={onFileChange} disabled={isBusy} />
            <button className="btn" type="submit" disabled={isBusy || !file}>
              {isProcessing ? "Processing…" : "Upload & Process"}
            </button>
          </div>
        </form>
        <div className="status-bar">
          <span>Status: {statusText}</span>
          <span>{isBusy ? `${elapsedSeconds}s elapsed` : ""}</span>
        </div>
        <div className="progress-track">
          <div className={`progress-fill ${progressClass}`} />
        </div>
        {uploadError && <div className="msg msg-error">{uploadError}</div>}
        {result && (
          <div className="result-card">
            {result.mode && <p className="result-meta"><strong>Mode:</strong> {result.mode}</p>}
            {result.pages && <p className="result-meta"><strong>Pages:</strong> {result.pages}</p>}
            {result.chars && <p className="result-meta"><strong>Characters:</strong> {result.chars}</p>}
            {result.chunksCount != null && <p className="result-meta"><strong>Chunks:</strong> {result.chunksCount}</p>}
            <div className="result-actions">
              <a className="dl-link" href={result.downloadUrl} download={result.filename}>Download TXT</a>
              {result.documentId && !chunksAvailable && (
                <button className="btn btn-ghost btn-sm" type="button" onClick={onGenerateChunks} disabled={isBusy}>
                  {isChunking ? "Regenerating…" : "Retry Chunks"}
                </button>
              )}
              {result.chunksDownloadUrl && (
                <a className="dl-link" href={result.chunksDownloadUrl} download>Download Chunks</a>
              )}
            </div>
          </div>
        )}
        {uploadQualityDebug && (
          <details className="dbg-details" style={{ marginTop: 10 }}>
            <summary>▸ Quality Decision</summary>
            <div className="dbg-body">
              <p className="dbg-line">
                <strong>Decision:</strong>{" "}
                {uploadQualityDebug.decision === "fingerprint-match"
                  ? "Reused existing chunks (identical file — SHA256 match)"
                  : uploadQualityDebug.decision === "existing-retained"
                  ? "Reused existing chunks (existing quality was sufficient)"
                  : uploadQualityDebug.decision === "new-upgraded"
                  ? "Replaced with new upload (better quality)"
                  : "New document"}
              </p>
              <p className="dbg-line">
                <strong>Method:</strong>{" "}
                {uploadQualityDebug.method === "fingerprint" ? "Binary fingerprint (no comparison needed)"
                  : uploadQualityDebug.method === "noise-check" ? "Noise check (heuristic)"
                  : uploadQualityDebug.method === "llm-score" ? "LLM quality scoring"
                  : uploadQualityDebug.method === "llm-unavailable" ? "LLM unavailable — kept existing"
                  : uploadQualityDebug.method || "—"}
              </p>
              {uploadQualityDebug.scoreNew != null && uploadQualityDebug.method !== "fingerprint" && (
                <p className="dbg-line">
                  <strong>
                    {uploadQualityDebug.method === "noise-check" ? "Noise density" : "Scores"}:
                  </strong>{" "}
                  New: {uploadQualityDebug.scoreNew.toFixed(4)}
                  {uploadQualityDebug.method === "noise-check" ? " (lower = cleaner)" : ""} / Existing:{" "}
                  {uploadQualityDebug.scoreExisting != null
                    ? uploadQualityDebug.scoreExisting.toFixed(4)
                    : "—"}
                  {uploadQualityDebug.method === "noise-check" ? "" : " / 10"}
                </p>
              )}
            </div>
          </details>
        )}
      </div>

      {/* Document list */}
      <div className="card">
        <div className="section-title">Your Documents</div>
        {deleteError && <div className="msg msg-error" style={{ marginBottom: 12 }}>{deleteError}</div>}
        {availableDocs.length === 0 ? (
          <div className="doc-list-empty">No documents yet. Upload a PDF above to get started.</div>
        ) : (
          <div className="doc-list">
            {availableDocs.map((doc) => (
              <div key={doc.document_id} className="doc-item">
                <span className="doc-item-icon">📄</span>
                <div className="doc-item-body">
                  <div className="doc-item-name" title={doc.filename}>{doc.title || doc.filename}</div>
                  <div className="doc-item-meta">
                    {doc.author ? `${doc.author} · ` : ""}
                    {doc.chunks_count} chunks
                    {doc.pages ? ` · ${doc.pages} pages` : ""}
                  </div>
                </div>
                <div className="doc-item-actions">
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => { setSelectedLearnDocId(doc.document_id); resetChat(); goTo("learn"); }}
                  >
                    Learn
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => onDeleteDocument(doc.document_id)}
                    disabled={deleteDocId === doc.document_id}
                  >
                    {deleteDocId === doc.document_id ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  // ── Learn Page ─────────────────────────────────────────────────────────────
  const renderLearn = () => {
    const activeSession = learnSessions.find((s) => s.session_id === activeLearnSessionId);
    const isAtLimit = (activeSession?.turn_count ?? 0) >= CHAT_MAX_MESSAGES;

    const chatPanel = (
      <div className="card chat-panel">
        <div className="section-title">
          {result?.documentId ? `Chat · ${result.documentFilename || "Uploaded document"}` : selectedLearnDocId ? `Chat · ${availableDocs.find((d) => d.document_id === selectedLearnDocId)?.title || "Selected document"}` : "Chat"}
        </div>

        {hasChatDocument && !chatAvailable && (
          <p style={{ fontSize: "0.875rem", color: "var(--muted)" }}>Chat unavailable. Configure backend LLM env vars and a valid model path.</p>
        )}

        {!hasChatDocument && (
          <p style={{ fontSize: "0.875rem", color: "var(--muted)" }}>Select a document above to start chatting.</p>
        )}

        {hasChatDocument && chatAvailable && (
          <>
            <div className="chat-list">
              {chatMessages.length === 0
                ? <p className="chat-empty">No messages yet. Ask a question about the document.</p>
                : chatMessages.map((msg, i) => (
                    <div key={`${msg.role}-${i}`} className="chat-msg-block">
                      {msg.role === "assistant" && msg.totalSeconds && (
                        <p className="chat-timing">Response: {formatSeconds(msg.totalSeconds)}</p>
                      )}
                      <p className={`chat-msg ${msg.role}`}>
                        <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong>{" "}
                        {msg.content}
                        {msg.streaming && <span className="cursor-blink">▍</span>}
                      </p>
                      {msg.role === "assistant" && msg.streaming && (
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{ marginTop: 4, fontSize: "0.78rem" }}
                          onClick={() => sessionAbortRefs.current.get(_sessionKey)?.abort()}
                        >
                          ■ Stop
                        </button>
                      )}
                      {msg.role === "assistant" && msg.stopped && (
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{ marginTop: 4, fontSize: "0.78rem" }}
                          onClick={() => regenerateChat(_sessionKey)}
                        >
                          ↺ Regenerate
                        </button>
                      )}
                    </div>
                  ))
              }
            </div>

            {isAtLimit && (
              <div className="msg-warning">
                Maximum {CHAT_MAX_MESSAGES} messages reached — start a new session to continue.
              </div>
            )}

            <div>
              {renderSessionModelSelector(_sessionKey)}
              <div className="pill-row">
                <button className={`pill${promptFormat === null ? " active" : ""}`} disabled={isChatSending || isAtLimit} onClick={() => setPromptFormat(null)}>Hybrid</button>
                <button className={`pill${promptFormat === "rag" ? " active" : ""}`} disabled={isChatSending || isAtLimit} onClick={() => setPromptFormat("rag")}>RAG</button>
                <button className={`pill${promptFormat === "base_knowledge" ? " active" : ""}`} disabled={isChatSending || isAtLimit} onClick={() => setPromptFormat("base_knowledge")}>Base Knowledge</button>
              </div>
              {promptFormat === "base_knowledge" && <p className="pill-hint">Answering from general knowledge — no document retrieval.</p>}
              {promptFormat === null && <p className="pill-hint">Auto-selects RAG or Base Knowledge based on retrieval confidence.</p>}
            </div>

            <form className="chat-form" onSubmit={onSendChat}>
              <textarea
                className="chat-textarea"
                placeholder={isAtLimit ? "Start a new session to continue…" : "Ask something about this document…"}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={isChatSending || isBusy || isAtLimit}
              />
              <button className="btn" type="submit" disabled={isChatSending || isBusy || !chatInput.trim() || isAtLimit}>
                {isChatSending ? "Sending…" : "Send"}
              </button>
            </form>
            {chatError && <div className="msg msg-error">{chatError}</div>}
          </>
        )}

        {/* Router Debug */}
        <RouterDebugPanel debug={chatRoutingDebug} />

        {/* Debug */}
        {chatDebug && (
          <details className="dbg-details">
            <summary>▸ Debug{chatDebug.routing_reason ? ` · ${chatDebug.routing_reason}` : ""}</summary>
            <div className="dbg-body">
              <div style={{ marginBottom: 10 }}>
                <p className="dbg-line"><strong>Total time:</strong> {formatSeconds(chatDebug.total_seconds)}</p>
                <p className="dbg-line"><strong>Retrieval:</strong> {formatSeconds(chatDebug.timing?.retrieval_seconds)}</p>
                <p className="dbg-line"><strong>Inference:</strong> {formatSeconds(chatDebug.timing?.inference_seconds)}</p>
                {chatDebug.prompt_mode && <p className="dbg-line"><strong>Prompt mode:</strong> {chatDebug.prompt_mode}</p>}
                {chatDebug.routing_reason && <p className="dbg-line"><strong>Routing:</strong> {chatDebug.routing_reason}</p>}
              </div>
              <div className="dbg-section">Final Prompt</div>
              <pre className="dbg-pre">{chatDebug.final_prompt || ""}</pre>
              {(chatDebug.raw_retrieved_excerpts || []).length > 0 ? (
                <>
                  <div className="dbg-section">Retrieved Chunks (raw)</div>
                  {chatDebug.raw_retrieved_excerpts.map((ex) => (
                    <div key={`raw-${ex.excerpt_id}`} className="dbg-excerpt">
                      <p className="dbg-excerpt-meta"><strong>{ex.excerpt_id}</strong> | Score {Number(ex.score || 0).toFixed(3)} | {ex.page_start === ex.page_end ? `p.${ex.page_start}` : `pp.${ex.page_start}-${ex.page_end}`} | {ex.heading || "—"}</p>
                      <p className="dbg-excerpt-text">{ex.text || ""}</p>
                    </div>
                  ))}
                  <div className="dbg-section">Sub-chunked (sent to model)</div>
                  {(chatDebug.retrieved_excerpts || []).map((ex) => (
                    <div key={`sub-${ex.excerpt_id}`} className="dbg-excerpt">
                      <p className="dbg-excerpt-meta"><strong>{ex.excerpt_id}</strong> | Score {Number(ex.score || 0).toFixed(3)} | {ex.page_start === ex.page_end ? `p.${ex.page_start}` : `pp.${ex.page_start}-${ex.page_end}`} | {ex.heading || "—"}</p>
                      <p className="dbg-excerpt-text">{ex.text || ""}</p>
                    </div>
                  ))}
                </>
              ) : (chatDebug.retrieved_excerpts || []).length > 0 ? (
                <>
                  <div className="dbg-section">Retrieved Excerpts</div>
                  {chatDebug.retrieved_excerpts.map((ex) => (
                    <div key={`${ex.excerpt_id}-${ex.page_start}`} className="dbg-excerpt">
                      <p className="dbg-excerpt-meta"><strong>{ex.excerpt_id}</strong> | Score {Number(ex.score || 0).toFixed(3)} | {ex.page_start === ex.page_end ? `p.${ex.page_start}` : `pp.${ex.page_start}-${ex.page_end}`} | {ex.heading || "—"}</p>
                      <p className="dbg-excerpt-text">{ex.text || ""}</p>
                    </div>
                  ))}
                </>
              ) : null}
            </div>
          </details>
        )}
      </div>
    );

    const sessionsSidebar = hasChatDocument && learnSessions.length > 0 && (
      <div className="sessions-sidebar">
        <div className="sessions-sidebar-header">
          <span>Sessions</span>
          <button
            className="btn btn-ghost btn-sm"
            style={{ fontSize: "0.75rem", padding: "2px 8px" }}
            onClick={() => createLearnSession(activeChatDocId)}
            title="New session"
          >+ New</button>
        </div>
        <div className="sessions-list">
          {learnSessions.map((s) => (
            <div key={s.session_id}>
              {renamingSessionId === s.session_id ? (
                <div className="session-rename-row">
                  <input
                    className="session-rename-input"
                    value={renameValue}
                    autoFocus
                    maxLength={100}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") onRenameSession(s.session_id); if (e.key === "Escape") setRenamingSessionId(null); }}
                    onBlur={() => onRenameSession(s.session_id)}
                  />
                </div>
              ) : (
                <div
                  className={`session-item${activeLearnSessionId === s.session_id ? " active" : ""}`}
                  onClick={() => onSelectLearnSession(s.session_id)}
                >
                  <span className="session-item-name">{s.name || "Session"}</span>
                  {sessionStatuses.get(s.session_id) === "sending" && (
                    <span className="session-loading-dot" title="Loading…">●</span>
                  )}
                  <span className="session-turn-badge">{s.turn_count ?? 0}/{CHAT_MAX_MESSAGES}</span>
                  <div className="session-actions" onClick={(e) => e.stopPropagation()}>
                    <button
                      className="session-icon-btn"
                      title="Rename"
                      onClick={() => { setRenamingSessionId(s.session_id); setRenameValue(s.name || ""); }}
                    >✎</button>
                    <button
                      className="session-icon-btn"
                      title="Delete"
                      onClick={() => onDeleteSession(s.session_id)}
                    >✕</button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );

    return (
      <div className="learn-layout">
        {/* Doc picker */}
        {!result?.documentId && (
          <div className="card">
            <div className="section-title">Select a Work</div>
            {availableDocs.length === 0 ? (
              <p style={{ fontSize: "0.875rem", color: "var(--muted)" }}>
                No documents available. <button className="btn btn-ghost btn-sm" onClick={() => goTo("library")}>Go to Library</button>
              </p>
            ) : (
              <div className="doc-pick-grid">
                {availableDocs.map((doc) => (
                  <div
                    key={doc.document_id}
                    className={`doc-pick-card${selectedLearnDocId === doc.document_id ? " selected" : ""}`}
                    onClick={async () => {
                      if (sessionCreatingRef.current) return;
                      sessionCreatingRef.current = true;
                      setSelectedLearnDocId(doc.document_id);
                      setActiveLearnSessionId(null);
                      setLearnSessions([]);
                      try {
                        await _loadOrCreateSession(doc.document_id);
                      } finally {
                        sessionCreatingRef.current = false;
                      }
                    }}
                  >
                    <div className="doc-pick-name">{doc.title || doc.filename}</div>
                    <div className="doc-pick-meta">{doc.author ? `${doc.author} · ` : ""}{doc.chunks_count} chunks</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Chat + Sessions */}
        {sessionsSidebar ? (
          <div className="learn-with-sessions">
            {sessionsSidebar}
            {chatPanel}
          </div>
        ) : chatPanel}
      </div>
    );
  };

  // ── Exam Page ──────────────────────────────────────────────────────────────
  const renderExam = () => {
    const examSidebarEl = examSessions.length > 0 && (
      <div className="sessions-sidebar">
        <div className="sessions-sidebar-header">
          <span>Sessions</span>
          <button
            className="btn btn-ghost btn-sm"
            style={{ fontSize: "0.75rem", padding: "2px 8px" }}
            onClick={() => { saveExamState(activeExamSessionId); createExamSession().then(() => resetExam()); }}
            title="New session"
          >+ New</button>
        </div>
        <div className="sessions-list">
          {examSessions.map((s) => (
            <div key={s.session_id}>
              {renamingSessionId === s.session_id ? (
                <div className="session-rename-row">
                  <input
                    className="session-rename-input"
                    value={renameValue}
                    autoFocus
                    maxLength={100}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") onRenameExamSession(s.session_id); if (e.key === "Escape") setRenamingSessionId(null); }}
                    onBlur={() => onRenameExamSession(s.session_id)}
                  />
                </div>
              ) : (
                <div
                  className={`session-item${activeExamSessionId === s.session_id ? " active" : ""}`}
                  onClick={() => onSelectExamSession(s.session_id)}
                >
                  <span className="session-item-name">{s.name || "Session"}</span>
                  <div className="session-actions" onClick={(e) => e.stopPropagation()}>
                    <button className="session-icon-btn" title="Rename" onClick={() => { setRenamingSessionId(s.session_id); setRenameValue(s.name || ""); }}>✎</button>
                    <button className="session-icon-btn" title="Delete" onClick={() => onDeleteExamSession(s.session_id)}>✕</button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
    return (
    <div className="exam-layout">
      <div className={examSidebarEl ? "learn-with-sessions" : ""}>
        {examSidebarEl}
        <div className="card exam-panel">
        <div className="paper-tabs">
          <button
            className={`paper-tab${examPaperType === "paper1" ? " active" : ""}`}
            disabled={examStep !== "setup" || examDocsStatus === "loading"}
            onClick={() => { setExamPaperType("paper1"); resetExam(); }}
          >
            Paper 1 — Guided Analysis (max 20)
          </button>
          <button
            className={`paper-tab${examPaperType === "paper2" ? " active" : ""}`}
            disabled={examStep !== "setup" || examDocsStatus === "loading"}
            onClick={() => { setExamPaperType("paper2"); resetExam(); }}
          >
            Paper 2 — Comparative Essay (max 40)
          </button>
        </div>

        {/* Setup step */}
        {examStep === "setup" && examDocsStatus === "idle" && (
          <div>
            <p style={{ fontSize: "0.875rem", color: "var(--muted)", marginBottom: 14 }}>
              {examPaperType === "paper1"
                ? "A random passage and guiding question will be generated."
                : "Select two works from your library to compare."}
            </p>
            {renderSessionModelSelector(activeExamSessionId || "__exam__")}
            <button className="btn" onClick={() => onEnterExam(examPaperType)}>Start Exam</button>
          </div>
        )}

        {examDocsStatus === "loading" && <p style={{ color: "var(--muted)", fontSize: "0.875rem" }}>Loading exam data…</p>}
        {examDocsStatus === "error" && <div className="msg msg-error">{examDocsError}</div>}

        {/* Paper 2 doc picker */}
        {examStep === "setup" && examDocsStatus === "ready" && examPaperType === "paper2" && (
          <div>
            <div className="section-title">Select Two Works</div>
            <p style={{ fontSize: "0.82rem", color: "var(--muted)", marginBottom: 12 }}>Choose exactly 2 works to compare in your essay.</p>
            {examAllDocs.length === 0 ? (
              <p style={{ fontSize: "0.875rem", color: "var(--muted)" }}>No documents with chunks available. Upload PDFs first.</p>
            ) : (
              <div className="doc-pick-grid-exam">
                {examAllDocs.map((doc) => {
                  const isSel = examSelectedDocIds.includes(doc.document_id);
                  const isOff = !isSel && examSelectedDocIds.length >= 2;
                  return (
                    <div
                      key={doc.document_id}
                      className={`doc-pick-card${isSel ? " selected" : ""}${isOff ? " disabled" : ""}`}
                      onClick={() => !isOff && onToggleDocSelection(doc.document_id)}
                    >
                      <div className="doc-pick-name">{doc.title || doc.filename}</div>
                      <div className="doc-pick-meta">{doc.author ? `${doc.author} · ` : ""}{doc.chunks_count} chunks</div>
                    </div>
                  );
                })}
              </div>
            )}
            {renderSessionModelSelector(activeExamSessionId || "__exam__")}
            <button className="btn" style={{ marginTop: 14 }} onClick={onConfirmDocSelection} disabled={examSelectedDocIds.length !== 2}>Continue →</button>
          </div>
        )}

        {/* Writing step */}
        {examStep === "writing" && (
          <div>
            {examPaperType === "paper1" && examPassage && <div className="passage-box">{examPassage}</div>}
            {examPaperType === "paper2" && (
              <>
                <div className="tags-row">
                  {examSelectedDocIds.map((id) => {
                    const doc = examAllDocs.find((d) => d.document_id === id);
                    return <span key={id} className="tag">{doc ? (doc.title || doc.filename) : id}</span>;
                  })}
                </div>
                <div className="radio-group">
                  <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>Context:</span>
                  <label className="radio-label"><input type="radio" name="ctxMode" value="chunks" checked={examContextMode === "chunks"} onChange={() => setExamContextMode("chunks")} /> RAG (excerpt retrieval)</label>
                  <label className="radio-label"><input type="radio" name="ctxMode" value="titles_only" checked={examContextMode === "titles_only"} onChange={() => setExamContextMode("titles_only")} /> Base knowledge</label>
                </div>
              </>
            )}
            {examQuestionText && <div className="question-box">{examQuestionText}</div>}
            <form onSubmit={onSubmitExamAnswer}>
              <p style={{ fontSize: "0.82rem", color: "var(--muted)", marginBottom: 8 }}>Write your response below. No AI assistance during this step.</p>
              {renderSessionModelSelector(activeExamSessionId || "__exam__")}
              <textarea
                className="exam-textarea"
                placeholder="Write your answer here…"
                value={examAnswer}
                onChange={(e) => setExamAnswer(e.target.value)}
                disabled={examGradingStatus === "grading"}
              />
              {examGradingStatus === "grading" && (
                <div className="msg msg-info" style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                  <span style={{ fontSize: "1.3rem" }}>⏳</span>
                  <div style={{ flex: 1 }}>
                    <strong>Grading in progress — this typically takes around a minute.</strong>
                    <div style={{ fontSize: "0.82rem", marginTop: 2, opacity: 0.85 }}>
                      You can safely browse the app while waiting. · {examGradingElapsed}s elapsed
                    </div>
                  </div>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => examGradingAbortRef.current?.abort()}>Cancel</button>
                </div>
              )}
              <button className="btn" type="submit" disabled={!examAnswer.trim() || examGradingStatus === "grading"}>
                {examGradingStatus === "grading" ? "Grading…" : "Submit for Grading"}
              </button>
              {examGradingStatus === "error" && <div className="msg msg-error">{examGradingError}</div>}
            </form>
          </div>
        )}

        {/* Results step */}
        {examStep === "done" && examGrading && (() => {
          const criteria = examGrading.criteria || [];
          const streamingCriterion = criteria.find((c) => (examFeedbacks[c.criterion] || {}).status === "streaming");
          const doneCount = criteria.filter((c) => { const s = (examFeedbacks[c.criterion] || {}).status; return s === "done" || s === "canceled" || s === "error"; }).length;
          const examModelKey = sessionModels.get(activeExamSessionId || "__exam__") || null;
          return (
          <div>
            {/* Prominent feedback-generating banner */}
            {examFeedbackStatus === "streaming" && (
              <div className="msg msg-warning" style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                <span style={{ fontSize: "1.2rem", flexShrink: 0 }}>✍</span>
                <div style={{ flex: 1 }}>
                  <strong>Detailed coaching feedback is being generated.</strong>
                  <div style={{ fontSize: "0.82rem", marginTop: 2 }}>
                    {streamingCriterion
                      ? `Generating Criterion ${streamingCriterion.criterion} — step ${doneCount + 1} of ${criteria.length}`
                      : `Step ${doneCount + 1} of ${criteria.length}`}
                    {" · This may take a minute or two per criterion."}
                  </div>
                </div>
                <button className="btn btn-ghost btn-sm" onClick={() => examFeedbackAbortRef.current?.abort()}>Stop</button>
              </div>
            )}

            <div className="exam-results">
              <div className="score-heading">Total: {examGrading.total_score} / {examGrading.max_score}</div>
              <p className="context-badge">
                {(examGrading.context_mode === "titles_only" || examDebug?.context_mode_effective === "titles_only")
                  ? `Graded with: base knowledge (no retrieval)${examGrading.context_mode === "chunks" && examDebug?.context_mode_effective === "titles_only" ? " — overridden by router" : ""}`
                  : "Graded with: RAG (excerpt retrieval)"}
              </p>

              {criteria.map((c) => {
                const fb = examFeedbacks[c.criterion] || { status: "pending", text: "", error: "", isOpen: false };
                const isStreaming = fb.status === "streaming";
                const isPending = fb.status === "pending" || fb.status === "idle";
                const isCanceled = fb.status === "canceled";
                const isError = fb.status === "error";
                const isDone = fb.status === "done";
                const canGenerate = (isPending || isCanceled || isError) && examFeedbackStatus !== "streaming";
                return (
                  <div key={c.criterion} className="criterion-card">
                    <div className="criterion-head">
                      <span className="criterion-name">{c.criterion} — {c.label}</span>
                      <span className="criterion-badge">{c.score} / {c.max_score}</span>
                    </div>
                    <p className="criterion-note">
                      <span className="criterion-label">Examiner note:</span>
                      {c.feedback}
                    </p>
                    {/* Streaming feedback (inline) */}
                    {(fb.text.length > 0 || isStreaming) && (
                      <details
                        open={fb.isOpen}
                        onToggle={(e) => { const open = e.currentTarget.open; setExamFeedbacks((prev) => ({ ...prev, [c.criterion]: { ...prev[c.criterion], isOpen: open } })); }}
                        style={{ marginTop: 8 }}
                      >
                        <summary style={{ cursor: "pointer", fontSize: "0.82rem", color: "var(--muted)", fontWeight: 600 }}>
                          {isStreaming ? "Detailed coaching (generating…)" : isCanceled ? "Detailed coaching (partial)" : "Detailed coaching"}
                        </summary>
                        <div style={{ marginTop: 6, fontSize: "0.875rem", lineHeight: 1.6, whiteSpace: "pre-wrap", color: "var(--text-2)" }}>
                          {fb.text}
                          {isStreaming && <span className="cursor-block" />}
                        </div>
                      </details>
                    )}
                    {/* Status indicators + action buttons */}
                    <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                      {isPending && fb.text.length === 0 && (
                        <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>Detailed coaching not yet generated.</span>
                      )}
                      {isCanceled && (
                        <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>Generation stopped.</span>
                      )}
                      {isError && (
                        <span style={{ fontSize: "0.78rem", color: "var(--error)" }}>Error: {fb.error}</span>
                      )}
                      {canGenerate && (
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{ fontSize: "0.78rem" }}
                          onClick={() => {
                            setExamFeedbackStatus("streaming");
                            generateCriterionFeedback(
                              c.criterion, criteria, examAnswer.trim(), examQuestionText,
                              examPaperType, examPassage, examSelectedDocIds, examContextMode, examModelKey,
                            ).then(() => {
                              setExamFeedbacks((prev) => {
                                const statuses = Object.values(prev).map((f) => f.status);
                                if (statuses.every((s) => s === "done" || s === "canceled" || s === "error")) {
                                  setExamFeedbackStatus("done");
                                }
                                return prev;
                              });
                            });
                          }}
                        >
                          {isCanceled || isError ? "↺ Regenerate" : "Generate coaching"}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}

              {examGrading.overall_comments && (
                <div className="overall-box">
                  <div className="overall-title">Overall Comments</div>
                  <div className="overall-text">{examGrading.overall_comments}</div>
                </div>
              )}
            </div>

            <div style={{ marginTop: 14 }}>
              <button className="btn" onClick={() => {
                examFeedbackAbortRef.current?.abort();
                if (examPaperType === "paper1") { setExamAnswer(""); setExamGrading(null); setExamGradingStatus("idle"); setExamGradingError(""); setExamDebug(null); setExamFeedbacks({}); setExamFeedbackStatus("idle"); setExamTokenDebug(null); setExamStep("writing"); }
                else { setExamSelectedDocIds([]); setExamAnswer(""); setExamQuestionText(""); setExamGrading(null); setExamGradingStatus("idle"); setExamGradingError(""); setExamDebug(null); setExamFeedbacks({}); setExamFeedbackStatus("idle"); setExamTokenDebug(null); setExamStep("setup"); }
              }}>
                Try Again
              </button>
            </div>

            {/* Token debug */}
            {examTokenDebug && (
              <details className="dbg-details">
                <summary>▸ Token Usage</summary>
                <div className="dbg-body" style={{ fontSize: "0.78rem" }}>
                  {examTokenDebug.nCtx != null && <p className="dbg-line"><strong>Context window:</strong> {examTokenDebug.nCtx.toLocaleString()} tokens</p>}
                  <p className="dbg-line" style={{ marginBottom: 10 }}><strong>Grading call:</strong> ~{examTokenDebug.gradingPromptTokens.toLocaleString()} tokens</p>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.78rem" }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left" }}>
                        {["Criterion", "Prompt tokens", "Remaining", "Inference", "Status"].map((h) => (
                          <th key={h} style={{ padding: "3px 8px", color: "var(--muted)" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {examTokenDebug.calls.map((call) => (
                        <tr key={call.criterion} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td style={{ padding: "3px 8px" }}>{call.criterion}</td>
                          <td style={{ padding: "3px 8px" }}>{call.promptTokens != null ? `~${call.promptTokens.toLocaleString()}` : "—"}</td>
                          <td style={{ padding: "3px 8px" }}>{call.remainingBudget != null ? call.remainingBudget.toLocaleString() : "—"}</td>
                          <td style={{ padding: "3px 8px" }}>{call.inferenceSeconds != null ? formatSeconds(call.inferenceSeconds) : "—"}</td>
                          <td style={{ padding: "3px 8px" }}>{call.status}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p style={{ marginTop: 8, color: "var(--muted)" }}><strong>Total (session):</strong> ~{examTokenDebug.totalTokensEstimated.toLocaleString()} tokens</p>
                  <p style={{ marginTop: 4, fontSize: "0.72rem", color: "var(--muted)" }}>Counts use model tokenizer when available; otherwise words × 1.33.</p>
                </div>
              </details>
            )}

            {/* Router Debug */}
            <RouterDebugPanel debug={examDebug?.routing_debug} />

            {/* Grading debug */}
            {examDebug && (
              <details className="dbg-details">
                <summary>▸ Grading Debug</summary>
                <div className="dbg-body">
                  <p className="dbg-line"><strong>Inference:</strong> {formatSeconds(examDebug.inference_seconds)}</p>
                  {examDebug.prompt_tokens != null && <p className="dbg-line"><strong>Prompt tokens:</strong> ~{examDebug.prompt_tokens.toLocaleString()}</p>}
                  {examDebug.routing_reason && (
                    <p className="dbg-line">
                      <strong>Routing:</strong> {examDebug.routing_reason}
                      {examDebug.context_mode_effective && examDebug.context_mode_effective !== examGrading.context_mode && (
                        <span style={{ color: "var(--warn)", marginLeft: 8 }}>
                          (overrode selection → {examDebug.context_mode_effective})
                        </span>
                      )}
                    </p>
                  )}
                  {examDebug.top_semantic_score != null && (
                    <p className="dbg-line"><strong>Top semantic score:</strong> {examDebug.top_semantic_score.toFixed(3)}</p>
                  )}
                  {examDebug.retrieval_modes?.length > 0 && (
                    <p className="dbg-line"><strong>Retrieval modes:</strong> {examDebug.retrieval_modes.join(", ")}</p>
                  )}
                  {(examDebug.excerpts_per_doc || []).map((docExcerpts, di) =>
                    docExcerpts.length > 0 ? (
                      <React.Fragment key={di}>
                        <div className="dbg-section">Work {di + 1} — Retrieved Excerpts</div>
                        {docExcerpts.map((ex) => (
                          <div key={ex.excerpt_id} className="dbg-excerpt">
                            <p className="dbg-excerpt-meta">
                              <strong>{ex.excerpt_id}</strong>
                              {" | "}Score {Number(ex.score || 0).toFixed(3)}
                              {ex.semantic_score != null && ` | Semantic ${Number(ex.semantic_score).toFixed(3)}`}
                              {" | "}{ex.page_start === ex.page_end ? `p.${ex.page_start}` : `pp.${ex.page_start}-${ex.page_end}`}
                              {" | "}{ex.heading || "—"}
                            </p>
                            <p className="dbg-excerpt-text">{ex.text || ""}</p>
                          </div>
                        ))}
                      </React.Fragment>
                    ) : null
                  )}
                  <div className="dbg-section">Prompt Sent to Model</div>
                  <pre className="dbg-pre">{examDebug.prompt || ""}</pre>
                  <div className="dbg-section">Raw Model Output</div>
                  <pre className="dbg-pre">{examDebug.raw_output || ""}</pre>
                </div>
              </details>
            )}
          </div>
          );
        })()}
      </div>
      </div>
    </div>
    );
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // ROOT RENDER
  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <>
      <style>{CSS}</style>
      <div className="layout">
        {renderHeader()}
        <main className="page">
          {activePage === "dashboard" && renderDashboard()}
          {activePage === "library" && renderLibrary()}
          {activePage === "learn" && renderLearn()}
          {activePage === "exam" && renderExam()}
        </main>
      </div>
    </>
  );
}
