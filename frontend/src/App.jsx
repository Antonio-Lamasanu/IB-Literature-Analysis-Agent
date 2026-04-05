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

/* ── Misc ── */
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.spacer { margin-top: 14px; }

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

  // ── Chat State ────────────────────────────────────────────────────────────
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatStatus, setChatStatus] = useState("idle");
  const [chatError, setChatError] = useState("");
  const [chatDebug, setChatDebug] = useState(null);
  const [promptFormat, setPromptFormat] = useState(null);

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

  // ── Refs ──────────────────────────────────────────────────────────────────
  const startedAtRef = useRef(null);
  const intervalRef = useRef(null);
  const downloadUrlRef = useRef(null);

  // ── Derived ───────────────────────────────────────────────────────────────
  const isBusy = uploadStatus === "processing" || uploadStatus === "chunking";
  const isProcessing = uploadStatus === "processing";
  const isChunking = uploadStatus === "chunking";
  const isChatSending = chatStatus === "sending";
  const activeChatDocId = result?.documentId || selectedLearnDocId;
  const hasChatDocument = Boolean(activeChatDocId);
  const chatAvailable = llmAvailable && Boolean(activeChatDocId);
  const chunksAvailable = Boolean(result?.chunksAvailable);
  const modesAvailable = availableDocs.length > 0 || Boolean(result?.documentId);

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

  // ── Reset helpers ──────────────────────────────────────────────────────────
  const resetExam = () => {
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
    setExamDebug(null);
    setExamFeedbacks({});
    setExamFeedbackStatus("idle");
    setExamTokenDebug(null);
  };

  const resetResult = () => {
    if (downloadUrlRef.current) { URL.revokeObjectURL(downloadUrlRef.current); downloadUrlRef.current = null; }
    setResult(null);
    setSelectedLearnDocId(null);
    resetExam();
  };

  const resetChat = () => {
    setChatMessages([]);
    setChatInput("");
    setChatStatus("idle");
    setChatError("");
    setChatDebug(null);
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
  const onSendChat = async (e) => {
    e.preventDefault();
    if (!hasChatDocument || !chatAvailable) return;
    const userText = chatInput.trim();
    if (!userText) return;

    const nextMessages = [...chatMessages, { role: "user", content: userText }];
    setChatMessages(nextMessages);
    setChatInput("");
    setChatError("");
    setChatDebug(null);
    setChatStatus("sending");
    setChatMessages((prev) => [...prev, { role: "assistant", content: "", streaming: true }]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: activeChatDocId, messages: nextMessages, prompt_format: promptFormat }),
      });
      if (!response.ok) {
        setChatMessages((prev) => prev.slice(0, -1));
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
          if (ev.type === "token") {
            setChatMessages((prev) => { const msgs = [...prev]; const last = msgs[msgs.length - 1]; msgs[msgs.length - 1] = { ...last, content: last.content + ev.text }; return msgs; });
          } else if (ev.type === "done") {
            setChatDebug(ev.debug ? { ...ev.debug, prompt_mode: ev.prompt_mode } : null);
            setChatMessages((prev) => { const msgs = [...prev]; const last = msgs[msgs.length - 1]; msgs[msgs.length - 1] = { ...last, streaming: false, totalSeconds: Number(ev.debug?.total_seconds) || null }; return msgs; });
          } else if (ev.type === "error") {
            setChatMessages((prev) => prev.slice(0, -1));
            throw new Error(ev.detail || "Chat stream error.");
          }
        }
      }
      setChatStatus("idle");
    } catch (err) { setChatStatus("error"); setChatError(err?.message || "Unexpected error while chatting."); }
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
  };

  const startFeedbackStreaming = async (gradingData, studentAnswer, gradingPromptTokens) => {
    const criteria = gradingData.criteria || [];
    const initial = {};
    for (const c of criteria) initial[c.criterion] = { status: "idle", text: "", error: "", isOpen: false };
    setExamFeedbacks(initial);
    setExamFeedbackStatus("streaming");
    setExamTokenDebug({
      nCtx: null,
      gradingPromptTokens: gradingPromptTokens || 0,
      calls: criteria.map((c) => ({ criterion: c.criterion, label: c.label, promptTokens: null, remainingBudget: null, inferenceSeconds: null, status: "pending" })),
      totalTokensEstimated: gradingPromptTokens || 0,
    });

    for (const c of criteria) {
      setExamFeedbacks((prev) => ({ ...prev, [c.criterion]: { ...prev[c.criterion], status: "streaming", isOpen: true } }));
      setExamTokenDebug((prev) => prev ? { ...prev, calls: prev.calls.map((call) => call.criterion === c.criterion ? { ...call, status: "streaming" } : call) } : prev);
      const body = {
        paper_type: examPaperType, criterion: c.criterion, score: c.score, max_score: c.max_score,
        student_answer: studentAnswer, question: examQuestionText,
        passage_text: examPaperType === "paper1" ? examPassage : null,
        document_ids: examPaperType === "paper2" ? examSelectedDocIds : [],
        context_mode: examPaperType === "paper2" ? examContextMode : "chunks",
      };
      try {
        const response = await fetch(`${API_BASE_URL}/api/exam/criterion-feedback/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
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
            if (ev.type === "token") setExamFeedbacks((prev) => ({ ...prev, [c.criterion]: { ...prev[c.criterion], text: (prev[c.criterion]?.text || "") + ev.text } }));
            else if (ev.type === "done") doneDebug = ev.debug;
            else if (ev.type === "error") throw new Error(ev.detail || "Feedback stream error.");
          }
        }
        setExamFeedbacks((prev) => ({ ...prev, [c.criterion]: { ...prev[c.criterion], status: "done" } }));
        if (doneDebug) {
          setExamTokenDebug((prev) => {
            if (!prev) return prev;
            return {
              ...prev, nCtx: doneDebug.n_ctx ?? prev.nCtx,
              totalTokensEstimated: prev.totalTokensEstimated + (doneDebug.prompt_tokens || 0),
              calls: prev.calls.map((call) => call.criterion === c.criterion ? { ...call, promptTokens: doneDebug.prompt_tokens, remainingBudget: doneDebug.remaining_budget, inferenceSeconds: doneDebug.inference_seconds, status: "done" } : call),
            };
          });
        }
      } catch (err) {
        setExamFeedbacks((prev) => ({ ...prev, [c.criterion]: { ...prev[c.criterion], status: "error", error: err.message } }));
        setExamTokenDebug((prev) => prev ? { ...prev, calls: prev.calls.map((call) => call.criterion === c.criterion ? { ...call, status: "error" } : call) } : prev);
      }
    }
    setExamFeedbackStatus("done");
  };

  const onSubmitExamAnswer = async (e) => {
    e.preventDefault();
    if (!examQuestionText || !examAnswer.trim()) return;
    setExamGradingStatus("grading");
    setExamGradingError("");
    setExamFeedbacks({});
    setExamFeedbackStatus("idle");
    setExamTokenDebug(null);
    try {
      const studentAnswer = examAnswer.trim();
      const body = {
        paper_type: examPaperType, question: examQuestionText, student_answer: studentAnswer,
        document_ids: examPaperType === "paper2" ? examSelectedDocIds : [],
        context_mode: examPaperType === "paper2" ? examContextMode : "chunks",
        passage_text: examPaperType === "paper1" ? examPassage : null,
      };
      const response = await fetch(`${API_BASE_URL}/api/exam/submit-answer`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!response.ok) throw new Error(await parseErrorDetail(response, "Grading failed."));
      const data = await response.json();
      setExamGrading(data);
      setExamDebug(data.debug || null);
      setExamGradingStatus("done");
      setExamStep("done");
      startFeedbackStreaming(data, studentAnswer, data.debug?.prompt_tokens || 0);
    } catch (err) { setExamGradingStatus("error"); setExamGradingError(err?.message || "Unexpected error during grading."); }
  };

  // ── Navigation helpers ─────────────────────────────────────────────────────
  const goTo = (page) => {
    if (page === "exam") resetExam();
    setActivePage(page);
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER HELPERS
  // ─────────────────────────────────────────────────────────────────────────────

  // Model selector (used in header)
  const recommendedTier = systemInfo?.recommended_model;
  const recommendedModel = availableModels.find((m) => getModelTier(m.filename) === recommendedTier);

  const renderModelSelector = () => {
    if (availableModels.length === 0) return null;
    return (
      <div className="model-wrap">
        <select
          className="model-select"
          value={activeModel || ""}
          disabled={modelSwitching}
          onChange={(e) => onSelectModel(e.target.value)}
          title="Active model"
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
        {modelSwitching && <span className="model-status">Switching…</span>}
        {modelError && <span className="model-err">{modelError}</span>}
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
        {renderModelSelector()}
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
  const renderLearn = () => (
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
                  onClick={() => { setSelectedLearnDocId(doc.document_id); resetChat(); }}
                >
                  <div className="doc-pick-name">{doc.title || doc.filename}</div>
                  <div className="doc-pick-meta">{doc.author ? `${doc.author} · ` : ""}{doc.chunks_count} chunks</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Chat */}
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
                    </div>
                  ))
              }
            </div>

            <div>
              <div className="pill-row">
                <button className={`pill${promptFormat === null ? " active" : ""}`} disabled={isChatSending} onClick={() => setPromptFormat(null)}>Hybrid</button>
                <button className={`pill${promptFormat === "rag" ? " active" : ""}`} disabled={isChatSending} onClick={() => setPromptFormat("rag")}>RAG</button>
                <button className={`pill${promptFormat === "base_knowledge" ? " active" : ""}`} disabled={isChatSending} onClick={() => setPromptFormat("base_knowledge")}>Base Knowledge</button>
              </div>
              {promptFormat === "base_knowledge" && <p className="pill-hint">Answering from general knowledge — no document retrieval.</p>}
              {promptFormat === null && <p className="pill-hint">Auto-selects RAG or Base Knowledge based on retrieval confidence.</p>}
            </div>

            <form className="chat-form" onSubmit={onSendChat}>
              <textarea
                className="chat-textarea"
                placeholder="Ask something about this document…"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={isChatSending || isBusy}
              />
              <button className="btn" type="submit" disabled={isChatSending || isBusy || !chatInput.trim()}>
                {isChatSending ? "Sending…" : "Send"}
              </button>
            </form>
            {chatError && <div className="msg msg-error">{chatError}</div>}
          </>
        )}

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
    </div>
  );

  // ── Exam Page ──────────────────────────────────────────────────────────────
  const renderExam = () => (
    <div className="exam-layout">
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
              <textarea
                className="exam-textarea"
                placeholder="Write your answer here…"
                value={examAnswer}
                onChange={(e) => setExamAnswer(e.target.value)}
                disabled={examGradingStatus === "grading"}
              />
              <button className="btn" type="submit" disabled={!examAnswer.trim() || examGradingStatus === "grading"}>
                {examGradingStatus === "grading" ? "Grading…" : "Submit for Grading"}
              </button>
              {examGradingStatus === "error" && <div className="msg msg-error">{examGradingError}</div>}
            </form>
          </div>
        )}

        {/* Results step */}
        {examStep === "done" && examGrading && (
          <div>
            <div className="exam-results">
              <div className="score-heading">Total: {examGrading.total_score} / {examGrading.max_score}</div>
              <p className="context-badge">
                {examGrading.context_mode === "titles_only" ? "Graded with: base knowledge (no retrieval)" : "Graded with: RAG (excerpt retrieval)"}
              </p>

              {(examGrading.criteria || []).map((c) => {
                const fb = examFeedbacks[c.criterion] || { status: "idle", text: "", error: "", isOpen: false };
                const isStreaming = fb.status === "streaming";
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
                    {(fb.text.length > 0 || isStreaming) && (
                      <details
                        open={fb.isOpen}
                        onToggle={(e) => { const open = e.currentTarget.open; setExamFeedbacks((prev) => ({ ...prev, [c.criterion]: { ...prev[c.criterion], isOpen: open } })); }}
                        style={{ marginTop: 8 }}
                      >
                        <summary style={{ cursor: "pointer", fontSize: "0.82rem", color: "var(--muted)", fontWeight: 600 }}>
                          {isStreaming ? "Detailed coaching (streaming…)" : "Detailed coaching"}
                        </summary>
                        <div style={{ marginTop: 6, fontSize: "0.875rem", lineHeight: 1.6, whiteSpace: "pre-wrap", color: "var(--text-2)" }}>
                          {fb.text}
                          {isStreaming && <span className="cursor-block" />}
                        </div>
                      </details>
                    )}
                    {fb.status === "error" && <p style={{ fontSize: "0.78rem", color: "var(--error)", marginTop: 4 }}>Feedback unavailable: {fb.error}</p>}
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

            {/* Grading debug */}
            {examDebug && (
              <details className="dbg-details">
                <summary>▸ Grading Debug</summary>
                <div className="dbg-body">
                  <p className="dbg-line"><strong>Inference:</strong> {formatSeconds(examDebug.inference_seconds)}</p>
                  {examDebug.prompt_tokens != null && <p className="dbg-line"><strong>Prompt tokens:</strong> ~{examDebug.prompt_tokens.toLocaleString()}</p>}
                  <div className="dbg-section">Prompt Sent to Model</div>
                  <pre className="dbg-pre">{examDebug.prompt || ""}</pre>
                  <div className="dbg-section">Raw Model Output</div>
                  <pre className="dbg-pre">{examDebug.raw_output || ""}</pre>
                </div>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );

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
