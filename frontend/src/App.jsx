import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function getFileStem(filename) {
  if (!filename) return "extracted";
  const lastDot = filename.lastIndexOf(".");
  return lastDot > 0 ? filename.slice(0, lastDot) : filename;
}

function parseDownloadFilename(contentDispositionHeader, fallbackName) {
  if (!contentDispositionHeader) return fallbackName;

  const utf8Match = contentDispositionHeader.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    return decodeURIComponent(utf8Match[1]);
  }

  const basicMatch = contentDispositionHeader.match(/filename="?([^";]+)"?/i);
  return basicMatch?.[1] || fallbackName;
}

async function parseErrorDetail(response, fallback = "Request failed.") {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    try {
      const data = await response.clone().json();
      return data?.detail || fallback;
    } catch {
      const text = await response.text();
      return text || fallback;
    }
  }

  const text = await response.text();
  return text || fallback;
}

function formatSeconds(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "0.000s";
  return `${numeric.toFixed(3)}s`;
}

function toApiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path}`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatStatus, setChatStatus] = useState("idle");
  const [chatError, setChatError] = useState("");
  const [chatDebug, setChatDebug] = useState(null);

  // Exam mode state
  const [activeMode, setActiveMode] = useState("learn");
  const [examPaperType, setExamPaperType] = useState("paper1");
  const [examStep, setExamStep] = useState("setup");       // setup|writing|done
  const [examPassage, setExamPassage] = useState("");
  const [examQuestionText, setExamQuestionText] = useState("");
  const [examAllDocs, setExamAllDocs] = useState([]);
  const [examDocsStatus, setExamDocsStatus] = useState("idle"); // idle|loading|ready|error
  const [examDocsError, setExamDocsError] = useState("");
  const [examSelectedDocIds, setExamSelectedDocIds] = useState([]);
  const [examAnswer, setExamAnswer] = useState("");
  const [examP2Questions, setExamP2Questions] = useState([]);       // [{id, text}]
  const [examContextMode, setExamContextMode] = useState("chunks"); // chunks|titles_only
  const [examGrading, setExamGrading] = useState(null);
  const [examGradingStatus, setExamGradingStatus] = useState("idle"); // idle|grading|done|error
  const [examGradingError, setExamGradingError] = useState("");
  const [examDebug, setExamDebug] = useState(null);

  // Pre-loaded docs and LLM status (fetched on mount)
  const [availableDocs, setAvailableDocs] = useState([]);
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [selectedLearnDocId, setSelectedLearnDocId] = useState(null);

  const startedAtRef = useRef(null);
  const intervalRef = useRef(null);
  const downloadUrlRef = useRef(null);

  const isBusy = status === "processing" || status === "chunking";
  const isProcessing = status === "processing";
  const isChunking = status === "chunking";
  const isChatSending = chatStatus === "sending";
  const activeChatDocId = result?.documentId || selectedLearnDocId;
  const hasChatDocument = Boolean(activeChatDocId);
  const modesAvailable = availableDocs.length > 0 || Boolean(result?.documentId);
  const chatAvailable = llmAvailable && Boolean(activeChatDocId);
  const chunksAvailable = Boolean(result?.chunksAvailable);

  useEffect(() => {
    if (!isBusy) return undefined;

    intervalRef.current = window.setInterval(() => {
      if (!startedAtRef.current) return;
      const elapsedMs = Date.now() - startedAtRef.current;
      setElapsedSeconds(Math.floor(elapsedMs / 1000));
    }, 200);

    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isBusy]);

  useEffect(() => {
    return () => {
      if (downloadUrlRef.current) {
        URL.revokeObjectURL(downloadUrlRef.current);
        downloadUrlRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE_URL}/api/documents`).then((r) => r.ok ? r.json() : []),
      fetch(`${API_BASE_URL}/api/status`).then((r) => r.ok ? r.json() : {}),
    ]).then(([docs, statusData]) => {
      setAvailableDocs((docs || []).filter((d) => d.chunks_available));
      setLlmAvailable(Boolean(statusData?.chat_available));
    }).catch(() => {});
  }, []);

  const statusText = useMemo(() => {
    if (status === "processing") return "Processing PDF and generating chunks...";
    if (status === "chunking") return "Regenerating chunks...";
    if (status === "success") return "Done";
    if (status === "error") return "Failed";
    return "Ready";
  }, [status]);

  const resetExam = () => {
    setActiveMode("learn");
    setExamPaperType("paper1");
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
  };

  const resetResult = () => {
    if (downloadUrlRef.current) {
      URL.revokeObjectURL(downloadUrlRef.current);
      downloadUrlRef.current = null;
    }
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

  const onFileChange = (event) => {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setError("");
    resetResult();
    resetChat();
  };

  const onSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError("Please choose a PDF file first.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("File must end with .pdf");
      return;
    }

    setError("");
    setStatus("processing");
    setElapsedSeconds(0);
    startedAtRef.current = Date.now();
    resetResult();
    resetChat();

    const formData = new FormData();
    formData.append("file", file);
    const endpoint = "/api/process-pdf";
    const fallbackName = `${getFileStem(file.name)}.txt`;

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const detail = await parseErrorDetail(response, "Failed to process PDF.");
        throw new Error(detail);
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      downloadUrlRef.current = downloadUrl;
      const mode = response.headers.get("X-Processing-Mode");
      const pages = response.headers.get("X-Pages");
      const chars = response.headers.get("X-Text-Chars");
      const chunksCount = response.headers.get("X-Chunks-Count");
      const chunksAvailableHeader = response.headers.get("X-Chunks-Available");
      const chunksDownloadUrl = response.headers.get("X-Chunks-Download-Url");
      const chunkSchemaVersion = response.headers.get("X-Chunk-Schema-Version");
      const documentId = response.headers.get("X-Document-Id");
      const documentFilename = response.headers.get("X-Document-Filename");
      const chatAvailableHeader = response.headers.get("X-Chat-Available");
      const filename = parseDownloadFilename(response.headers.get("Content-Disposition"), fallbackName);

      setResult({
        mode,
        pages,
        chars,
        chunksCount,
        chunksAvailable: chunksAvailableHeader === "1",
        chunksDownloadUrl: chunksDownloadUrl ? toApiUrl(chunksDownloadUrl) : "",
        chunkSchemaVersion: chunkSchemaVersion || "",
        filename,
        downloadUrl,
        documentId,
        documentFilename,
        chatAvailable: chatAvailableHeader === "1",
      });
      if (chatAvailableHeader === "1") setLlmAvailable(true);
      // Refresh available docs so newly uploaded doc appears in learn-mode picker
      fetch(`${API_BASE_URL}/api/documents`)
        .then((r) => r.ok ? r.json() : [])
        .then((docs) => setAvailableDocs((docs || []).filter((d) => d.chunks_available)))
        .catch(() => {});
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setError(err?.message || "Unexpected error while processing PDF.");
    } finally {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (startedAtRef.current) {
        const elapsedMs = Date.now() - startedAtRef.current;
        setElapsedSeconds(Math.floor(elapsedMs / 1000));
      }
      startedAtRef.current = null;
    }
  };

  const onGenerateChunks = async () => {
    if (!result?.documentId || isBusy) return;

    setError("");
    setStatus("chunking");
    setElapsedSeconds(0);
    startedAtRef.current = Date.now();

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/documents/${encodeURIComponent(result.documentId)}/generate-chunks`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        const detail = await parseErrorDetail(response, "Failed to generate chunks.");
        throw new Error(detail);
      }

      const data = await response.json();
      setResult((previous) => {
        if (!previous) return previous;
        return {
          ...previous,
          chunksAvailable: Boolean(data?.chunks_available),
          chunksCount: data?.chunks_count ?? null,
          chunkTargetTokens: data?.chunk_target_tokens ?? null,
          chunkOverlapTokens: data?.chunk_overlap_tokens ?? null,
          chunkSchemaVersion: data?.chunk_schema_version || "",
          chunksDownloadUrl: data?.chunks_download_url ? toApiUrl(data.chunks_download_url) : "",
          chunksPath: data?.chunks_path || "",
          chunkMetaPath: data?.chunk_meta_path || "",
        };
      });
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setError(err?.message || "Unexpected error while generating chunks.");
    } finally {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (startedAtRef.current) {
        const elapsedMs = Date.now() - startedAtRef.current;
        setElapsedSeconds(Math.floor(elapsedMs / 1000));
      }
      startedAtRef.current = null;
    }
  };

  const onSendChat = async (event) => {
    event.preventDefault();

    if (!hasChatDocument) {
      setChatError("No processed TXT document available yet.");
      return;
    }
    if (!chatAvailable) {
      setChatError("Chat is unavailable. Check backend LLM configuration.");
      return;
    }

    const userMessageText = chatInput.trim();
    if (!userMessageText) return;

    const nextMessages = [...chatMessages, { role: "user", content: userMessageText }];
    setChatMessages(nextMessages);
    setChatInput("");
    setChatError("");
    setChatDebug(null);
    setChatStatus("sending");

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          document_id: activeChatDocId,
          messages: nextMessages,
        }),
      });

      if (!response.ok) {
        const detail = await parseErrorDetail(response, "Failed to get chat response.");
        throw new Error(detail);
      }

      const data = await response.json();
      const reply = String(data?.reply || "").trim();
      if (!reply) {
        throw new Error("Chat endpoint returned an empty reply.");
      }

      setChatDebug(data?.debug || null);
      setChatMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: reply,
          totalSeconds: Number(data?.debug?.total_seconds) || null,
        },
      ]);
      setChatStatus("idle");
    } catch (err) {
      setChatStatus("error");
      setChatError(err?.message || "Unexpected error while chatting.");
    }
  };

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
      } catch (err) {
        setExamDocsStatus("error");
        setExamDocsError(err?.message || "Failed to load Paper 1.");
      }
    } else {
      setExamDocsStatus("loading");
      setExamSelectedDocIds([]);
      setExamStep("setup");
      try {
        const [docsRes, qRes] = await Promise.all([
          fetch(`${API_BASE_URL}/api/documents`),
          fetch(`${API_BASE_URL}/api/exam/paper2/questions`),
        ]);
        if (!docsRes.ok) throw new Error(await parseErrorDetail(docsRes, "Failed to load documents."));
        if (!qRes.ok) throw new Error(await parseErrorDetail(qRes, "Failed to load questions."));
        const docsData = await docsRes.json();
        const qData = await qRes.json();
        setExamAllDocs((docsData || []).filter((d) => d.chunks_available));
        setExamP2Questions(qData.questions || []);
        setExamDocsStatus("ready");
      } catch (err) {
        setExamDocsStatus("error");
        setExamDocsError(err?.message || "Failed to load data.");
      }
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

  const onSubmitExamAnswer = async (event) => {
    event.preventDefault();
    if (!examQuestionText || !examAnswer.trim()) return;
    setExamGradingStatus("grading");
    setExamGradingError("");

    try {
      const body = {
        paper_type: examPaperType,
        question: examQuestionText,
        student_answer: examAnswer.trim(),
        document_ids: examPaperType === "paper2" ? examSelectedDocIds : [],
        context_mode: examPaperType === "paper2" ? examContextMode : "chunks",
      };
      const response = await fetch(`${API_BASE_URL}/api/exam/submit-answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const detail = await parseErrorDetail(response, "Grading failed.");
        throw new Error(detail);
      }
      const data = await response.json();
      setExamGrading(data);
      setExamDebug(data.debug || null);
      setExamGradingStatus("done");
      setExamStep("done");
    } catch (err) {
      setExamGradingStatus("error");
      setExamGradingError(err?.message || "Unexpected error during grading.");
    }
  };

  return (
    <>
      <style>{`
        :root {
          color-scheme: light;
          --bg: #f4f7fb;
          --card: #ffffff;
          --text: #14213d;
          --muted: #5c677d;
          --accent: #0f766e;
          --accent-soft: #99f6e4;
          --error: #b42318;
          --border: #d6deeb;
          --shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
        }

        * { box-sizing: border-box; }

        body {
          margin: 0;
          font-family: "Segoe UI", "Helvetica Neue", sans-serif;
          background: radial-gradient(circle at top right, #d8f3ff 0%, var(--bg) 52%);
          color: var(--text);
        }

        .app {
          min-height: 100vh;
          display: grid;
          place-items: center;
          padding: 24px;
        }

        .card {
          width: min(1400px, 100%);
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 18px;
          box-shadow: var(--shadow);
          padding: 28px;
        }

        .title {
          margin: 0 0 10px;
          font-size: 1.8rem;
          letter-spacing: 0.3px;
        }

        .subtitle {
          margin: 0 0 24px;
          color: var(--muted);
        }

        .form-row {
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
          margin-bottom: 18px;
        }

        .file-input {
          flex: 1;
          min-width: 260px;
          padding: 10px;
          border: 1px solid var(--border);
          border-radius: 10px;
          background: #fff;
        }

        .btn {
          border: none;
          border-radius: 10px;
          padding: 10px 18px;
          font-weight: 600;
          cursor: pointer;
          background: var(--accent);
          color: #fff;
        }

        .btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .status {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 10px;
          color: var(--muted);
          font-size: 0.95rem;
        }

        .progress {
          width: 100%;
          height: 14px;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: linear-gradient(90deg, #eef2ff 0%, #f5f3ff 100%);
          overflow: hidden;
          position: relative;
        }

        .progress.indeterminate::before {
          content: "";
          position: absolute;
          left: -32%;
          width: 32%;
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, var(--accent-soft), var(--accent));
          animation: slide 1.1s ease-in-out infinite;
        }

        .progress.complete::before {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg, var(--accent-soft), var(--accent));
        }

        @keyframes slide {
          0% { left: -35%; }
          100% { left: 103%; }
        }

        .error {
          margin-top: 12px;
          color: var(--error);
          font-weight: 600;
        }

        .result {
          margin-top: 20px;
          padding: 16px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: #f8fbff;
        }

        .meta {
          margin: 6px 0;
          color: #1f2937;
        }

        .result-actions {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 14px;
        }

        .download-link {
          display: inline-block;
          text-decoration: none;
          background: #0b63f6;
          color: #fff;
          border-radius: 8px;
          padding: 9px 14px;
          font-weight: 600;
        }

        .secondary-btn {
          background: #134e4a;
        }

        .chat-layout {
          margin-top: 24px;
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 16px;
          align-items: start;
        }

        .chat {
          padding: 16px;
          border: 1px solid var(--border);
          border-radius: 12px;
          background: #fff;
        }

        .debug-panel {
          padding: 16px;
          border: 1px solid var(--border);
          border-radius: 12px;
          background: #fbfdff;
        }

        .chat-title {
          margin: 0 0 10px;
          font-size: 1.1rem;
        }

        .debug-title {
          margin: 0 0 10px;
          font-size: 1.1rem;
        }

        .chat-note {
          margin: 0;
          color: var(--muted);
        }

        .debug-note {
          margin: 0;
          color: var(--muted);
        }

        .chat-list {
          margin-top: 12px;
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 12px;
          max-height: 300px;
          overflow-y: auto;
          background: #fafcff;
        }

        .chat-empty {
          margin: 0;
          color: var(--muted);
        }

        .chat-message {
          margin: 0 0 12px;
          padding: 10px 12px;
          border-radius: 10px;
          line-height: 1.4;
          white-space: pre-wrap;
        }

        .chat-message-block {
          margin-bottom: 12px;
        }

        .chat-message-timing {
          margin: 0 0 8px;
          color: var(--muted);
          font-size: 0.88rem;
          text-align: center;
        }

        .chat-message.user {
          background: #e5f7f3;
          border: 1px solid #bfe9df;
        }

        .chat-message.assistant {
          background: #f3f5ff;
          border: 1px solid #dbe0ff;
        }

        .chat-form {
          margin-top: 12px;
          display: flex;
          gap: 10px;
          align-items: flex-end;
        }

        .chat-input {
          flex: 1;
          min-height: 78px;
          resize: vertical;
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 10px;
          font: inherit;
        }

        .chat-send {
          border: none;
          border-radius: 10px;
          padding: 10px 16px;
          font-weight: 600;
          cursor: pointer;
          background: #0b63f6;
          color: #fff;
          min-width: 88px;
        }

        .chat-send:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .chat-error {
          margin-top: 8px;
          color: var(--error);
          font-weight: 600;
        }

        .debug-summary {
          margin: 0 0 12px;
          padding: 12px;
          border: 1px solid var(--border);
          border-radius: 10px;
          background: #fff;
        }

        .debug-line {
          margin: 0 0 6px;
          color: #1f2937;
        }

        .debug-section-title {
          margin: 16px 0 8px;
          font-size: 0.98rem;
        }

        .debug-prompt {
          margin: 0;
          padding: 12px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #0f172a;
          color: #e2e8f0;
          font-size: 0.86rem;
          line-height: 1.45;
          white-space: pre-wrap;
          max-height: 320px;
          overflow: auto;
        }

        .debug-excerpts {
          display: grid;
          gap: 10px;
        }

        .debug-excerpt {
          border: 1px solid var(--border);
          border-radius: 10px;
          background: #fff;
          padding: 12px;
        }

        .debug-excerpt-meta {
          margin: 0 0 8px;
          color: #334155;
          font-size: 0.92rem;
        }

        .debug-excerpt-text {
          margin: 0;
          color: #1f2937;
          white-space: pre-wrap;
          line-height: 1.45;
        }

        @media (max-width: 640px) {
          .card { padding: 20px; }
          .title { font-size: 1.5rem; }
          .chat-layout { grid-template-columns: 1fr; }
          .chat-form { flex-direction: column; align-items: stretch; }
          .chat-send { width: 100%; }
        }

        .mode-toggle {
          display: flex;
          gap: 8px;
          margin: 20px 0 0;
        }

        .mode-btn {
          border: none;
          border-radius: 10px;
          padding: 10px 20px;
          font-weight: 600;
          cursor: pointer;
          background: #e2e8f0;
          color: var(--text);
        }

        .mode-btn.active {
          background: var(--accent);
          color: #fff;
        }

        .exam-panel {
          margin-top: 24px;
          padding: 20px;
          border: 1px solid var(--border);
          border-radius: 12px;
          background: #fff;
        }

        .exam-title {
          margin: 0 0 6px;
          font-size: 1.1rem;
        }

        .exam-note {
          margin: 0 0 14px;
          color: var(--muted);
          font-size: 0.95rem;
        }

        .exam-question-box {
          background: #fffbe6;
          border: 1px solid #f59e0b;
          border-radius: 10px;
          padding: 14px 16px;
          margin: 12px 0;
          font-size: 1.05rem;
          line-height: 1.5;
        }

        .exam-answer-area {
          width: 100%;
          min-height: 200px;
          resize: vertical;
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 10px;
          font: inherit;
          margin: 10px 0;
        }

        .exam-results {
          margin-top: 20px;
          padding: 16px;
          border-radius: 12px;
          border: 2px solid var(--accent);
          background: #f0fdf9;
        }

        .exam-score-heading {
          font-size: 1.4rem;
          margin: 0 0 14px;
          color: var(--accent);
        }

        .criterion-row {
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 12px 14px;
          margin-bottom: 10px;
          background: #fff;
        }

        .criterion-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }

        .criterion-label {
          font-weight: 600;
        }

        .criterion-score-badge {
          background: var(--accent);
          color: #fff;
          border-radius: 20px;
          padding: 3px 12px;
          font-weight: 700;
          font-size: 0.95rem;
          white-space: nowrap;
        }

        .criterion-feedback {
          margin: 0;
          color: #374151;
          line-height: 1.45;
        }

        .exam-overall {
          margin-top: 14px;
          padding: 12px 14px;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #f8f9ff;
        }

        .exam-overall-title {
          margin: 0 0 6px;
          font-size: 0.98rem;
          font-weight: 600;
        }

        .exam-overall-text {
          margin: 0;
          color: #374151;
          line-height: 1.45;
        }

        .paper-select {
          padding: 8px 12px;
          border: 1px solid var(--border);
          border-radius: 10px;
          font: inherit;
          background: #fff;
          margin-bottom: 14px;
        }

        .paper-select:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .doc-picker-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 10px;
          margin: 12px 0;
        }

        .doc-card {
          border: 2px solid var(--border);
          border-radius: 10px;
          padding: 12px;
          cursor: pointer;
          background: #fff;
          transition: border-color 0.15s;
        }

        .doc-card:hover:not(.disabled) {
          border-color: var(--accent);
        }

        .doc-card.selected {
          border-color: var(--accent);
          background: #f0fdf9;
        }

        .doc-card.disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }

        .doc-card-name {
          font-weight: 600;
          font-size: 0.95rem;
          margin: 0 0 4px;
        }

        .doc-card-meta {
          color: var(--muted);
          font-size: 0.82rem;
          margin: 0;
        }
      `}</style>

      <main className="app">
        <section className="card">
          <h1 className="title">PDF to TXT Extractor</h1>
          <p className="subtitle">
            Upload a PDF. The backend extracts TXT and persists retrieval chunks in the same step so chat is ready immediately.
          </p>

          <form onSubmit={onSubmit}>
            <div className="form-row">
              <input
                className="file-input"
                type="file"
                accept="application/pdf,.pdf"
                onChange={onFileChange}
                disabled={isBusy}
              />
              <button className="btn" type="submit" disabled={isBusy || !file}>
                {isProcessing ? "Processing..." : "Upload & Process"}
              </button>
            </div>
          </form>

          <div className="status">
            <span>Status: {statusText}</span>
            <span>Elapsed: {elapsedSeconds}s</span>
          </div>

          <div className={`progress ${status === "processing" ? "indeterminate" : ""} ${status === "success" ? "complete" : ""}`} />

          {error ? <p className="error">{error}</p> : null}

          {result ? (
            <div className="result">
              {result.mode ? <p className="meta"><strong>Mode:</strong> {result.mode}</p> : null}
              {result.pages ? <p className="meta"><strong>Pages:</strong> {result.pages}</p> : null}
              {result.chars ? <p className="meta"><strong>Characters:</strong> {result.chars}</p> : null}
              {result.documentId ? <p className="meta"><strong>Document ID:</strong> {result.documentId}</p> : null}
              {result.documentFilename ? <p className="meta"><strong>Document Filename:</strong> {result.documentFilename}</p> : null}
              <p className="meta"><strong>Chunks Available:</strong> {chunksAvailable ? "Yes" : "No"}</p>
              {result.chunksCount !== null && result.chunksCount !== undefined ? (
                <p className="meta"><strong>Chunks:</strong> {result.chunksCount}</p>
              ) : null}
              {result.chunkTargetTokens ? (
                <p className="meta"><strong>Chunk Target Tokens:</strong> {result.chunkTargetTokens}</p>
              ) : null}
              {result.chunkOverlapTokens ? (
                <p className="meta"><strong>Chunk Overlap Tokens:</strong> {result.chunkOverlapTokens}</p>
              ) : null}
              {result.chunkSchemaVersion ? (
                <p className="meta"><strong>Chunk Schema:</strong> {result.chunkSchemaVersion}</p>
              ) : null}
              {result.chunksPath ? (
                <p className="meta"><strong>Chunks Path:</strong> {result.chunksPath}</p>
              ) : null}
              <div className="result-actions">
                <a className="download-link" href={result.downloadUrl} download={result.filename}>
                  Download TXT
                </a>
                {result.documentId && !chunksAvailable ? (
                  <button className="btn secondary-btn" type="button" onClick={onGenerateChunks} disabled={isBusy || !result.documentId}>
                    {isChunking ? "Regenerating..." : "Retry Chunk Generation"}
                  </button>
                ) : null}
                {result.chunksDownloadUrl ? (
                  <a className="download-link" href={result.chunksDownloadUrl} download>
                    Download Chunks JSON
                  </a>
                ) : null}
              </div>
            </div>
          ) : null}

          {modesAvailable ? (
            <div className="mode-toggle">
              <button
                className={`mode-btn${activeMode === "learn" ? " active" : ""}`}
                type="button"
                onClick={() => setActiveMode("learn")}
              >
                Learn Mode
              </button>
              <button
                className={`mode-btn${activeMode === "exam" ? " active" : ""}`}
                type="button"
                onClick={() => setActiveMode("exam")}
              >
                Exam Mode
              </button>
            </div>
          ) : null}

          {activeMode === "learn" ? (
          <div className="chat-layout">
            <section className="chat">
              <h2 className="chat-title">Document Chat</h2>

              {/* No freshly-uploaded doc: show picker from registry */}
              {!result?.documentId ? (
                <div style={{ marginBottom: 16 }}>
                  <p className="chat-note" style={{ marginBottom: 10 }}>
                    Select a work to chat about:
                  </p>
                  {availableDocs.length === 0 ? (
                    <p className="chat-note">No works found. Upload a PDF above to get started.</p>
                  ) : (
                    <div className="doc-picker-grid">
                      {availableDocs.map((doc) => (
                        <div
                          key={doc.document_id}
                          className={`doc-card${selectedLearnDocId === doc.document_id ? " selected" : ""}`}
                          onClick={() => {
                            setSelectedLearnDocId(doc.document_id);
                            setChatMessages([]);
                            setChatDebug(null);
                          }}
                        >
                          <p className="doc-card-name">{doc.title || doc.filename}</p>
                          <p className="doc-card-meta">
                            {doc.author ? `${doc.author} · ` : ""}{doc.chunks_count} chunks
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : null}

              {hasChatDocument && !chatAvailable ? (
                <p className="chat-note">Chat is unavailable. Configure backend LLM env vars and a valid model path.</p>
              ) : null}

              {hasChatDocument && chatAvailable ? (
                <>
                  <div className="chat-list">
                    {chatMessages.length === 0 ? (
                      <p className="chat-empty">No messages yet. Ask a question about the extracted document.</p>
                    ) : (
                      chatMessages.map((message, index) => (
                        <div key={`${message.role}-${index}`} className="chat-message-block">
                          {message.role === "assistant" && message.totalSeconds ? (
                            <p className="chat-message-timing">
                              Response time: {formatSeconds(message.totalSeconds)}
                            </p>
                          ) : null}
                          <p className={`chat-message ${message.role}`}>
                            <strong>{message.role === "user" ? "You" : "Assistant"}:</strong> {message.content}
                          </p>
                        </div>
                      ))
                    )}
                  </div>

                  <form className="chat-form" onSubmit={onSendChat}>
                    <textarea
                      className="chat-input"
                      placeholder="Ask something about this document..."
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      disabled={isChatSending || isBusy}
                    />
                    <button className="chat-send" type="submit" disabled={isChatSending || isBusy || !chatInput.trim()}>
                      {isChatSending ? "Sending..." : "Send"}
                    </button>
                  </form>
                  {chatError ? <p className="chat-error">{chatError}</p> : null}
                </>
              ) : null}
            </section>

            <aside className="debug-panel">
              <h2 className="debug-title">Chat Debug</h2>
              {!hasChatDocument ? (
                <p className="debug-note">Select or upload a work first to inspect prompts and retrieval.</p>
              ) : null}
              {hasChatDocument && !chatAvailable ? (
                <p className="debug-note">Chat debug becomes available once backend chat is configured.</p>
              ) : null}
              {hasChatDocument && chatAvailable && !chatDebug ? (
                <p className="debug-note">Send a prompt to see the final prompt, timings, and retrieved excerpts.</p>
              ) : null}

              {chatDebug ? (
                <>
                  <div className="debug-summary">
                    <p className="debug-line"><strong>Total time:</strong> {formatSeconds(chatDebug.total_seconds)}</p>
                    <p className="debug-line"><strong>Retrieval mode:</strong> {chatDebug.retrieval_mode || "unknown"}</p>
                    <p className="debug-line"><strong>Retrieval:</strong> {formatSeconds(chatDebug.timing?.retrieval_seconds)}</p>
                    <p className="debug-line"><strong>Prompt build:</strong> {formatSeconds(chatDebug.timing?.prompt_build_seconds)}</p>
                    <p className="debug-line"><strong>Inference:</strong> {formatSeconds(chatDebug.timing?.inference_seconds)}</p>
                  </div>

                  <h3 className="debug-section-title">Final Prompt</h3>
                  <pre className="debug-prompt">{chatDebug.final_prompt || ""}</pre>

                  <h3 className="debug-section-title">Retrieved Excerpts</h3>
                  <div className="debug-excerpts">
                    {(chatDebug.retrieved_excerpts || []).length === 0 ? (
                      <p className="debug-note">No scored excerpts were returned for this request.</p>
                    ) : (
                      chatDebug.retrieved_excerpts.map((excerpt) => {
                        const pageLabel =
                          excerpt.page_start === excerpt.page_end
                            ? `Page ${excerpt.page_start}`
                            : `Pages ${excerpt.page_start}-${excerpt.page_end}`;

                        return (
                          <article
                            key={`${excerpt.excerpt_id}-${excerpt.page_start}-${excerpt.page_end}`}
                            className="debug-excerpt"
                          >
                            <p className="debug-excerpt-meta">
                              <strong>Excerpt {excerpt.excerpt_id}</strong> | Score {Number(excerpt.score || 0).toFixed(3)} | {pageLabel} | {excerpt.heading || "UNKNOWN"}
                            </p>
                            <p className="debug-excerpt-text">{excerpt.text || ""}</p>
                          </article>
                        );
                      })
                    )}
                  </div>
                </>
              ) : null}
            </aside>
          </div>
          ) : null}

          {activeMode === "exam" && modesAvailable ? (
            <section className="exam-panel">
              <h2 className="exam-title">IB Literature Exam</h2>

              {/* ── Step: initial paper selection + start ── */}
              {examStep === "setup" && examDocsStatus === "idle" ? (
                <div>
                  <div style={{ marginBottom: 14 }}>
                    <label htmlFor="paper-select" style={{ fontWeight: 600, marginRight: 8 }}>Paper:</label>
                    <select
                      id="paper-select"
                      className="paper-select"
                      value={examPaperType}
                      onChange={(e) => setExamPaperType(e.target.value)}
                    >
                      <option value="paper1">Paper 1 — Guided Literary Analysis (max 20)</option>
                      <option value="paper2">Paper 2 — Comparative Essay (max 40)</option>
                    </select>
                  </div>
                  <button className="btn" type="button" onClick={() => onEnterExam(examPaperType)}>
                    Start Exam
                  </button>
                </div>
              ) : null}

              {/* Loading state */}
              {examDocsStatus === "loading" ? (
                <p className="exam-note">Loading exam data…</p>
              ) : null}

              {/* Error state */}
              {examDocsStatus === "error" ? (
                <p className="chat-error">{examDocsError}</p>
              ) : null}

              {/* ── Step: document picker (Paper 2 only) ── */}
              {examStep === "setup" && examDocsStatus === "ready" && examPaperType === "paper2" ? (
                <div>
                  <h3 style={{ margin: "0 0 8px", fontSize: "1rem" }}>Select Two Works</h3>
                  <p className="exam-note">Choose exactly 2 works to compare in your essay.</p>
                  {examAllDocs.length === 0 ? (
                    <p className="exam-note">No documents with chunks available. Upload and process PDFs first.</p>
                  ) : (
                    <div className="doc-picker-grid">
                      {examAllDocs.map((doc) => {
                        const isSelected = examSelectedDocIds.includes(doc.document_id);
                        const isDisabled = !isSelected && examSelectedDocIds.length >= 2;
                        return (
                          <div
                            key={doc.document_id}
                            className={`doc-card${isSelected ? " selected" : ""}${isDisabled ? " disabled" : ""}`}
                            onClick={() => !isDisabled && onToggleDocSelection(doc.document_id)}
                          >
                            <p className="doc-card-name">{doc.title || doc.filename}</p>
                            <p className="doc-card-meta">
                              {doc.author ? `${doc.author} · ` : ""}{doc.chunks_count} chunks
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <button
                    className="btn"
                    type="button"
                    onClick={onConfirmDocSelection}
                    disabled={examSelectedDocIds.length !== 2}
                    style={{ marginTop: 12 }}
                  >
                    Continue →
                  </button>
                </div>
              ) : null}

              {/* ── Step: writing ── */}
              {examStep === "writing" ? (
                <div>
                  {/* Paper 1: show passage */}
                  {examPaperType === "paper1" && examPassage ? (
                    <div className="exam-passage">{examPassage}</div>
                  ) : null}

                  {/* Paper 2: show selected works + context mode toggle */}
                  {examPaperType === "paper2" ? (
                    <>
                      <div className="selected-works-tags">
                        {examSelectedDocIds.map((docId) => {
                          const doc = examAllDocs.find((d) => d.document_id === docId);
                          return (
                            <span key={docId} className="work-tag">
                              {doc ? (doc.title || doc.filename) : docId}
                            </span>
                          );
                        })}
                      </div>
                      <div style={{ marginBottom: 14 }}>
                        <span style={{ fontWeight: 600, marginRight: 12 }}>Context:</span>
                        <label style={{ marginRight: 16, cursor: "pointer" }}>
                          <input
                            type="radio"
                            name="contextMode"
                            value="chunks"
                            checked={examContextMode === "chunks"}
                            onChange={() => setExamContextMode("chunks")}
                            style={{ marginRight: 4 }}
                          />
                          Chunk context
                        </label>
                        <label style={{ cursor: "pointer" }}>
                          <input
                            type="radio"
                            name="contextMode"
                            value="titles_only"
                            checked={examContextMode === "titles_only"}
                            onChange={() => setExamContextMode("titles_only")}
                            style={{ marginRight: 4 }}
                          />
                          Titles only
                        </label>
                      </div>
                    </>
                  ) : null}

                  {/* Question */}
                  {examQuestionText ? (
                    <div className="exam-question-box" style={{ whiteSpace: "pre-wrap" }}>
                      {examQuestionText}
                    </div>
                  ) : null}

                  <form onSubmit={onSubmitExamAnswer}>
                    <p className="exam-note" style={{ marginTop: 14 }}>
                      Write your response below. No AI assistance is available during this step.
                    </p>
                    <textarea
                      className="exam-answer-area"
                      placeholder="Write your answer here…"
                      value={examAnswer}
                      onChange={(e) => setExamAnswer(e.target.value)}
                      disabled={examGradingStatus === "grading"}
                    />
                    <button
                      className="btn"
                      type="submit"
                      disabled={!examAnswer.trim() || examGradingStatus === "grading"}
                    >
                      {examGradingStatus === "grading" ? "Grading your response…" : "Submit for Grading"}
                    </button>
                    {examGradingStatus === "error" ? (
                      <p className="chat-error">{examGradingError}</p>
                    ) : null}
                  </form>
                </div>
              ) : null}

              {/* ── Step: results ── */}
              {examStep === "done" && examGrading ? (
                <div className="exam-results">
                  <h3 className="exam-score-heading">
                    Total: {examGrading.total_score} / {examGrading.max_score}
                  </h3>
                  {examGrading.context_mode === "titles_only" ? (
                    <p style={{ color: "var(--muted)", fontSize: "0.85rem", margin: "0 0 12px" }}>
                      Graded with: titles only
                    </p>
                  ) : (
                    <p style={{ color: "var(--muted)", fontSize: "0.85rem", margin: "0 0 12px" }}>
                      Graded with: chunk context
                    </p>
                  )}

                  {(examGrading.criteria || []).map((c) => (
                    <div key={c.criterion} className="criterion-row">
                      <div className="criterion-header">
                        <span className="criterion-label">{c.criterion} — {c.label}</span>
                        <span className="criterion-score-badge">{c.score} / {c.max_score}</span>
                      </div>
                      <p className="criterion-feedback">{c.feedback}</p>
                    </div>
                  ))}

                  {examGrading.overall_comments ? (
                    <div className="exam-overall">
                      <p className="exam-overall-title">Overall Comments</p>
                      <p className="exam-overall-text">{examGrading.overall_comments}</p>
                    </div>
                  ) : null}

                  <div style={{ marginTop: 14 }}>
                    <button
                      className="btn"
                      type="button"
                      onClick={() => {
                        if (examPaperType === "paper1") {
                          // Re-enter Paper 1 (same question)
                          setExamAnswer("");
                          setExamGrading(null);
                          setExamGradingStatus("idle");
                          setExamGradingError("");
                          setExamDebug(null);
                          setExamStep("writing");
                        } else {
                          // Go back to doc picker for Paper 2
                          setExamSelectedDocIds([]);
                          setExamAnswer("");
                          setExamQuestionText("");
                          setExamGrading(null);
                          setExamGradingStatus("idle");
                          setExamGradingError("");
                          setExamDebug(null);
                          setExamStep("setup");
                        }
                      }}
                    >
                      Try Again
                    </button>
                  </div>

                  {examDebug ? (
                    <details style={{ marginTop: 16 }}>
                      <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "0.95rem", color: "var(--muted)" }}>
                        Debug Info
                      </summary>
                      <div style={{ marginTop: 10 }}>
                        <div className="debug-summary">
                          <p className="debug-line"><strong>Inference time:</strong> {formatSeconds(examDebug.inference_seconds)}</p>
                        </div>
                        <h3 className="debug-section-title">Prompt Sent to Model</h3>
                        <pre className="debug-prompt">{examDebug.prompt || ""}</pre>
                        <h3 className="debug-section-title">Raw Model Output</h3>
                        <pre className="debug-prompt">{examDebug.raw_output || ""}</pre>
                      </div>
                    </details>
                  ) : null}
                </div>
              ) : null}
            </section>
          ) : null}

        </section>
      </main>
    </>
  );
}
