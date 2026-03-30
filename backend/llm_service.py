from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama


class LLMServiceError(RuntimeError):
    pass


class LLMDisabledError(LLMServiceError):
    pass


class LLMNotConfiguredError(LLMServiceError):
    pass


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _parse_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw.strip()))
    except (TypeError, ValueError):
        return default


def _parse_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw.strip()))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    model_path: str
    n_ctx: int
    n_threads: int
    max_tokens: int
    temperature: float
    use_chat_api: bool = False  # use create_chat_completion instead of raw completion


@dataclass(frozen=True)
class LLMInferenceResult:
    reply: str
    final_prompt: str
    prompt_build_seconds: float
    inference_seconds: float
    history_turns_dropped: int = 0
    context_trimmed: bool = False


@dataclass
class _PromptBuildResult:
    prompt: str
    turns_dropped: int
    context_trimmed: bool


EXCERPT_BLOCK_RE = re.compile(r"(?ms)^\[(?:Excerpt|History Match) .*?(?=^\[(?:Excerpt|History Match) |\Z)")


def _estimate_text_tokens(text: str) -> int:
    words = len((text or "").split())
    if words <= 0:
        return 0
    return max(1, int(words * 1.33))


def load_llm_config_from_env() -> LLMConfig:
    cpu_count = os.cpu_count() or 4
    default_threads = max(1, cpu_count // 2)
    return LLMConfig(
        enabled=_parse_bool_env("LLM_ENABLED", False),
        model_path=(os.getenv("LLM_MODEL_PATH") or "").strip(),
        n_ctx=_parse_int_env("LLM_N_CTX", 2048, minimum=256),
        n_threads=_parse_int_env("LLM_N_THREADS", default_threads, minimum=1),
        max_tokens=_parse_int_env("LLM_MAX_TOKENS", 512, minimum=1),
        temperature=_parse_float_env("LLM_TEMPERATURE", 0.2, minimum=0.0),
    )


class LLMService:
    def __init__(self, config: LLMConfig | None = None):
        self._config = config or load_llm_config_from_env()
        self._llm: Llama | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def config(self) -> LLMConfig:
        return self._config

    def is_chat_available(self) -> bool:
        if not self._config.enabled:
            return False
        model_path = self._config.model_path.strip()
        if not model_path:
            return False
        return Path(model_path).exists()

    def generate_reply(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> str:
        return self.generate_reply_with_debug(
            document_text=document_text,
            messages=messages,
            max_history_messages=max_history_messages,
            title=title,
            author=author,
        ).reply

    def _run_chat_completion(
        self,
        llm: Llama,
        messages: list[dict[str, str]],
    ) -> tuple[str, float]:
        """Call create_chat_completion and return (reply, inference_seconds)."""
        t0 = time.perf_counter()
        with self._inference_lock:
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                top_k=40,
                min_p=0.05,
                repeat_penalty=1.1,
                stop=["\nQUESTION:", "\nCONVERSATION SO FAR:", "\nDOCUMENT EXCERPTS:"],
            )
        inference_seconds = time.perf_counter() - t0
        reply = output["choices"][0]["message"]["content"].strip()
        reply = re.sub(r"(?s)<think>.*?</think>\s*", "", reply).strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")
        return reply, inference_seconds

    def generate_reply_with_debug(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> LLMInferenceResult:
        normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise LLMServiceError("At least one chat message is required.")

        if not document_text or not document_text.strip():
            raise LLMServiceError("Document text is empty.")

        llm = self._get_llm()

        if self._config.use_chat_api:
            latest_user_index = max(
                (i for i, m in enumerate(normalized_messages) if m["role"] == "user"), default=-1
            )
            if latest_user_index < 0:
                raise LLMServiceError("At least one user message is required.")
            work_ref = f'"{title}" by {author}' if title else "this literary work"
            history_start = max(0, latest_user_index - max(1, max_history_messages))
            history = normalized_messages[history_start:latest_user_index]
            latest_question = normalized_messages[latest_user_index]["content"]
            chat_messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        f"You are an IB Literature analyst answering questions about "
                        f"{work_ref} using the provided document excerpts. "
                        "Keep answers short and concise."
                    ),
                },
                *history,
                {
                    "role": "user",
                    "content": f"Document excerpts:\n{document_text.strip()}\n\nQuestion: {latest_question}",
                },
            ]
            reply, inference_seconds = self._run_chat_completion(llm, chat_messages)
            return LLMInferenceResult(
                reply=reply,
                final_prompt=json.dumps(chat_messages, ensure_ascii=False, indent=2),
                prompt_build_seconds=0.0,
                inference_seconds=inference_seconds,
            )

        prompt_build_started = time.perf_counter()
        prompt = self._build_prompt_with_token_budget(
            llm=llm,
            document_context=document_text,
            messages=normalized_messages,
            max_history_messages=max_history_messages,
            title=title,
            author=author,
        )
        prompt_build_seconds = time.perf_counter() - prompt_build_started

        inference_started = time.perf_counter()
        with self._inference_lock:
            output = llm(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\nQUESTION:", "\nCONVERSATION SO FAR:", "\nDOCUMENT EXCERPTS:"],
            )
        inference_seconds = time.perf_counter() - inference_started

        reply = output["choices"][0]["text"].strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=reply,
            final_prompt=prompt,
            prompt_build_seconds=prompt_build_seconds,
            inference_seconds=inference_seconds,
        )

    def unload(self) -> None:
        """Release the loaded model from memory."""
        with self._load_lock:
            self._llm = None
        gc.collect()

    def extract_title_and_author(self, text_sample: str) -> tuple[str, str]:
        """Infer the title and author of a literary work from the first portion of its text.

        Returns a (title, author) tuple. Falls back to ("Unknown", "Unknown") on any failure.
        """
        prompt = (
            "Identify the title and author of this literary work.\n"
            "Reply in exactly this format:\n"
            "Title: <title>\n"
            "Author: <author>\n\n"
            "If you cannot determine the author, write Author: Unknown.\n"
            f"Text:\n{text_sample.strip()}\n\n"
            "Title:"
        )
        try:
            llm = self._get_llm()
            with self._inference_lock:
                output = llm(
                    prompt,
                    max_tokens=48,
                    temperature=0.0,
                    stop=["\n\n"],
                )
            raw = output["choices"][0]["text"].strip()
            # Prepend "Title:" since it was the stop token that opened the answer
            full = "Title:" + raw
            title_match = re.search(r"Title:\s*(.+)", full)
            author_match = re.search(r"Author:\s*(.+)", full)
            title = title_match.group(1).strip() if title_match else "Unknown"
            author = author_match.group(1).strip() if author_match else "Unknown"
            return (title or "Unknown", author or "Unknown")
        except Exception:
            return ("Unknown", "Unknown")

    def generate_raw_reply(self, prompt: str) -> "LLMInferenceResult":
        """Run inference on a pre-built prompt string with no context injection."""
        llm = self._get_llm()
        inference_started = time.perf_counter()
        with self._inference_lock:
            output = llm(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\nQUESTION:", "\nCONVERSATION SO FAR:", "\nDOCUMENT EXCERPTS:"],
            )
        inference_seconds = time.perf_counter() - inference_started
        reply = output["choices"][0]["text"].strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=reply,
            final_prompt=prompt,
            prompt_build_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def generate_raw_reply_stream(self, prompt: str) -> Generator[str, None, "LLMInferenceResult"]:
        """Stream tokens from a pre-built prompt with no context injection.

        Usage identical to generate_reply_stream — yield tokens, catch StopIteration for result.
        """
        llm = self._get_llm()
        inference_started = time.perf_counter()
        tokens: list[str] = []
        with self._inference_lock:
            stream = llm(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\nQUESTION:", "\nCONVERSATION SO FAR:", "\nDOCUMENT EXCERPTS:"],
                stream=True,
            )
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                if token:
                    tokens.append(token)
                    yield token

        inference_seconds = time.perf_counter() - inference_started
        full_reply = "".join(tokens).strip()
        if not full_reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=full_reply,
            final_prompt=prompt,
            prompt_build_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def generate_feedback_stream(self, prompt: str) -> Generator[str, None, "LLMInferenceResult"]:
        """Stream tokens for a feedback prompt with no stop tokens.

        Unlike generate_raw_reply_stream, this passes stop=[] so chat-specific stop
        sequences cannot truncate mid-sentence feedback output.
        """
        llm = self._get_llm()
        inference_started = time.perf_counter()
        tokens: list[str] = []
        with self._inference_lock:
            stream = llm(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=[],
                stream=True,
            )
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                if token:
                    tokens.append(token)
                    yield token

        inference_seconds = time.perf_counter() - inference_started
        full_reply = "".join(tokens).strip()
        if not full_reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=full_reply,
            final_prompt=prompt,
            prompt_build_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def count_tokens(self, text: str) -> int:
        """Return exact token count via the loaded model's tokenizer.

        Falls back to words × 1.33 estimate if the model is not yet loaded.
        """
        if self._llm is None:
            return max(1, int(len(text.split()) * 1.33))
        try:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return max(1, int(len(text.split()) * 1.33))

    def get_context_window_size(self) -> int:
        """Return the configured context window size (n_ctx)."""
        return self._config.n_ctx

    def generate_base_knowledge_reply_with_debug(
        self,
        *,
        title: str,
        author: str,
        session_messages: list[dict[str, str]],
        question: str,
    ) -> LLMInferenceResult:
        """base_knowledge inference with token-budget history trimming."""
        llm = self._get_llm()

        if self._config.use_chat_api:
            chat_messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        f'You are an IB Literature analyst answering questions about "{title}" '
                        f"by {author}. Keep answers short and concise."
                    ),
                },
                *list(session_messages),
                {"role": "user", "content": question},
            ]
            reply, inference_seconds = self._run_chat_completion(llm, chat_messages)
            return LLMInferenceResult(
                reply=reply,
                final_prompt=json.dumps(chat_messages, ensure_ascii=False, indent=2),
                prompt_build_seconds=0.0,
                inference_seconds=inference_seconds,
            )

        t0 = time.perf_counter()
        build_result = self._build_base_knowledge_prompt_with_budget(
            llm=llm, title=title, author=author,
            session_messages=session_messages, question=question,
        )
        prompt_build_seconds = time.perf_counter() - t0

        inference_started = time.perf_counter()
        with self._inference_lock:
            output = llm(
                build_result.prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\nQUESTION:", "\nCONVERSATION SO FAR:", "\nDOCUMENT EXCERPTS:"],
            )
        inference_seconds = time.perf_counter() - inference_started

        reply = output["choices"][0]["text"].strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=reply,
            final_prompt=build_result.prompt,
            prompt_build_seconds=prompt_build_seconds,
            inference_seconds=inference_seconds,
            history_turns_dropped=build_result.turns_dropped,
            context_trimmed=build_result.context_trimmed,
        )

    def _build_prompt_with_token_budget(
        self,
        *,
        llm: Llama,
        document_context: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> str:
        max_prompt_tokens = max(256, self._config.n_ctx - self._config.max_tokens - 32)
        context_text = document_context.strip()

        prompt = self._build_prompt(
            document_text=context_text,
            messages=messages,
            max_history_messages=max_history_messages,
            title=title,
            author=author,
        )
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        if prompt_tokens <= max_prompt_tokens:
            return prompt

        excerpt_blocks = self._split_excerpt_blocks(context_text)
        if len(excerpt_blocks) > 1:
            trimmed_context = context_text
            remaining_blocks = excerpt_blocks[:]
            for _ in range(min(2, len(remaining_blocks) - 1)):
                remaining_blocks.pop()
                trimmed_context = "\n\n".join(remaining_blocks).strip()
                prompt = self._build_prompt(
                    document_text=trimmed_context,
                    messages=messages,
                    max_history_messages=max_history_messages,
                    title=title,
                    author=author,
                )
                prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
                if prompt_tokens <= max_prompt_tokens:
                    return prompt
            context_text = trimmed_context

        if context_text:
            overflow_ratio = max_prompt_tokens / max(1, prompt_tokens)
            trim_chars = max(1_000, int(len(context_text) * max(0.55, min(0.95, overflow_ratio))))
            context_text = context_text[:trim_chars].rstrip()

        return self._build_prompt(
            document_text=context_text,
            messages=messages,
            max_history_messages=max_history_messages,
            title=title,
            author=author,
        )

    def _build_base_knowledge_prompt_with_budget(
        self,
        *,
        llm: Llama,
        title: str,
        author: str,
        session_messages: list[dict[str, str]],
        question: str,
    ) -> _PromptBuildResult:
        max_prompt_tokens = max(256, self._config.n_ctx - self._config.max_tokens - 32)

        def _build(msgs: list[dict[str, str]]) -> str:
            history_lines = [
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in msgs
            ]
            history = "\n".join(history_lines).strip() or "(none)"
            return (
                f"You are an IB Literature analyst answering a question about "
                f'"{title}" by {author}. Keep answers short and concise.\n'
                f"CONVERSATION SO FAR:\n{history}\n\n"
                f"QUESTION:\n{question}\n\n"
                "RESPONSE:"
            )

        msgs = list(session_messages)
        prompt = _build(msgs)
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))

        original_count = len(msgs)
        while prompt_tokens > max_prompt_tokens and len(msgs) >= 2:
            msgs = msgs[2:]
            prompt = _build(msgs)
            prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))

        turns_dropped = (original_count - len(msgs)) // 2
        # context_trimmed is True if pairs were dropped OR the prompt still overflows
        # (edge case: a single pair longer than the entire budget — loop exits unresolved)
        context_trimmed = turns_dropped > 0 or prompt_tokens > max_prompt_tokens
        return _PromptBuildResult(prompt=prompt, turns_dropped=turns_dropped, context_trimmed=context_trimmed)

    def generate_reply_stream(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> Generator[str, None, LLMInferenceResult]:
        """Stream reply tokens one-by-one; the generator's return value is LLMInferenceResult.

        Usage:
            gen = svc.generate_reply_stream(...)
            try:
                while True:
                    token = next(gen)
                    # send token to client
            except StopIteration as exc:
                result = exc.value  # LLMInferenceResult with timings and full reply
        """
        normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            raise LLMServiceError("At least one chat message is required.")
        if not document_text or not document_text.strip():
            raise LLMServiceError("Document text is empty.")

        llm = self._get_llm()
        prompt_build_started = time.perf_counter()
        prompt = self._build_prompt_with_token_budget(
            llm=llm,
            document_context=document_text,
            messages=normalized_messages,
            max_history_messages=max_history_messages,
            title=title,
            author=author,
        )
        prompt_build_seconds = time.perf_counter() - prompt_build_started

        inference_started = time.perf_counter()
        tokens: list[str] = []
        with self._inference_lock:
            stream = llm(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\nQUESTION:", "\nCONVERSATION SO FAR:", "\nDOCUMENT EXCERPTS:"],
                stream=True,
            )
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                if token:
                    tokens.append(token)
                    yield token

        inference_seconds = time.perf_counter() - inference_started
        full_reply = "".join(tokens).strip()
        if not full_reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=full_reply,
            final_prompt=prompt,
            prompt_build_seconds=prompt_build_seconds,
            inference_seconds=inference_seconds,
        )

    def estimate_context_token_budget(
        self,
        *,
        messages: list[dict[str, str]],
        max_history_messages: int,
    ) -> int:
        normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            return max(256, self._config.n_ctx - self._config.max_tokens - 160)

        latest_user_index = -1
        for idx in range(len(normalized_messages) - 1, -1, -1):
            if normalized_messages[idx]["role"] == "user":
                latest_user_index = idx
                break

        if latest_user_index < 0:
            return max(256, self._config.n_ctx - self._config.max_tokens - 160)

        history_window_start = max(0, latest_user_index - max(1, max_history_messages))
        history = normalized_messages[history_window_start:latest_user_index]
        history_lines = [
            f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
            for message in history
        ]
        latest_user_message = normalized_messages[latest_user_index]["content"]
        reserved_tokens = (
            120
            + _estimate_text_tokens("\n".join(history_lines))
            + _estimate_text_tokens(latest_user_message)
        )
        available = self._config.n_ctx - self._config.max_tokens - reserved_tokens
        return max(256, available)

    def _split_excerpt_blocks(self, document_context: str) -> list[str]:
        matches = [match.group(0).strip() for match in EXCERPT_BLOCK_RE.finditer(document_context or "")]
        if matches:
            return matches
        stripped = (document_context or "").strip()
        return [stripped] if stripped else []

    def _build_prompt(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> str:
        latest_user_index = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx]["role"] == "user":
                latest_user_index = idx
                break

        if latest_user_index < 0:
            raise LLMServiceError("At least one user message is required.")

        history_window_start = max(0, latest_user_index - max(1, max_history_messages))
        history = messages[history_window_start:latest_user_index]
        history_lines = []
        for message in history:
            speaker = "User" if message["role"] == "user" else "Assistant"
            history_lines.append(f"{speaker}: {message['content']}")

        conversation_history = "\n".join(history_lines).strip() or "(none)"
        latest_user_message = messages[latest_user_index]["content"]

        work_ref = f'"{title}" by {author}' if title else "this literary work"
        return (
            f"You are an IB Literature analyst answering a question about "
            f"{work_ref}, using the provided document excerpts. Keep answers short and concise.\n"
            f"DOCUMENT EXCERPTS:\n{document_text.strip()}\n\n"
            f"CONVERSATION SO FAR:\n{conversation_history}\n\n"
            f"QUESTION:\n{latest_user_message}\n\n"
            "RESPONSE:"
        )

    def _normalize_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _get_llm(self) -> Llama:
        if not self._config.enabled:
            raise LLMDisabledError("LLM is disabled. Set LLM_ENABLED=1 to enable chat.")

        model_path = self._resolve_model_path()
        with self._load_lock:
            if self._llm is None:
                try:
                    extra: dict = {}
                    if self._config.use_chat_api:
                        extra["chat_format"] = "chatml"
                    self._llm = Llama(
                        model_path=str(model_path),
                        n_ctx=self._config.n_ctx,
                        n_threads=self._config.n_threads,
                        verbose=False,
                        **extra,
                    )
                except Exception as exc:
                    raise LLMServiceError(f"Failed to load LLM model: {exc}") from exc
            return self._llm

    def _resolve_model_path(self) -> Path:
        model_path = self._config.model_path.strip()
        if not model_path:
            raise LLMNotConfiguredError(
                "LLM model path is not configured. Set LLM_MODEL_PATH to a local GGUF file."
            )
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise LLMNotConfiguredError(f"LLM model file does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise LLMNotConfiguredError(f"LLM model path is not a file: {resolved_path}")
        return resolved_path


class RemoteLLMService:
    """LLM service backed by an external OpenAI-compatible server (e.g. llama-server).

    Sends requests to /v1/chat/completions using proper chat message format,
    which is required for models like Qwen3.5 that use special chat templates.

    Start the server separately (see llama.cpp / unsloth docs) before use.
    """

    def __init__(
        self,
        server_url: str,
        model: str | None = None,
        max_tokens: int = 640,
        temperature: float = 0.6,
        n_ctx: int = 10000,
    ):
        self._server_url = server_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._n_ctx = n_ctx

    def is_chat_available(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self._server_url}/models",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def unload(self) -> None:
        """No-op: the remote server manages its own model lifecycle."""

    def _chat_completions(self, messages: list[dict[str, str]]) -> dict:
        payload: dict = {
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._model:
            payload["model"] = self._model

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._server_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMServiceError(f"LLM server returned {exc.code}: {body}") from exc
        except Exception as exc:
            raise LLMServiceError(f"Failed to reach LLM server at {self._server_url}: {exc}") from exc

    def _normalize(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        out = []
        for item in messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            out.append({"role": role, "content": content})
        return out

    def generate_reply_with_debug(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> LLMInferenceResult:
        normalized = self._normalize(messages)
        if not normalized:
            raise LLMServiceError("At least one chat message is required.")
        if not document_text or not document_text.strip():
            raise LLMServiceError("Document text is empty.")

        latest_user_index = max(
            (i for i, m in enumerate(normalized) if m["role"] == "user"), default=-1
        )
        if latest_user_index < 0:
            raise LLMServiceError("At least one user message is required.")

        work_ref = f'"{title}" by {author}' if title else "this literary work"
        system_msg = {
            "role": "system",
            "content": (
                f"You are an IB Literature analyst answering questions about {work_ref} "
                "using the provided document excerpts. Keep answers short and concise."
            ),
        }

        history_start = max(0, latest_user_index - max(1, max_history_messages))
        history = normalized[history_start:latest_user_index]

        latest_question = normalized[latest_user_index]["content"]
        user_msg = {
            "role": "user",
            "content": (
                f"Document excerpts:\n{document_text.strip()}\n\nQuestion: {latest_question}"
            ),
        }

        chat_messages = [system_msg] + history + [user_msg]

        t0 = time.perf_counter()
        response = self._chat_completions(chat_messages)
        inference_seconds = time.perf_counter() - t0

        reply = response["choices"][0]["message"]["content"].strip()
        # Strip Qwen thinking blocks if present (<think>...</think>)
        reply = re.sub(r"(?s)<think>.*?</think>\s*", "", reply).strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")

        return LLMInferenceResult(
            reply=reply,
            final_prompt=json.dumps(chat_messages, ensure_ascii=False, indent=2),
            prompt_build_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def generate_reply(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
        title: str = "",
        author: str = "",
    ) -> str:
        return self.generate_reply_with_debug(
            document_text=document_text,
            messages=messages,
            max_history_messages=max_history_messages,
            title=title,
            author=author,
        ).reply

    def generate_base_knowledge_reply_with_debug(
        self,
        *,
        title: str,
        author: str,
        session_messages: list[dict[str, str]],
        question: str,
    ) -> LLMInferenceResult:
        system_msg = {
            "role": "system",
            "content": (
                f'You are an IB Literature analyst answering questions about "{title}" '
                f"by {author}. Keep answers short and concise."
            ),
        }
        chat_messages = [system_msg] + list(session_messages) + [{"role": "user", "content": question}]

        t0 = time.perf_counter()
        response = self._chat_completions(chat_messages)
        inference_seconds = time.perf_counter() - t0

        reply = response["choices"][0]["message"]["content"].strip()
        reply = re.sub(r"(?s)<think>.*?</think>\s*", "", reply).strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")

        return LLMInferenceResult(
            reply=reply,
            final_prompt=json.dumps(chat_messages, ensure_ascii=False, indent=2),
            prompt_build_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def generate_raw_reply(self, prompt: str) -> LLMInferenceResult:
        t0 = time.perf_counter()
        response = self._chat_completions([{"role": "user", "content": prompt}])
        inference_seconds = time.perf_counter() - t0
        reply = response["choices"][0]["message"]["content"].strip()
        reply = re.sub(r"(?s)<think>.*?</think>\s*", "", reply).strip()
        if not reply:
            raise LLMServiceError("Model returned an empty response.")
        return LLMInferenceResult(
            reply=reply,
            final_prompt=prompt,
            prompt_build_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def extract_title_and_author(self, text_sample: str) -> tuple[str, str]:
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Identify the title and author of this literary work.\n"
                        "Reply in exactly this format:\nTitle: <title>\nAuthor: <author>\n\n"
                        f"Text:\n{text_sample.strip()}"
                    ),
                }
            ]
            response = self._chat_completions(messages)
            raw = response["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"(?s)<think>.*?</think>\s*", "", raw).strip()
            title_match = re.search(r"Title:\s*(.+)", raw)
            author_match = re.search(r"Author:\s*(.+)", raw)
            title = title_match.group(1).strip() if title_match else "Unknown"
            author = author_match.group(1).strip() if author_match else "Unknown"
            return (title or "Unknown", author or "Unknown")
        except Exception:
            return ("Unknown", "Unknown")

    def estimate_context_token_budget(
        self,
        *,
        messages: list[dict[str, str]],
        max_history_messages: int,
    ) -> int:
        # Remote server (llama.cpp) handles its own context window;
        # return a generous budget so retrieval fetches enough context.
        return 32_000

    def generate_reply_stream(self, **kwargs) -> Generator[str, None, LLMInferenceResult]:
        raise LLMServiceError(
            "Streaming is not yet supported for the remote LLM server. "
            "Use the non-streaming /api/chat endpoint instead."
        )
        yield  # make this a generator

    def generate_raw_reply_stream(self, prompt: str) -> Generator[str, None, LLMInferenceResult]:
        raise LLMServiceError(
            "Streaming is not yet supported for the remote LLM server. "
            "Use the non-streaming /api/chat endpoint instead."
        )
        yield  # make this a generator

    def generate_feedback_stream(self, prompt: str) -> Generator[str, None, LLMInferenceResult]:
        raise LLMServiceError(
            "Streaming is not yet supported for the remote LLM server. "
            "Use the non-streaming /api/chat endpoint instead."
        )
        yield  # make this a generator

    def count_tokens(self, text: str) -> int:
        """Estimate token count using words × 1.33 (no local tokenizer available)."""
        return max(1, int(len(text.split()) * 1.33))

    def get_context_window_size(self) -> int:
        """Return the configured context window size."""
        return self._n_ctx


_service_instance: LLMService | None = None
_service_instance_lock = threading.Lock()


def get_llm_service() -> LLMService | RemoteLLMService:
    global _service_instance
    with _service_instance_lock:
        if _service_instance is None:
            server_url = os.getenv("LLM_SERVER_URL", "").strip()
            if server_url:
                server_model = os.getenv("LLM_SERVER_MODEL", "").strip() or None
                max_tokens = _parse_int_env("LLM_MAX_TOKENS", 640, minimum=1)
                temperature = _parse_float_env("LLM_TEMPERATURE", 0.6, minimum=0.0)
                _service_instance = RemoteLLMService(
                    server_url=server_url,
                    model=server_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                _service_instance = LLMService()
        return _service_instance
