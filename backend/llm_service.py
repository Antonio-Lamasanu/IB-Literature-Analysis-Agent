from __future__ import annotations

import gc
import os
import re
import threading
import time
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


@dataclass(frozen=True)
class LLMInferenceResult:
    reply: str
    final_prompt: str
    prompt_build_seconds: float
    inference_seconds: float


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
    ) -> str:
        return self.generate_reply_with_debug(
            document_text=document_text,
            messages=messages,
            max_history_messages=max_history_messages,
        ).reply

    def generate_reply_with_debug(
        self,
        *,
        document_text: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
    ) -> LLMInferenceResult:
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
                stop=["\nUser:"],
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
                stop=["\nUser:"],
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

    def _build_prompt_with_token_budget(
        self,
        *,
        llm: Llama,
        document_context: str,
        messages: list[dict[str, str]],
        max_history_messages: int,
    ) -> str:
        max_prompt_tokens = max(256, self._config.n_ctx - self._config.max_tokens - 32)
        context_text = document_context.strip()

        prompt = self._build_prompt(
            document_text=context_text,
            messages=messages,
            max_history_messages=max_history_messages,
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

        return (
            "SYSTEM INSTRUCTIONS:\n"
            "- You must answer referencing the DOCUMENT CONTEXT below.\n"
            '- If the answer is not contained in the document, reply like this: "I could not find the exact answer, but based on the document " and give answer accordingly \n'
          #  '- If the answer is not in the document, say so and provide what context you can. \n'
           # "- Keep responses concise and factual.\n\n"
            f"DOCUMENT CONTEXT:\n{document_text.strip()}\n\n"
            f"RECENT CONVERSATION:\n{conversation_history}\n\n"
            f"LATEST USER QUESTION:\nUser: {latest_user_message}\n"
            "Assistant:"
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
                    self._llm = Llama(
                        model_path=str(model_path),
                        n_ctx=self._config.n_ctx,
                        n_threads=self._config.n_threads,
                        verbose=False,
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


_service_instance: LLMService | None = None
_service_instance_lock = threading.Lock()


def get_llm_service() -> LLMService:
    global _service_instance
    with _service_instance_lock:
        if _service_instance is None:
            _service_instance = LLMService()
        return _service_instance
