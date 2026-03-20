import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CHUNK_SCHEMA_VERSION = "simple-novel-v1"

PAGE_MARKER_RE = re.compile(r"^\s*===\s*PAGE\s+(\d+)\s*===\s*$", flags=re.IGNORECASE)
HEADING_MARKER_RE = re.compile(r"^\s*===\s*HEADING:\s*(.+?)\s*===\s*$", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CHAPTER_HEADING_RE = re.compile(
    r"^CHAPTER\s+([0-9]{1,3}|[IVXLCDM]{1,10})(?:\s*[:.\-]\s*.*)?$",
    flags=re.IGNORECASE,
)
BARE_CHAPTER_HEADING_RE = re.compile(r"^([0-9]{1,3}|[IVXLCDM]{1,10})\.?$", flags=re.IGNORECASE)
QUOTED_TEXT_RE = re.compile(r'["\u201c\u201d](.+?)["\u201c\u201d]', flags=re.DOTALL)
CAPITALIZED_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
TITLED_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Sir|Lady|Lord|Captain|Colonel|Professor|Father|Mother|Aunt|Uncle)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

INTRO_HEADINGS = {"INTRODUCTION", "FOREWORD", "PREFACE", "ABOUT THE AUTHOR"}
BACK_MATTER_HEADINGS = {"ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS", "APPENDIX", "NOTES", "INDEX", "AFTERWORD"}
DESCRIPTION_HINTS = {
    "bright",
    "cold",
    "dark",
    "deep",
    "empty",
    "gray",
    "green",
    "large",
    "little",
    "long",
    "old",
    "quiet",
    "red",
    "silent",
    "small",
    "soft",
    "still",
    "warm",
    "white",
    "window",
    "room",
    "door",
    "street",
    "garden",
    "house",
    "light",
    "eyes",
    "face",
}
NARRATION_HINTS = {
    "went",
    "came",
    "looked",
    "turned",
    "walked",
    "asked",
    "told",
    "said",
    "thought",
    "knew",
    "began",
    "stopped",
    "moved",
    "felt",
    "stood",
    "saw",
    "heard",
}
NAME_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "Author",
    "Book",
    "Books",
    "But",
    "Chapter",
    "Contents",
    "Download",
    "Ebook",
    "EBooks",
    "Fire",
    "Free",
    "Great",
    "He",
    "Her",
    "His",
    "I",
    "If",
    "In",
    "Introduction",
    "It",
    "Miss",
    "Mr",
    "Mrs",
    "Ms",
    "My",
    "No",
    "Not",
    "Of",
    "On",
    "Or",
    "Page",
    "Pages",
    "Planet",
    "She",
    "The",
    "Their",
    "There",
    "They",
    "This",
    "We",
    "When",
    "While",
    "Who",
    "Why",
    "You",
    "Your",
}


@dataclass
class ParagraphUnit:
    doc_id: str
    unit_id: str
    page: int
    heading: str
    chapter_index: int | None
    chapter_label: str | None
    section_type: str
    index_in_chapter: int
    text: str
    token_estimate: int
    char_start: int
    char_end: int
    segment_key: str


@dataclass
class ChunkParams:
    target_tokens: int = 500
    overlap_tokens: int = 100
    min_tokens: int = 200
    max_tokens: int = 800
    include_raw: bool = False


@dataclass
class ChunkRecord:
    unit_id: str
    doc_id: str
    chapter_label: str | None
    section_type: str
    position: dict
    content: dict
    metadata: dict
    raw_units: list[dict] | None = None

    @property
    def token_estimate(self) -> int:
        return int(self.content.get("token_estimate", 0))

    def to_dict(self, include_raw: bool = False) -> dict:
        result = {
            "unit_id": self.unit_id,
            "doc_id": self.doc_id,
            "position": self.position,
            "content": self.content,
            "metadata": self.metadata,
        }
        if include_raw:
            result["_raw_units"] = self.raw_units or []
        return result


def _safe_stem(filename_stem: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", filename_stem or "").strip("-").lower()
    return safe or "document"


def build_doc_id(filename_stem: str, text: str) -> str:
    safe = _safe_stem(filename_stem)
    digest = hashlib.sha1(f"{safe}\n{text}".encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


def _roman_to_int(value: str) -> int | None:
    value = (value or "").strip().upper()
    if not value:
        return None

    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(value):
        current = roman_values.get(char)
        if current is None:
            return None
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if total > 0 else None


def _chapter_token_to_int(token: str) -> int | None:
    token = (token or "").strip().rstrip(".")
    if not token:
        return None
    if token.isdigit():
        value = int(token)
        return value if value > 0 else None
    return _roman_to_int(token)


def _normalize_compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _parse_heading_context(raw_heading: str, *, allow_bare_chapter: bool = False) -> tuple[str, int | None, str | None]:
    compact = _normalize_compact_text(raw_heading)
    if not compact:
        return "UNKNOWN", None, None

    upper = compact.upper().rstrip(":")
    if upper in {"TABLE OF CONTENTS"}:
        return "CONTENTS", None, None
    if upper in INTRO_HEADINGS or upper in BACK_MATTER_HEADINGS or upper == "CONTENTS":
        return upper, None, None

    chapter_match = CHAPTER_HEADING_RE.match(upper)
    if chapter_match:
        chapter_index = _chapter_token_to_int(chapter_match.group(1))
        if chapter_index is not None:
            label = f"CHAPTER {chapter_index}"
            return label, chapter_index, label

    if allow_bare_chapter:
        bare_match = BARE_CHAPTER_HEADING_RE.match(upper)
        if bare_match:
            chapter_index = _chapter_token_to_int(bare_match.group(1))
            if chapter_index is not None:
                label = f"CHAPTER {chapter_index}"
                return label, chapter_index, label

    return upper, None, None


def normalize_heading(heading: str) -> str:
    return _parse_heading_context(heading, allow_bare_chapter=True)[0]


def derive_section_type(heading: str, page: int, chapter_index: int | None = None) -> str:
    normalized = normalize_heading(heading)
    if normalized == "CONTENTS":
        return "toc"
    if normalized in INTRO_HEADINGS:
        return "intro"
    if normalized in BACK_MATTER_HEADINGS:
        return "back_matter"
    if chapter_index is not None or normalized.startswith("CHAPTER"):
        return "body"
    if normalized == "UNKNOWN":
        return "front_matter" if page <= 5 else "unknown"
    return "unknown"


def _extract_inline_heading(line: str, *, allow_bare_chapter: bool = False) -> tuple[str, int | None, str | None] | None:
    normalized, chapter_index, chapter_label = _parse_heading_context(
        line,
        allow_bare_chapter=allow_bare_chapter,
    )
    if normalized in INTRO_HEADINGS or normalized == "CONTENTS" or normalized in BACK_MATTER_HEADINGS:
        return normalized, chapter_index, chapter_label
    if chapter_index is not None:
        return normalized, chapter_index, chapter_label
    return None


def is_toc_page(page_text: str) -> bool:
    if not page_text.strip():
        return False

    non_empty_lines = [line.strip() for line in page_text.splitlines() if line.strip()][:12]
    for line in non_empty_lines:
        heading_match = HEADING_MARKER_RE.match(line)
        candidate = heading_match.group(1).strip() if heading_match else line
        candidate = re.sub(r"\s+", " ", candidate).strip().lower().rstrip(":.")
        if candidate in {"contents", "table of contents"}:
            return True
    return False


def _is_noise_paragraph(paragraph_text: str) -> bool:
    stripped = paragraph_text.strip()
    if len(stripped) < 6:
        return True

    letters = sum(1 for char in stripped if char.isalpha())
    alphabetic_ratio = letters / max(1, len(stripped))
    word_count = len(stripped.split())
    return alphabetic_ratio < 0.4 and word_count <= 3


def _split_pages(marked_text: str) -> list[tuple[int, list[str]]]:
    pages: list[tuple[int, list[str]]] = []
    current_page = 1
    current_lines: list[str] = []

    for line in marked_text.splitlines():
        page_match = PAGE_MARKER_RE.match(line.strip())
        if page_match:
            if current_lines:
                pages.append((current_page, current_lines))
            current_page = int(page_match.group(1))
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines or not pages:
        pages.append((current_page, current_lines))

    return pages


def _build_segment_key(
    *,
    chapter_index: int | None,
    chapter_label: str | None,
    heading: str,
    section_type: str,
) -> str:
    if chapter_index is not None:
        return f"chapter:{chapter_index}"
    if chapter_label:
        return f"chapter-label:{chapter_label}"
    if heading and heading != "UNKNOWN":
        return f"heading:{heading}"
    return f"section:{section_type}"


def parse_units_from_marked_text(
    marked_text: str,
    doc_id: str = "document",
    drop_noise_paras: bool = True,
) -> list[ParagraphUnit]:
    pages = _split_pages(marked_text)
    units: list[ParagraphUnit] = []
    next_unit_id = 1
    last_non_toc_heading = "UNKNOWN"
    last_non_toc_chapter_index: int | None = None
    last_non_toc_chapter_label: str | None = None
    segment_counts: Counter[str] = Counter()
    char_cursor = 0

    for page_no, lines in pages:
        in_toc_page = is_toc_page("\n".join(lines))
        page_heading = "CONTENTS" if in_toc_page else last_non_toc_heading
        page_chapter_index = None if in_toc_page else last_non_toc_chapter_index
        page_chapter_label = None if in_toc_page else last_non_toc_chapter_label
        non_empty_line_index = 0
        paragraph_lines: list[str] = []

        def flush_paragraph() -> None:
            nonlocal next_unit_id, paragraph_lines, char_cursor
            paragraph_text = "\n".join(paragraph_lines).strip()
            paragraph_lines = []
            if not paragraph_text:
                return
            if drop_noise_paras and _is_noise_paragraph(paragraph_text):
                return

            section_type = derive_section_type(page_heading, page_no, page_chapter_index)
            segment_key = _build_segment_key(
                chapter_index=page_chapter_index,
                chapter_label=page_chapter_label,
                heading=page_heading,
                section_type=section_type,
            )
            segment_counts[segment_key] += 1

            char_start = char_cursor
            char_end = char_start + len(paragraph_text)
            units.append(
                ParagraphUnit(
                    doc_id=doc_id,
                    unit_id=f"{doc_id}_u{next_unit_id:05d}",
                    page=page_no,
                    heading=page_heading,
                    chapter_index=page_chapter_index,
                    chapter_label=page_chapter_label,
                    section_type=section_type,
                    index_in_chapter=segment_counts[segment_key],
                    text=paragraph_text,
                    token_estimate=estimate_tokens(paragraph_text),
                    char_start=char_start,
                    char_end=char_end,
                    segment_key=segment_key,
                )
            )
            next_unit_id += 1
            char_cursor = char_end + 2

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                flush_paragraph()
                continue

            non_empty_line_index += 1

            heading_match = HEADING_MARKER_RE.match(stripped)
            if heading_match:
                flush_paragraph()
                if in_toc_page:
                    continue
                page_heading, page_chapter_index, page_chapter_label = _parse_heading_context(
                    heading_match.group(1),
                    allow_bare_chapter=True,
                )
                last_non_toc_heading = page_heading
                last_non_toc_chapter_index = page_chapter_index
                last_non_toc_chapter_label = page_chapter_label
                continue

            if not in_toc_page and not paragraph_lines:
                inline_heading = _extract_inline_heading(
                    stripped,
                    allow_bare_chapter=non_empty_line_index <= 3,
                )
                if inline_heading:
                    flush_paragraph()
                    page_heading, page_chapter_index, page_chapter_label = inline_heading
                    last_non_toc_heading = page_heading
                    last_non_toc_chapter_index = page_chapter_index
                    last_non_toc_chapter_label = page_chapter_label
                    continue

            paragraph_lines.append(stripped)

        flush_paragraph()

    return units


def _split_text_by_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    max_words = max(1, int(max_tokens / 1.33))
    return [" ".join(words[i : i + max_words]).strip() for i in range(0, len(words), max_words)]


def _split_oversized_paragraph(text: str, max_tokens: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if estimate_tokens(stripped) <= max_tokens:
        return [stripped]

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(stripped) if sentence.strip()]
    if len(sentences) <= 1:
        return _split_text_by_words(stripped, max_tokens)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if current and estimate_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    final_chunks: list[str] = []
    for chunk in chunks:
        if estimate_tokens(chunk) <= max_tokens:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_text_by_words(chunk, max_tokens))

    return [chunk for chunk in final_chunks if chunk.strip()]


def _expand_units(units: list[ParagraphUnit], max_tokens: int) -> list[ParagraphUnit]:
    expanded: list[ParagraphUnit] = []
    for unit in units:
        if unit.token_estimate <= max_tokens:
            expanded.append(unit)
            continue

        pieces = _split_oversized_paragraph(unit.text, max_tokens)
        piece_cursor = unit.char_start
        for piece_index, piece in enumerate(pieces, start=1):
            piece_end = piece_cursor + len(piece)
            expanded.append(
                ParagraphUnit(
                    doc_id=unit.doc_id,
                    unit_id=f"{unit.unit_id}.p{piece_index}",
                    page=unit.page,
                    heading=unit.heading,
                    chapter_index=unit.chapter_index,
                    chapter_label=unit.chapter_label,
                    section_type=unit.section_type,
                    index_in_chapter=unit.index_in_chapter,
                    text=piece,
                    token_estimate=estimate_tokens(piece),
                    char_start=piece_cursor,
                    char_end=piece_end,
                    segment_key=unit.segment_key,
                )
            )
            piece_cursor = piece_end + 1

    return expanded


def _sanitize_params(params: ChunkParams) -> ChunkParams:
    min_tokens = max(1, params.min_tokens)
    max_tokens = max(min_tokens, params.max_tokens)
    target_tokens = min(max(params.target_tokens, min_tokens), max_tokens)
    overlap_tokens = max(0, min(params.overlap_tokens, max_tokens - 1))
    return ChunkParams(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        include_raw=params.include_raw,
    )


def _estimate_dialogue_ratio(text: str) -> float:
    word_count = len(text.split())
    if word_count <= 0:
        return 0.0

    quoted_words = sum(len(fragment.split()) for fragment in QUOTED_TEXT_RE.findall(text))
    dialogue_line_words = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(('"', "\u201c", "'", "\u2018", "-", "\u2014")) or re.match(r"^[A-Z][a-z]+:\s", stripped):
            dialogue_line_words += len(stripped.split())

    ratio = min(word_count, quoted_words + int(dialogue_line_words * 0.7)) / word_count
    return round(max(0.0, min(1.0, ratio)), 2)


def _extract_character_mentions(text: str) -> list[str]:
    candidates: Counter[str] = Counter()
    for titled_match in TITLED_NAME_RE.finditer(text):
        candidates[titled_match.group(1).strip()] += 2

    for name_match in CAPITALIZED_NAME_RE.finditer(text):
        candidate = name_match.group(0).strip()
        words = candidate.split()
        if any(word in NAME_STOPWORDS for word in words):
            continue
        if len(words) == 1 and words[0].upper() == words[0]:
            continue
        candidates[candidate] += 1

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1], len(item[0].split()), item[0]),
    )
    return [name for name, score in ranked if score >= 2][:6]


def _derive_unit_type(text: str, *, has_dialogue: bool, dialogue_ratio: float) -> str:
    if dialogue_ratio >= 0.6:
        return "dialogue"
    if has_dialogue and dialogue_ratio >= 0.2:
        return "mixed"

    tokens = [token.lower() for token in re.findall(r"\b[\w']+\b", text)]
    description_score = sum(1 for token in tokens if token in DESCRIPTION_HINTS) + text.count(",")
    narration_score = sum(1 for token in tokens if token in NARRATION_HINTS)
    return "description" if description_score > narration_score + 1 else "narration"


def _build_chunk_record(
    *,
    chunk_number: int,
    chunk_index_in_chapter: int,
    window: list[ParagraphUnit],
    absolute_percent: float,
    include_raw: bool,
) -> ChunkRecord:
    chunk_text = "\n\n".join(item.text for item in window).strip()
    word_count = len(chunk_text.split())
    token_estimate = estimate_tokens(chunk_text) if chunk_text else 0
    dialogue_ratio = _estimate_dialogue_ratio(chunk_text)
    has_dialogue = dialogue_ratio >= 0.12 or bool(QUOTED_TEXT_RE.search(chunk_text))
    character_mentions = _extract_character_mentions(chunk_text)
    unit_type = _derive_unit_type(chunk_text, has_dialogue=has_dialogue, dialogue_ratio=dialogue_ratio)

    raw_units = None
    if include_raw:
        raw_units = [
            {
                "unit_id": item.unit_id,
                "page": item.page,
                "heading": item.heading,
                "text": item.text,
            }
            for item in window
        ]

    return ChunkRecord(
        unit_id=f"{window[0].doc_id}_chunk_{chunk_number:05d}",
        doc_id=window[0].doc_id,
        chapter_label=window[0].chapter_label,
        section_type=window[0].section_type,
        position={
            "chapter": window[0].chapter_index,
            "index_in_chapter": chunk_index_in_chapter,
            "absolute_percent": round(max(0.0, min(1.0, absolute_percent)), 4),
            "page_start": window[0].page,
            "page_end": window[-1].page,
        },
        content={
            "text": chunk_text,
            "word_count": word_count,
            "token_estimate": token_estimate,
        },
        metadata={
            "unit_type": unit_type,
            "character_mentions": character_mentions,
            "has_dialogue": has_dialogue,
            "dialogue_ratio": dialogue_ratio,
        },
        raw_units=raw_units,
    )


def build_chunks(units: list[ParagraphUnit], params: ChunkParams) -> list[ChunkRecord]:
    if not units:
        return []

    normalized_params = _sanitize_params(params)
    expanded_units = _expand_units(units, normalized_params.max_tokens)
    total_chars = max((unit.char_end for unit in expanded_units), default=0)
    chunks: list[ChunkRecord] = []
    chunk_number = 1
    chunk_indexes_by_segment: Counter[str] = Counter()

    index = 0
    total_units = len(expanded_units)
    while index < total_units:
        segment_key = expanded_units[index].segment_key
        segment_end = index
        while segment_end < total_units and expanded_units[segment_end].segment_key == segment_key:
            segment_end += 1

        cursor = index
        while cursor < segment_end:
            window: list[ParagraphUnit] = []
            token_total = 0
            next_cursor = cursor

            while next_cursor < segment_end:
                candidate = expanded_units[next_cursor]
                if window and candidate.page != window[-1].page and token_total >= normalized_params.min_tokens:
                    break
                if window and token_total + candidate.token_estimate > normalized_params.max_tokens:
                    break

                window.append(candidate)
                token_total += candidate.token_estimate
                next_cursor += 1

                if token_total >= normalized_params.target_tokens:
                    break

            if not window:
                window.append(expanded_units[cursor])
                next_cursor = cursor + 1

            chunk_indexes_by_segment[segment_key] += 1
            midpoint = 0.0
            if total_chars > 0:
                midpoint = (window[0].char_start + window[-1].char_end) / 2 / total_chars

            chunks.append(
                _build_chunk_record(
                    chunk_number=chunk_number,
                    chunk_index_in_chapter=chunk_indexes_by_segment[segment_key],
                    window=window,
                    absolute_percent=midpoint,
                    include_raw=normalized_params.include_raw,
                )
            )
            chunk_number += 1

            if next_cursor >= segment_end:
                break

            if normalized_params.overlap_tokens <= 0:
                cursor = next_cursor
                continue

            overlap_sum = 0
            overlap_start = next_cursor
            rewind = next_cursor - 1
            while rewind >= cursor:
                overlap_sum += expanded_units[rewind].token_estimate
                overlap_start = rewind
                if overlap_sum >= normalized_params.overlap_tokens:
                    break
                rewind -= 1

            candidate_cursor = overlap_start
            if candidate_cursor <= cursor:
                candidate_cursor = cursor + 1
            if candidate_cursor >= next_cursor:
                candidate_cursor = next_cursor
            cursor = candidate_cursor

        index = segment_end

    return chunks


def filter_chunks_for_embedding(
    chunks: list[ChunkRecord],
    *,
    min_tokens: int = 120,
    exclude_section_types: list[str] | set[str] | None = None,
) -> list[ChunkRecord]:
    exclude = {item.strip().lower() for item in (exclude_section_types or []) if item.strip()}
    min_tokens = max(1, min_tokens)
    return [
        chunk
        for chunk in chunks
        if chunk.token_estimate >= min_tokens and chunk.section_type.lower() not in exclude
    ]


def write_chunks_jsonl(path: str | Path, chunks: list[ChunkRecord], include_raw: bool = False) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk.to_dict(include_raw=include_raw), ensure_ascii=False))
            output_file.write("\n")


def write_meta_json(
    path: str | Path,
    *,
    doc_id: str,
    original_filename: str,
    pages: int,
    processing_mode: str,
    total_chars: int,
    total_chunks: int,
    params: ChunkParams,
    chapter_count: int = 0,
) -> None:
    payload = {
        "doc_id": doc_id,
        "original_filename": original_filename,
        "pages": pages,
        "processing_mode": processing_mode,
        "total_chars": total_chars,
        "total_chunks": total_chunks,
        "chapter_count": chapter_count,
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_params": {
            "target_tokens": params.target_tokens,
            "overlap_tokens": params.overlap_tokens,
            "min_tokens": params.min_tokens,
            "max_tokens": params.max_tokens,
            "include_raw": params.include_raw,
        },
    }

    output_path = Path(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
