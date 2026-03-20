from dataclasses import dataclass
from math import ceil
import os
from pathlib import Path
import re
import fitz
import pytesseract
from PIL import Image
from pypdf import PdfReader


@dataclass
class ExtractionResult:
    text: str
    pages_count: int
    chars_count: int
    mode: str  # native | ocr


def configure_tesseract(tesseract_cmd: str | None) -> None:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def extract_text_with_threshold(pdf_path: str | Path, char_threshold: int = 200) -> ExtractionResult:
    path = Path(pdf_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF: {exc}") from exc

    pages_count = len(reader.pages)
    page_texts: list[str] = []

    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            # Continue even if one page extraction fails.
            page_texts.append("")

    raw_text = "\n\n".join(page_texts).strip()
    raw_chars_count = len(raw_text)
    chars_per_page = raw_chars_count / max(1, pages_count)

    mode = "native" if chars_per_page >= char_threshold else "ocr"
    text = raw_text

    if mode == "native":
        text = clean_extracted_pages(
            page_texts,
            strip_header_pattern=_env_bool("OCR_HEADER_PATTERN_STRIP", True),
            heading_markers=_env_bool("OCR_HEADING_MARKERS", True),
            safe_corrections=_env_bool("OCR_SAFE_CORRECTIONS", False),
        )

    chars_count = len(text)

    return ExtractionResult(
        text=text,
        pages_count=pages_count,
        chars_count=chars_count,
        mode=mode,
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()

    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False

    return default


def _cleanup_sample_lines() -> int:
    try:
        return max(1, int(os.getenv("OCR_HEADER_FOOTER_LINES", "3").strip()))
    except ValueError:
        return 3


def _cleanup_freq_threshold() -> float:
    try:
        freq_threshold = float(os.getenv("OCR_HEADER_FOOTER_FREQ", "0.35").strip())
    except ValueError:
        freq_threshold = 0.35

    return min(max(freq_threshold, 0.0), 1.0)


def normalize_unicode(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2032": "'",
        "\u2033": '"',
        "\u00b4": "'",
        "\u0060": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u200b": "",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00ad": "",
    }

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)

    cleaned_lines: list[str] = []

    for line in normalized.split("\n"):
        compact = re.sub(r"\s+", " ", line).strip()
        cleaned_lines.append(compact)

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


def apply_safe_corrections(text: str) -> str:
    corrected = text

    corrected = re.sub(r"\bTf\b", "If", corrected)
    corrected = re.sub(r"\bT\b(?=\s+know\b)", "I", corrected)
    corrected = re.sub(r"\bT\b(?=\s+am\b)", "I", corrected)

    return corrected


def _canonical_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def _looks_like_progress_or_page_line(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    if re.fullmatch(r"\d+", stripped):
        return True

    return bool(
        re.search(r"\b(location|page)\b.*\bof\b", stripped, flags=re.IGNORECASE)
        and re.search(r"\d+\s*%", stripped)
    )


def strip_repeating_headers_footers(
    pages: list[str],
    sample_lines: int = 3,
    freq_threshold: float = 0.35,
) -> list[str]:

    if not pages:
        return []

    sample_lines = max(1, sample_lines)

    total_pages = len(pages)
    min_count = max(2, ceil(total_pages * freq_threshold))

    top_counts: dict[str, int] = {}
    bottom_counts: dict[str, int] = {}

    page_meta: list[tuple[list[str], list[int], list[int]]] = []

    for page in pages:
        lines = page.splitlines()

        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]

        top_indices = non_empty_indices[:sample_lines]
        bottom_indices = non_empty_indices[-sample_lines:] if non_empty_indices else []

        page_meta.append((lines, top_indices, bottom_indices))

        top_seen: set[str] = set()
        for idx in top_indices:
            canon = _canonical_line(lines[idx])
            if canon:
                top_seen.add(canon)

        bottom_seen: set[str] = set()
        for idx in bottom_indices:
            canon = _canonical_line(lines[idx])
            if canon:
                bottom_seen.add(canon)

        for canon in top_seen:
            top_counts[canon] = top_counts.get(canon, 0) + 1

        for canon in bottom_seen:
            bottom_counts[canon] = bottom_counts.get(canon, 0) + 1

    frequent_top = {line for line, count in top_counts.items() if count >= min_count}
    frequent_bottom = {line for line, count in bottom_counts.items() if count >= min_count}

    cleaned_pages: list[str] = []

    for lines, top_indices, bottom_indices in page_meta:
        drop_indices: set[int] = set()

        for idx in top_indices:
            raw = lines[idx]
            canon = _canonical_line(raw)

            if canon in frequent_top or _looks_like_progress_or_page_line(raw):
                drop_indices.add(idx)

        for idx in bottom_indices:
            raw = lines[idx]
            canon = _canonical_line(raw)

            if canon in frequent_bottom or _looks_like_progress_or_page_line(raw):
                drop_indices.add(idx)

        kept_lines = [line for i, line in enumerate(lines) if i not in drop_indices]

        cleaned_pages.append("\n".join(kept_lines).strip())

    return cleaned_pages


_PAGE_NUM_THEN_CAPS_RE = re.compile(r"^\s*\d+\s+[A-Z][A-Z\s'\u2019\-\.:,]{4,}\s*$")
_CAPS_THEN_PAGE_NUM_RE = re.compile(r"^\s*[A-Z][A-Z\s'\u2019\-\.:,]{4,}\s+\d+\s*$")
_PAGE_NUM_THEN_TITLE_RE = re.compile(r"^\s*\d+\s+[A-Za-z][A-Za-z\s'\u2019\-\.:,]{4,}\s*$")


def _title_case_header_candidate(line: str) -> bool:
    if not _PAGE_NUM_THEN_TITLE_RE.fullmatch(line):
        return False

    compact = re.sub(r"\s+", " ", line).strip()

    if len(compact) > 70:
        return False

    title_part = re.sub(r"^\d+\s+", "", compact)

    words = [word for word in re.findall(r"[A-Za-z']+", title_part) if word]

    if not words or len(words) > 9:
        return False

    letters = [char for char in title_part if char.isalpha()]

    if not letters:
        return False

    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    title_case_ratio = sum(1 for word in words if word[0].isupper()) / len(words)

    has_inner_period = "." in title_part[:-1]

    if uppercase_ratio >= 0.7:
        return True

    return (not has_inner_period) and title_case_ratio >= 0.8


def strip_page_headers_by_pattern(
    pages: list[str],
    top_lines: int = 2,
    bottom_lines: int = 0,
) -> list[str]:

    if not pages:
        return []

    top_lines = max(0, top_lines)
    bottom_lines = max(0, bottom_lines)

    cleaned_pages: list[str] = []

    for page in pages:
        lines = page.splitlines()

        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]

        candidate_indices: list[int] = []

        if top_lines:
            candidate_indices.extend(non_empty_indices[:top_lines])

        if bottom_lines:
            candidate_indices.extend(non_empty_indices[-bottom_lines:])

        drop_indices: set[int] = set()

        for idx in candidate_indices:
            line = lines[idx].strip()

            if not line:
                continue

            if _PAGE_NUM_THEN_CAPS_RE.fullmatch(line) or _CAPS_THEN_PAGE_NUM_RE.fullmatch(line):
                drop_indices.add(idx)
                continue

            if _title_case_header_candidate(line):
                drop_indices.add(idx)

        kept_lines = [line for i, line in enumerate(lines) if i not in drop_indices]

        cleaned_pages.append("\n".join(kept_lines).strip())

    return cleaned_pages


_CHAPTER_HEADING_RE = re.compile(
    r"^\s*chapter\s+([0-9]+|[ivxlcdm]+)\.?\s*$",
    flags=re.IGNORECASE,
)

_SPECIAL_HEADING_RE = re.compile(
    r"^\s*(preface|foreword|about the author)\s*$",
    flags=re.IGNORECASE,
)


def _normalized_heading(line: str) -> str | None:
    chapter_match = _CHAPTER_HEADING_RE.fullmatch(line)

    if chapter_match:
        chapter_no = chapter_match.group(1).upper()
        return f"CHAPTER {chapter_no}"

    label_match = _SPECIAL_HEADING_RE.fullmatch(line)

    if label_match:
        return label_match.group(1).upper()

    return None


def mark_headings(text: str) -> str:
    if not text.strip():
        return ""

    output_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            output_lines.append("")
            continue

        normalized_heading = _normalized_heading(stripped)

        if normalized_heading:
            output_lines.append(f"=== HEADING: {normalized_heading} ===")
        else:
            output_lines.append(stripped)

    return "\n".join(output_lines).strip()


def _looks_like_heading_for_merge(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith("=== HEADING: "):
        return True

    if _normalized_heading(stripped):
        return True

    if len(stripped) <= 60 and re.search(r"[A-Z]", stripped):
        if stripped == stripped.upper() and re.fullmatch(r"[A-Z0-9\s'\u2019\-\.:,&]+", stripped):
            return True

    return False


def merge_wrapped_lines(text: str) -> str:
    """
    Rebuild paragraphs conservatively while preserving blank lines.

    Examples:
    "con-\n-eeive" -> "conceeive"
    "a\nstirring" -> "a stirring"
    "It was,\nBroken" -> "It was, Broken"
    """

    if not text.strip():
        return ""

    lines = text.splitlines()
    merged: list[str] = []

    idx = 0

    while idx < len(lines):
        current = lines[idx].strip()

        if not current:
            if merged and merged[-1] != "":
                merged.append("")
            idx += 1
            continue

        while idx + 1 < len(lines):
            nxt = lines[idx + 1].strip()

            if not nxt:
                break

            if _looks_like_heading_for_merge(current) or _looks_like_heading_for_merge(nxt):
                break

            # End-of-line hyphenation fix
            if re.search(r"[A-Za-z]{2,}-\s*$", current):
                next_alpha = re.sub(r"^[^A-Za-z]+", "", nxt)

                if next_alpha and re.match(r"^[A-Za-z]", next_alpha):
                    current = current[:-1] + next_alpha
                    idx += 1
                    continue

            next_alpha_prefix = re.sub(r"^[^A-Za-z]+", "", nxt)

            starts_with_lower = bool(next_alpha_prefix and next_alpha_prefix[0].islower())
            starts_with_upper = bool(next_alpha_prefix and next_alpha_prefix[0].isupper())

            if starts_with_lower and re.search(r"(?:^|\s)(a|I)$", current):
                current = f"{current} {nxt}"
                idx += 1
                continue

            ends_with_strong_punct = bool(re.search(r'[.?!:;"\)\]]\s*$', current))
            current_ends_comma = current.endswith(",")

            can_soft_merge = (
                (not ends_with_strong_punct and starts_with_lower)
                or (current_ends_comma and starts_with_upper)
            )

            if can_soft_merge:
                current = f"{current} {nxt}"
                idx += 1
                continue

            break

        merged.append(current)
        idx += 1

    rebuilt = "\n".join(merged).strip()

    return re.sub(r"\n{3,}", "\n\n", rebuilt)


def format_with_page_markers(pages: list[str]) -> str:
    blocks: list[str] = []

    for page_no, page_text in enumerate(pages, start=1):
        marker = f"=== PAGE {page_no} ==="
        cleaned = page_text.strip()

        blocks.append(f"{marker}\n{cleaned}" if cleaned else marker)

    return "\n\n".join(blocks).strip()


def clean_extracted_pages(
    page_texts: list[str],
    *,
    strip_header_pattern: bool = True,
    heading_markers: bool = True,
    safe_corrections: bool = False,
) -> str:

    if not page_texts:
        return ""

    normalized_pages = [normalize_unicode(text) for text in page_texts]

    stripped_pages = strip_repeating_headers_footers(
        normalized_pages,
        sample_lines=_cleanup_sample_lines(),
        freq_threshold=_cleanup_freq_threshold(),
    )

    if strip_header_pattern:
        stripped_pages = strip_page_headers_by_pattern(
            stripped_pages,
            top_lines=2,
            bottom_lines=0,
        )

    if heading_markers:
        stripped_pages = [mark_headings(text) for text in stripped_pages]

    merged_pages = [merge_wrapped_lines(text) for text in stripped_pages]

    if safe_corrections:
        merged_pages = [apply_safe_corrections(text) for text in merged_pages]

    return format_with_page_markers(merged_pages)


def extract_text_with_ocr(pdf_path: str | Path, lang: str = "eng", zoom: float = 2.0) -> str:
    path = Path(pdf_path)

    page_texts: list[str] = []

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF for OCR: {exc}") from exc

    try:
        for page in doc:
            matrix = fitz.Matrix(zoom, zoom)

            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            page_text = pytesseract.image_to_string(image, lang=lang) or ""

            page_texts.append(page_text)

    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc

    finally:
        doc.close()

    cleaning_enabled = _env_bool("OCR_CLEANING", True)

    if not cleaning_enabled:
        return "\n\n".join((text or "").strip() for text in page_texts).strip()

    return clean_extracted_pages(
        page_texts,
        strip_header_pattern=_env_bool("OCR_HEADER_PATTERN_STRIP", True),
        heading_markers=_env_bool("OCR_HEADING_MARKERS", True),
        safe_corrections=_env_bool("OCR_SAFE_CORRECTIONS", False),
    )