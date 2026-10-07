"""
ingestion/loader.py - FIXED VERSION
Uses pdfplumber with column-aware extraction to preserve document structure,
including multi-column resume/CV layouts.
"""

import hashlib
import re
import zipfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("ingestion.loader")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_docx_clean(file_path: Path) -> List[Document]:
    with zipfile.ZipFile(str(file_path), 'r') as z:
        if 'word/document.xml' not in z.namelist():
            raise RuntimeError("Not a valid docx: word/document.xml not found")
        xml = z.read('word/document.xml').decode('utf-8', errors='replace')
    text_nodes = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml, re.DOTALL)
    raw = ' '.join(t for t in text_nodes if t.strip())
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = re.sub(r'ADDIN\s+\S+\s*\{.*?\}', ' ', raw, flags=re.DOTALL)
    raw = re.sub(r'\[\d+\]', '', raw)
    raw = re.sub(r'[ \t]{2,}', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = raw.strip()
    if len(raw) < 50:
        raise RuntimeError(f"Document text too short ({len(raw)} chars).")
    logger.info(f"Extracted {len(raw)} chars from docx via direct XML method")
    return [Document(page_content=raw, metadata={"page": 0})]


def _detect_columns(words, page_width, gap_threshold_ratio=0.04):
    """
    Detect column boundaries by finding large horizontal gaps in word
    x-positions across the page. Works for the common 2-column resume
    layout where a wide whitespace gutter separates left/right columns.

    Returns a sorted list of x-boundaries splitting the page into columns.
    e.g. [] for single column, [split_x] for 2 columns.
    """
    if not words:
        return []

    # Build a histogram of x-ranges covered by word bounding boxes
    xs = sorted((w["x0"], w["x1"]) for w in words)

    # Merge overlapping/close intervals to find "ink" coverage on the x-axis
    merged = []
    gap_threshold = page_width * gap_threshold_ratio
    for x0, x1 in xs:
        if merged and x0 - merged[-1][1] <= gap_threshold:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))

    # Only treat as multi-column if there's exactly one big internal gap
    # roughly in the middle third of the page (avoids false positives on
    # normal paragraphs that just have uneven line lengths).
    if len(merged) < 2:
        return []

    boundaries = []
    for i in range(len(merged) - 1):
        gap_start, gap_end = merged[i][1], merged[i + 1][0]
        gap_size = gap_end - gap_start
        gap_center = (gap_start + gap_end) / 2
        # Require a substantial gutter (>6% of page width) roughly centered
        if gap_size > page_width * 0.06 and page_width * 0.25 < gap_center < page_width * 0.75:
            boundaries.append(gap_center)

    return boundaries[:1]  # Only support 2-column split (most common case)


def _extract_page_text_column_aware(page) -> str:
    """
    Extract text from a pdfplumber page, detecting multi-column layouts
    and reading each column top-to-bottom before moving to the next,
    instead of pdfplumber's default left-to-right line reading which
    garbles multi-column resumes/CVs (e.g. merges unrelated lines from
    the left and right columns into one nonsensical line).
    """
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)

    if not words:
        return page.extract_text() or ""

    boundaries = _detect_columns(words, page.width)

    if not boundaries:
        # Single column — default extraction is reliable
        return page.extract_text() or ""

    split_x = boundaries[0]
    left_words  = [w for w in words if (w["x0"] + w["x1"]) / 2 < split_x]
    right_words = [w for w in words if (w["x0"] + w["x1"]) / 2 >= split_x]

    def words_to_text(col_words):
        if not col_words:
            return ""
        # Group by line (top position), then join left-to-right within a line
        col_words = sorted(col_words, key=lambda w: (round(w["top"], 1), w["x0"]))
        lines = []
        current_line = []
        current_top = None
        line_tolerance = 3  # px tolerance for "same line"

        for w in col_words:
            if current_top is None or abs(w["top"] - current_top) <= line_tolerance:
                current_line.append(w)
                current_top = w["top"] if current_top is None else current_top
            else:
                lines.append(" ".join(x["text"] for x in sorted(current_line, key=lambda x: x["x0"])))
                current_line = [w]
                current_top = w["top"]
        if current_line:
            lines.append(" ".join(x["text"] for x in sorted(current_line, key=lambda x: x["x0"])))

        return "\n".join(lines)

    left_text  = words_to_text(left_words)
    right_text = words_to_text(right_words)

    # Read left column fully, then right column — this matches how a human
    # reads a 2-column resume, unlike raw left-to-right line extraction.
    logger.debug(
        f"Multi-column page detected (split at x={split_x:.0f}/{page.width:.0f}): "
        f"left={len(left_words)} words, right={len(right_words)} words"
    )
    return left_text + "\n\n" + right_text


def _load_pdf_with_structure(file_path: Path) -> List[Document]:
    """
    Load PDF using pdfplumber to preserve document structure.

    Detects multi-column layouts (common in resumes/CVs) and extracts
    each column separately, reading top-to-bottom per column, instead of
    naively reading left-to-right across the full page width — which
    merges unrelated text from side-by-side columns into garbled lines.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(file_path)).load()

    docs = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            logger.info(f"Extracting text from {len(pdf.pages)} page(s) with pdfplumber")

            for page_num, page in enumerate(pdf.pages):
                text = _extract_page_text_column_aware(page)

                if not text or len(text.strip()) < 50:
                    logger.warning(f"Page {page_num} produced minimal text ({len(text)} chars)")
                    continue

                # Enhance paragraph breaks: add \n before common section headers
                # Resume sections typically don't have explicit spacing in PDFs
                section_keywords = [
                    "Education", "Experience", "Projects", "Skills", "Certifications",
                    "Work Experience", "Technical Skills", "Internship", "Summary",
                    "Background", "Expertise", "Employment", "Accomplishments"
                ]

                for keyword in section_keywords:
                    text = re.sub(
                        rf'(\n)({keyword})',
                        r'\n\n\2',
                        text,
                        flags=re.IGNORECASE
                    )

                # Clean up excessive whitespace but preserve paragraph breaks
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = text.strip()

                docs.append(Document(
                    page_content=text,
                    metadata={"page": page_num}
                ))

                logger.info(f"Page {page_num}: Extracted {len(text)} chars, {len(text.split())} words")

    except Exception as exc:
        logger.error(f"pdfplumber extraction failed: {exc}")
        logger.warning("Falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(file_path)).load()

    if not docs:
        raise RuntimeError("No text could be extracted from PDF")

    logger.info(f"Successfully extracted {len(docs)} page(s) from PDF")
    return docs


def load_document(file_path, display_name=None):
    """
    Load a document from file.

    Supports: .pdf, .txt, .md, .docx

    Returns:
        List[Document]: Loaded documents with metadata
    """
    path = Path(file_path)
    ext  = path.suffix.lower()
    name = display_name or path.name

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")

    logger.info(f"Loading document: {name} ({ext})")

    try:
        if ext == ".pdf":
            docs = _load_pdf_with_structure(path)
        elif ext == ".txt":
            docs = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".md":
            docs = UnstructuredMarkdownLoader(str(path)).load()
        elif ext == ".docx":
            docs = _load_docx_clean(path)
        else:
            raise ValueError(f"Unhandled extension: {ext}")

    except Exception as exc:
        logger.error(f"Failed to load {name}: {exc}")
        raise RuntimeError(f"Could not parse '{name}': {exc}") from exc

    # Add consistent metadata
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source":       name,
            "file_type":    ext.lstrip("."),
            "page":         doc.metadata.get("page", i),
            "content_hash": _content_hash(doc.page_content),
        })
        doc.metadata.pop("file_path", None)
        doc.metadata.pop("source_path", None)

    logger.info(f"Loaded {len(docs)} page(s) from '{name}'")
    return docs