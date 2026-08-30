import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import lxml.html

from services.rag_service import DocumentMetadata


_PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data",
    "financial_documents",
    "processed",
)

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class DocumentChunk:
    chunk_id: str
    chunk_index: int
    text: str
    company_symbol: str
    company_name: str
    document_type: str
    document_year: Optional[int]
    source: str
    source_url: Optional[str]
    document_id: str
    section: Optional[str]


@dataclass
class ProcessedDocument:
    company_symbol: str
    company_name: str
    document_type: str
    document_year: Optional[int]
    source: str
    source_url: Optional[str]
    document_id: str
    raw_path: str
    processed_path: str
    total_chunks: int
    chunks: List[dict]


def _get_processed_path(symbol: str, filename_stem: str) -> str:
    symbol_dir = os.path.join(_PROCESSED_DIR, symbol.upper())
    os.makedirs(symbol_dir, exist_ok=True)
    return os.path.join(symbol_dir, f"{filename_stem}.json")


def _clean_html(html_content: str) -> str:
    try:
        root = lxml.html.fromstring(html_content.encode('utf-8') if isinstance(html_content, str) else html_content)
    except Exception:
        return html_content if isinstance(html_content, str) else html_content.decode('utf-8', errors='ignore')

    for tag in root.iter("script", "style", "noscript"):
        tag.drop_tree()

    for comment in root.xpath('//comment()'):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)

    for hidden in root.xpath('//div[@style="display:none"] | //div[contains(@style,"display: none")]'):
        hidden.drop_tree()

    for hidden in root.xpath('//ix:hidden', namespaces={'ix': 'http://www.xbrl.org/2013/inlineXBRL'}):
        hidden.drop_tree()

    for header in root.xpath('//ix:header', namespaces={'ix': 'http://www.xbrl.org/2013/inlineXBRL'}):
        header.drop_tree()

    for br in root.iter('br'):
        br.tail = '\n' + (br.tail or '')

    block_tags = {'p', 'div', 'table', 'tr', 'th', 'td', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol', 'blockquote', 'section', 'article'}
    for element in root.iter():
        if element.tag in block_tags:
            if element.tail and not element.tail.startswith('\n'):
                element.tail = '\n' + element.tail

    text = root.text_content()

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def _chunk_text(text: str, chunk_size: int = _DEFAULT_CHUNK_SIZE, chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if chunk_size <= 0:
        return [text]

    paragraphs = text.split("\n")
    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if len(paragraph) > chunk_size:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunks.append(paragraph[start:end].strip())
                    start = end - chunk_overlap
                    if start < 0:
                        start = 0
                    if start >= len(paragraph):
                        break
                current_chunk = ""
            else:
                current_chunk = paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _compute_chunk_id(company_symbol: str, document_id: str, chunk_index: int, text: str) -> str:
    payload = f"{company_symbol.upper()}|{document_id}|{chunk_index}|{text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def process_document(
    raw_path: str,
    metadata: DocumentMetadata,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Optional[ProcessedDocument]:
    if not os.path.exists(raw_path):
        return None

    try:
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
    except Exception:
        return None

    if not html_content or not html_content.strip():
        return None

    cleaned_text = _clean_html(html_content)

    if not cleaned_text or not cleaned_text.strip():
        return None

    chunk_texts = _chunk_text(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    filename = os.path.basename(raw_path)
    filename_stem = os.path.splitext(filename)[0]
    processed_path = _get_processed_path(metadata.company_symbol, filename_stem)

    chunks = []
    for idx, chunk_text in enumerate(chunk_texts):
        chunk_id = _compute_chunk_id(
            company_symbol=metadata.company_symbol,
            document_id=metadata.document_id,
            chunk_index=idx,
            text=chunk_text,
        )

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            chunk_index=idx,
            text=chunk_text,
            company_symbol=metadata.company_symbol.upper(),
            company_name=metadata.company_name,
            document_type=metadata.document_type,
            document_year=metadata.document_year,
            source=metadata.source,
            source_url=metadata.source_url,
            document_id=metadata.document_id,
            section=None,
        )
        chunks.append(asdict(chunk))

    processed_doc = ProcessedDocument(
        company_symbol=metadata.company_symbol.upper(),
        company_name=metadata.company_name,
        document_type=metadata.document_type,
        document_year=metadata.document_year,
        source=metadata.source,
        source_url=metadata.source_url,
        document_id=metadata.document_id,
        raw_path=raw_path,
        processed_path=processed_path,
        total_chunks=len(chunks),
        chunks=chunks,
    )

    try:
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(asdict(processed_doc), f, indent=2, default=str)
    except Exception:
        return None

    return processed_doc


def get_processed_document(raw_path: str, metadata: DocumentMetadata) -> Optional[ProcessedDocument]:
    if not os.path.exists(raw_path):
        return None

    filename = os.path.basename(raw_path)
    filename_stem = os.path.splitext(filename)[0]
    processed_path = _get_processed_path(metadata.company_symbol, filename_stem)

    if not os.path.exists(processed_path):
        return None

    try:
        with open(processed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("raw_path") != raw_path:
            return None

        return ProcessedDocument(**data)
    except Exception:
        return None
