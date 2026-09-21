import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

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


# Section mappings for 10-Q and 10-K filings
# Each entry: (regex_pattern, section_name, filing_types)
# filing_types: "10-Q", "10-K", or "both"
# ORDER MATTERS: more specific patterns must come BEFORE generic fallbacks
_SEC_SECTION_PATTERNS = [
    # Part markers (must come before Item patterns)
    (r'Part\s+I\b', 'Part I', 'both'),
    (r'Part\s+II\b', 'Part II', 'both'),
    (r'Part\s+III\b', 'Part III', 'both'),
    (r'Part\s+IV\b', 'Part IV', 'both'),
    
    # 10-Q specific sections (specific patterns with full titles)
    # More flexible patterns to match real SEC header variations
    (r'Item\s+1\s*[\.:\-–—]?\s*Financial Statements\b', 'Item 1 - Financial Statements', '10-Q'),
    (r'Item\s+2\s*[\.:\-–—]?\s*Management[\'\u2019]?s?\s*Discussion\b', 'Item 2 - MD&A', '10-Q'),
    (r'Item\s+3\s*[\.:\-–—]?\s*Quantitative and Qualitative Disclosures\b', 'Item 3 - Market Risk', '10-Q'),
    (r'Item\s+4\s*[\.:\-–—]?\s*Controls and Procedures\b', 'Item 4 - Controls and Procedures', '10-Q'),
    
    # Part II 10-Q specific
    (r'Item\s+1\s*[\.:\-–—]?\s*Legal Proceedings\b', 'Item 1 - Legal Proceedings', '10-Q'),
    (r'Item\s+1A\s*[\.:\-–—]?\s*Risk Factors\b', 'Item 1A - Risk Factors', 'both'),
    (r'Item\s+2\s*[\.:\-–—]?\s*Unregistered Sales\b', 'Item 2 - Unregistered Sales', '10-Q'),
    (r'Item\s+3\s*[\.:\-–—]?\s*Defaults Upon Senior Securities\b', 'Item 3 - Defaults Upon Senior Securities', '10-Q'),
    (r'Item\s+4\s*[\.:\-–—]?\s*Mine Safety Disclosures\b', 'Item 4 - Mine Safety Disclosures', 'both'),
    (r'Item\s+5\s*[\.:\-–—]?\s*Other Information\b', 'Item 5 - Other Information', '10-Q'),
    (r'Item\s+6\s*[\.:\-–—]?\s*Exhibits\b', 'Item 6 - Exhibits', 'both'),
    
    # 10-K specific sections (Part I) - specific patterns
    (r'Item\s+1\s*[\.:\-–—]?\s*Business\b', 'Item 1 - Business', '10-K'),
    (r'Item\s+1A\s*[\.:\-–—]?\s*Risk Factors\b', 'Item 1A - Risk Factors', 'both'),
    (r'Item\s+1B\s*[\.:\-–—]?\s*Unresolved Staff Comments\b', 'Item 1B - Unresolved Staff Comments', '10-K'),
    (r'Item\s+1C\s*[\.:\-–—]?\s*Cybersecurity\b', 'Item 1C - Cybersecurity', '10-K'),
    (r'Item\s+2\s*[\.:\-–—]?\s*Properties\b', 'Item 2 - Properties', '10-K'),
    (r'Item\s+3\s*[\.:\-–—]?\s*Legal Proceedings\b', 'Item 3 - Legal Proceedings', '10-K'),
    (r'Item\s+4\s*[\.:\-–—]?\s*Mine Safety Disclosures\b', 'Item 4 - Mine Safety Disclosures', 'both'),
    
    # Part II 10-K specific
    (r'Item\s+5\s*[\.:\-–—]?\s*Market for Registrant\b', 'Item 5 - Market for Registrant', '10-K'),
    (r'Item\s+6\s*[\.:\-–—]?\s*Reserved\b', 'Item 6 - Reserved', '10-K'),
    (r'Item\s+7\s*[\.:\-–—]?\s*Management[\'\u2019]?s?\s*Discussion\b', 'Item 7 - MD&A', '10-K'),
    (r'Item\s+7A\s*[\.:\-–—]?\s*Quantitative and Qualitative Disclosures\b', 'Item 7A - Market Risk', '10-K'),
    (r'Item\s+8\s*[\.:\-–—]?\s*Financial Statements\b', 'Item 8 - Financial Statements', '10-K'),
    (r'Item\s+9\s*[\.:\-–—]?\s*Changes in and Disagreements\b', 'Item 9 - Changes and Disagreements', '10-K'),
    (r'Item\s+9A\s*[\.:\-–—]?\s*Controls and Procedures\b', 'Item 9A - Controls and Procedures', '10-K'),
    (r'Item\s+9B\s*[\.:\-–—]?\s*Other Information\b', 'Item 9B - Other Information', '10-K'),
    
    # Generic fallbacks (less specific, used when specific not matched)
    # These must come AFTER all specific patterns
    # 10-Q should NOT get 10-K style generics like Item 7, Item 8, etc.
    (r'Item\s+1A\b', 'Item 1A - Risk Factors', 'both'),
    (r'Item\s+1B\b', 'Item 1B - Unresolved Staff Comments', 'both'),
    (r'Item\s+1C\b', 'Item 1C - Cybersecurity', 'both'),
    # 10-Q generic fallbacks (only Items that exist in 10-Q Part II)
    (r'Item\s+1\b', 'Item 1', '10-Q'),
    (r'Item\s+2\b', 'Item 2', '10-Q'),
    (r'Item\s+3\b', 'Item 3', '10-Q'),
    (r'Item\s+4\b', 'Item 4', '10-Q'),
    (r'Item\s+5\b', 'Item 5', '10-Q'),
    (r'Item\s+6\b', 'Item 6', '10-Q'),
    # 10-K generic fallbacks (only for 10-K)
    (r'Item\s+7\b', 'Item 7', '10-K'),
    (r'Item\s+7A\b', 'Item 7A', '10-K'),
    (r'Item\s+8\b', 'Item 8', '10-K'),
    (r'Item\s+9\b', 'Item 9', '10-K'),
    (r'Item\s+9A\b', 'Item 9A', '10-K'),
    (r'Item\s+9B\b', 'Item 9B', '10-K'),
    (r'Item\s+10\b', 'Item 10', '10-K'),
    (r'Item\s+11\b', 'Item 11', '10-K'),
    (r'Item\s+12\b', 'Item 12', '10-K'),
    (r'Item\s+13\b', 'Item 13', '10-K'),
    (r'Item\s+14\b', 'Item 14', '10-K'),
    (r'Item\s+15\b', 'Item 15', '10-K'),
    (r'Item\s+16\b', 'Item 16', '10-K'),
]


def _get_section_patterns_for_filing(document_type: str) -> List[Tuple[str, str]]:
    """Get section patterns applicable to the given filing type."""
    filing_type = document_type.upper().strip()
    patterns = []
    for pattern, name, filing in _SEC_SECTION_PATTERNS:
        if filing == 'both' or filing == filing_type:
            patterns.append((pattern, name))
    return patterns


def _find_section_headers(text: str, document_type: str) -> List[Tuple[int, str]]:
    """
    Find all REAL section headers in the document text with their positions.
    
    Returns list of (position, section_name) tuples sorted by position.
    
    Uses contextual clues to distinguish actual section headers from
    Table of Contents references and textual references:
    - Real headers appear as standalone headings, often with the section title
    - TOC entries are in table/list format with page numbers or dots
    - Textual references like "see Item 7..." are embedded in sentences
    - We prefer more specific patterns (with title) over generic ones
    """
    patterns = _get_section_patterns_for_filing(document_type)
    headers = []
    
    for pattern, section_name in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos = match.start()
            
            # Check context to determine if this is a real section header
            context_before = text[max(0, pos - 100):pos]
            context_after = text[pos:pos + 200]
            
            # Skip if it looks like a TOC entry
            toc_indicators = [
                r'\.{4,}\s*\d',  # dots for page numbers followed by digits (...... 21)
                r'\.{3,}\s*\d{1,3}\b',  # three+ dots followed by page number
                r'\|\s*\d+\s*\|',  # pipe table format with numbers
                r'\bpage\s+\d+',  # "page 21"
            ]
            is_toc = any(re.search(ind, context_after, re.IGNORECASE) for ind in toc_indicators)
            
            # Also skip if it's in the first 2000 chars and looks like a TOC table
            if is_toc and pos < 2000:
                continue
            
# Skip if it looks like a textual reference (embedded in a sentence)
            # Real headers typically start at line beginning or after newline
            # Textual references like "see Item 7..." or "as described in Item 1A" 
            # have preceding text on the same "line"
            is_textual_ref = False
            # Check if preceded by sentence text (not just whitespace/newline)
            # Use raw context_before for newline check, stripped for content check
            context_before_stripped = context_before.strip()
            if context_before_stripped and not re.search(r'\n\s*$', context_before):
                # Look for words that indicate a reference rather than a header
                ref_indicators = [
                    r'\bsee\s+', r'\bsee\s+Item\s+', r'\bas\s+described\s+in\s+',
                    r'\bdescribed\s+in\s+', r'\bmentioned\s+in\s+', r'\bper\s+Item\s+',
                    r'\bunder\s+Item\s+', r'\baccording\s+to\s+Item\s+',
                    r'\bper\s+', r'\bref\.\s*Item\s+', r'\brefers\s+to\s+',
                ]
                is_textual_ref = any(re.search(ind, context_before, re.IGNORECASE) for ind in ref_indicators)
                
                # Also check if it's mid-sentence (no period/newline before, lowercase after)
                if not is_textual_ref:
                    # If context_before ends with lowercase letter and context_after starts with lowercase,
                    # it's likely mid-sentence
                    if (re.search(r'[a-z]\s*$', context_before) and 
                        re.search(r'^\s*[a-z]', context_after)):
                        is_textual_ref = True
            
            if is_textual_ref:
                continue
            
            # Real section header heuristic: 
            # Should be at start of "line" (preceded by newline or start of text)
            # or preceded only by whitespace
            is_real_header = (
                pos == 0 or 
                context_before.strip() == '' or
                context_before.endswith('\n')
            )
            
            # For generic fallbacks, be more strict - require header-like context
            is_generic_fallback = section_name in [
                'Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5', 'Item 6',
                'Item 7', 'Item 7A', 'Item 8', 'Item 9', 'Item 9A', 'Item 9B',
                'Item 10', 'Item 11', 'Item 12', 'Item 13', 'Item 14', 'Item 15', 'Item 16'
            ]
            
            if is_generic_fallback and not is_real_header:
                # Only accept generic fallback if it looks like a real header
                continue
            
            headers.append((pos, section_name))
    
    # Sort by position
    headers.sort(key=lambda x: x[0])
    
    # Deduplicate: if multiple patterns match at same position, keep most specific
    # But keep Part markers and different Item numbers separate
    deduped = []
    for pos, name in headers:
        if not deduped or pos - deduped[-1][0] > 50:  # Different position
            deduped.append((pos, name))
        else:
            # Same vicinity - check if both are Part markers or different Item numbers
            prev_name = deduped[-1][1]
            # Keep both if one is a Part marker and other is not
            is_part = name.startswith('Part ')
            prev_is_part = prev_name.startswith('Part ')
            if is_part != prev_is_part:
                deduped.append((pos, name))
            # Keep both if both are Item but different numbers
            elif name.startswith('Item ') and prev_name.startswith('Item '):
                # Extract item numbers
                m1 = re.match(r'Item\s+(\d+)', name)
                m2 = re.match(r'Item\s+(\d+)', prev_name)
                if m1 and m2 and m1.group(1) != m2.group(1):
                    deduped.append((pos, name))
            # Otherwise keep the more specific one (longer name usually)
            elif len(name) > len(prev_name):
                deduped[-1] = (pos, name)
    
    return deduped


def _assign_sections_to_chunks(chunk_texts: List[str], cleaned_text: str, document_type: str) -> List[Optional[str]]:
    """
    Assign section to each chunk using stateful section tracking.
    
    1. Find all real section headers in the document
    2. For each chunk, find its position in cleaned_text
    3. Assign the most recent section header that comes before the chunk
    """
    # Find all section headers
    headers = _find_section_headers(cleaned_text, document_type)
    
    if not headers:
        return [None] * len(chunk_texts)
    
    # For each chunk, find its position and assign the current section
    chunk_sections = []
    header_idx = 0
    current_section = None
    
    for chunk_text in chunk_texts:
        # Find chunk position in cleaned_text
        try:
            pos = cleaned_text.index(chunk_text[:100])
        except ValueError:
            chunk_sections.append(current_section)
            continue
        
        # Advance header_idx to the last header before this chunk
        while header_idx < len(headers) and headers[header_idx][0] <= pos:
            current_section = headers[header_idx][1]
            header_idx += 1
        
        chunk_sections.append(current_section)
    
    return chunk_sections


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


@dataclass
class ChunkComparison:
    """Comparison result between old and new chunks."""
    unchanged: List[str]  # chunk_ids that are identical
    changed: List[Tuple[str, str]]  # (old_chunk_id, new_chunk_id) for modified chunks
    new: List[str]  # new_chunk_ids with no old counterpart
    removed: List[str]  # old_chunk_ids with no new counterpart
    mapping: Dict[str, str]  # old_chunk_id -> new_chunk_id (for changed chunks)
    # Migration classifications for changed chunks
    classifications: Optional[Dict[str, str]] = None  # new_chunk_id -> classification


def _get_processed_path(symbol: str, filename_stem: str) -> str:
    symbol_dir = os.path.join(_PROCESSED_DIR, symbol.upper())
    os.makedirs(symbol_dir, exist_ok=True)
    return os.path.join(symbol_dir, f"{filename_stem}.json")


def _get_dry_run_path(symbol: str, filename_stem: str) -> str:
    """Get path for dry-run output (separate from production)."""
    dry_run_dir = os.path.join(_PROCESSED_DIR, "_dry_run", symbol.upper())
    os.makedirs(dry_run_dir, exist_ok=True)
    return os.path.join(dry_run_dir, f"{filename_stem}.json")


def _clean_html(html_content: str) -> str:
    """Extract clean text from SEC HTML, preserving Unicode and table structure."""
    # Remove XML declaration if present (lxml doesn't like encoding declarations in strings)
    html_content = re.sub(r'<\?xml[^>]*\?>', '', html_content, count=1)
    
    try:
        root = lxml.html.fromstring(html_content)
    except Exception:
        return html_content if isinstance(html_content, str) else html_content.decode('utf-8', errors='ignore')

    # Remove non-content elements
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

    # Preserve table structure by converting to markdown-like format
    for table in root.xpath('//table'):
        _convert_table_to_text(table)

    # Convert <br> to newlines
    for br in root.iter('br'):
        br.tail = '\n' + (br.tail or '')

    # Add newlines after block elements for better paragraph separation
    block_tags = {'p', 'div', 'table', 'tr', 'th', 'td', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol', 'blockquote', 'section', 'article'}
    for element in root.iter():
        if element.tag in block_tags:
            if element.tail and not element.tail.startswith('\n'):
                element.tail = '\n' + element.tail

    text = root.text_content()

    # Clean up lines but preserve meaningful whitespace
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def _convert_table_to_text(table) -> None:
    """Convert HTML table to a structured text representation."""
    rows = []
    for tr in table.xpath('.//tr'):
        cells = []
        for td in tr.xpath('.//th | .//td'):
            cell_text = td.text_content().strip()
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    
    if rows:
        # Replace table with structured text
        table_text = "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"
        # Create a text node to replace the table
        table.getparent().replace(table, lxml.html.fromstring(f"<div>{table_text}</div>"))


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Split on sentence boundaries while preserving the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, chunk_size: int = _DEFAULT_CHUNK_SIZE, chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Chunk text by sentences with overlap, avoiding mid-sentence splits."""
    if chunk_size <= 0:
        return [text]

    # Split into sentences first
    sentences = _split_into_sentences(text)
    
    chunks: List[str] = []
    current_chunk = ""
    current_sentences: List[str] = []

    for sentence in sentences:
        # If adding this sentence would exceed chunk_size
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(current_chunk.strip())
            
            # Build overlap from end of current chunk
            overlap_text = ""
            # Take sentences from end of current chunk until we have enough overlap
            for s in reversed(current_sentences):
                if len(overlap_text) + len(s) + 1 <= chunk_overlap:
                    overlap_text = s + " " + overlap_text
                else:
                    break
            
            current_chunk = overlap_text
            current_sentences = _split_into_sentences(overlap_text) if overlap_text else []
        
        current_chunk += sentence + " "
        current_sentences.append(sentence)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _trim_filename_header(text: str, filename_stem: str) -> str:
    """Remove filename/header pollution from the beginning of processed text."""
    # Remove the filename if it appears at the start
    filename_base = filename_stem.replace('-', ' ').replace('_', ' ')
    # Common patterns: "aapl-20260627", "tsla-20260630", "nvda-20260726"
    patterns = [
        r'^[a-z]+-\d{8}\s*',  # ticker-date
        r'^[a-z]+\s+\d{8}\s*',  # ticker date
        re.escape(filename_base) + r'\s*',
        r'^UNITED STATES\s*',  # Often appears after filename
        r'^SECURITIES AND EXCHANGE COMMISSION\s*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up any leading whitespace/newlines
    return text.lstrip()


def _compute_chunk_id(company_symbol: str, document_id: str, chunk_index: int, text: str) -> str:
    payload = f"{company_symbol.upper()}|{document_id}|{chunk_index}|{text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def process_document(
    raw_path: str,
    metadata: DocumentMetadata,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    dry_run: bool = False,
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

    # Trim filename/header from beginning
    filename = os.path.basename(raw_path)
    filename_stem = os.path.splitext(filename)[0]
    cleaned_text = _trim_filename_header(cleaned_text, filename_stem)

    chunk_texts = _chunk_text(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Assign sections using stateful tracking
    chunk_sections = _assign_sections_to_chunks(chunk_texts, cleaned_text, metadata.document_type)

    if dry_run:
        processed_path = _get_dry_run_path(metadata.company_symbol, filename_stem)
    else:
        processed_path = _get_processed_path(metadata.company_symbol, filename_stem)

    chunks = []
    for idx, chunk_text in enumerate(chunk_texts):
        chunk_id = _compute_chunk_id(
            company_symbol=metadata.company_symbol,
            document_id=metadata.document_id,
            chunk_index=idx,
            text=chunk_text,
        )

        section = chunk_sections[idx] if idx < len(chunk_sections) else None

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
            section=section,
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


def process_document_dry_run(
    raw_path: str,
    metadata: DocumentMetadata,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Optional[ProcessedDocument]:
    """Process document for dry-run comparison without modifying production data."""
    return process_document(raw_path, metadata, chunk_size, chunk_overlap, dry_run=True)


def _compute_jaccard(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / max(len(words1), len(words2))


def _classify_changed_chunk(old_text: str, new_text: str, jaccard: float) -> str:
    """
    Classify a changed chunk based on validated migration analysis.
    
    Classifications (from Stage 7.8.2-D Step 4 validation):
    - SAFE_REUSE: Jaccard >= 0.70 (267 chunks) - reuse old vector
    - NEED_NEW_EMBEDDING: Jaccard < 0.55 or specific semantic mismatches (22 chunks)
    - REVIEW: 0.55 <= Jaccard < 0.70 (30 chunks) - default to new embedding
    """
    if jaccard >= 0.70:
        return "SAFE_REUSE"
    elif jaccard < 0.55:
        return "NEED_NEW_EMBEDDING"
    else:
        return "REVIEW"


def compare_processed_documents(
    old_processed_doc: ProcessedDocument,
    new_processed_doc: ProcessedDocument,
) -> ChunkComparison:
    """Compare old and new processed documents to identify changes."""
    old_chunks = {c["chunk_id"]: c for c in old_processed_doc.chunks}
    new_chunks = {c["chunk_id"]: c for c in new_processed_doc.chunks}

    old_ids = set(old_chunks.keys())
    new_ids = set(new_chunks.keys())

    unchanged = []
    changed = []
    new = []
    removed = []
    mapping = {}
    classifications = {}

    # Unchanged: same chunk_id in both
    for cid in old_ids & new_ids:
        if old_chunks[cid]["text"] == new_chunks[cid]["text"]:
            unchanged.append(cid)
        else:
            changed.append((cid, cid))
            mapping[cid] = cid
            # Classify this changed chunk
            old_text = old_chunks[cid]["text"]
            new_text = new_chunks[cid]["text"]
            jaccard = _compute_jaccard(old_text, new_text)
            classifications[cid] = _classify_changed_chunk(old_text, new_text, jaccard)

    # Removed: in old but not in new
    for cid in old_ids - new_ids:
        removed.append(cid)

    # New: in new but not in old
    for cid in new_ids - old_ids:
        new.append(cid)

    # Try to match changed/removed/new by content similarity
    # For removed chunks, try to find a new chunk with similar text
    for old_cid in list(removed):
        old_text = old_chunks[old_cid]["text"]
        best_match = None
        best_score = 0
        
        for new_cid in list(new):
            new_text = new_chunks[new_cid]["text"]
            # Simple similarity: common words ratio
            old_words = set(old_text.lower().split())
            new_words = set(new_text.lower().split())
            if old_words and new_words:
                score = len(old_words & new_words) / max(len(old_words), len(new_words))
                if score > best_score and score > 0.5:  # Threshold for "same content, different chunking"
                    best_score = score
                    best_match = new_cid
        
        if best_match:
            removed.remove(old_cid)
            new.remove(best_match)
            changed.append((old_cid, best_match))
            mapping[old_cid] = best_match
            # Classify this matched pair
            new_text = new_chunks[best_match]["text"]
            classifications[best_match] = _classify_changed_chunk(old_text, new_text, best_score)

    return ChunkComparison(
        unchanged=unchanged,
        changed=changed,
        new=new,
        removed=removed,
        mapping=mapping,
        classifications=classifications,
    )


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


def get_dry_run_document(raw_path: str, metadata: DocumentMetadata) -> Optional[ProcessedDocument]:
    """Get dry-run processed document if it exists."""
    filename = os.path.basename(raw_path)
    filename_stem = os.path.splitext(filename)[0]
    dry_run_path = _get_dry_run_path(metadata.company_symbol, filename_stem)

    if not os.path.exists(dry_run_path):
        return None

    try:
        with open(dry_run_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("raw_path") != raw_path:
            return None

        return ProcessedDocument(**data)
    except Exception:
        return None
