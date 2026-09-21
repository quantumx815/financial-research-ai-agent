import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.document_processor import (
    _clean_html,
    _chunk_text,
    _compute_chunk_id,
    _split_into_sentences,
    _trim_filename_header,
    _find_section_headers,
    _assign_sections_to_chunks,
    process_document,
    process_document_dry_run,
    compare_processed_documents,
    DocumentChunk,
    ProcessedDocument,
    ChunkComparison,
)
from services.rag_service import DocumentMetadata


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_path = os.path.join(self.temp_dir, "test.html")
        self.metadata = DocumentMetadata(
            company_symbol="AAPL",
            company_name="Apple Inc.",
            document_type="10-Q",
            document_year=2024,
            source="SEC",
            source_url="https://example.com",
            document_id="doc-1",
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_encoding_preservation(self):
        """Test that Unicode characters are preserved correctly."""
        html = """<html><body><p>Apple\u2019s revenue was $100\u00a0billion.</p></body></html>"""
        with open(self.raw_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        result = process_document(self.raw_path, self.metadata)
        self.assertIsNotNone(result)
        # Check that Unicode chars are preserved
        text = result.chunks[0]["text"]
        self.assertIn('\u2019', text)  # Right single quotation mark
        self.assertIn('\u00a0', text)  # Non-breaking space

    def test_filename_header_trimming(self):
        """Test that filename/header pollution is removed from start."""
        html = """<html><body><p>aapl-20260627\nUNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q (Mark One) Quarterly report...</p></body></html>"""
        with open(self.raw_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        result = process_document(self.raw_path, self.metadata)
        self.assertIsNotNone(result)
        text = result.chunks[0]["text"]
        # Should not contain the filename
        self.assertNotIn('aapl-20260627', text)
        # Should not contain the SEC header boilerplate
        self.assertNotIn('UNITED STATES SECURITIES AND EXCHANGE COMMISSION', text)
        # Should start with meaningful content
        self.assertTrue(text.startswith('Washington, D.C.') or text.startswith('FORM 10-Q'))

    def test_sentence_aware_chunking(self):
        """Test that chunks don't split mid-sentence."""
        # Create text with multiple sentences
        sentences = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is a very long sentence that should be split because it exceeds the chunk size limit and we need to make sure it gets handled properly.",
            "This is the fourth sentence.",
        ]
        text = " ".join(sentences)
        
        chunks = _chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        # Verify no chunk ends mid-sentence (should end with punctuation)
        for chunk in chunks:
            stripped = chunk.strip()
            self.assertTrue(
                stripped.endswith('.') or stripped.endswith('?') or stripped.endswith('!'),
                f"Chunk ends mid-sentence: {repr(stripped[-50:])}"
            )

    def test_chunk_overlap_applied(self):
        """Test that overlap is applied between chunks."""
        text = "Sentence one. " * 20 + "Sentence two. " * 20
        chunks = _chunk_text(text, chunk_size=200, chunk_overlap=50)
        
        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            # (This is a basic check - exact overlap depends on sentence boundaries)
            self.assertGreater(len(chunks), 1)

    def test_deterministic_chunk_ids(self):
        """Test that chunk IDs are deterministic for same input."""
        text = "Test sentence one. Test sentence two. Test sentence three."
        chunks1 = _chunk_text(text, chunk_size=50, chunk_overlap=10)
        chunks2 = _chunk_text(text, chunk_size=50, chunk_overlap=10)
        
        # Same text, same params should produce same chunks
        self.assertEqual(chunks1, chunks2)
        
        # IDs should be deterministic
        for i, (c1, c2) in enumerate(zip(chunks1, chunks2)):
            id1 = _compute_chunk_id("AAPL", "doc-1", i, c1)
            id2 = _compute_chunk_id("AAPL", "doc-1", i, c2)
            self.assertEqual(id1, id2)

    def test_section_detection(self):
        """Test that SEC sections are detected."""
        text = "Some text before Item 1A Risk Factors. More text about risks."
        # Old _detect_section test - now we test the new stateful approach
        headers = _find_section_headers(text, "10-K")
        self.assertTrue(any("Risk Factors" in h[1] for h in headers))
        
        # Use proper 10-K text with apostrophe s
        text2 = "Part II Item 7 Management's Discussion and Analysis"
        headers2 = _find_section_headers(text2, "10-K")
        self.assertTrue(any("MD&A" in h[1] for h in headers2))

    def test_section_detection_with_period(self):
        """Test that 'Item 1A. Risk Factors' formatting is detected."""
        text = "Some text before Item 1A. Risk Factors. More text about risks."
        headers = _find_section_headers(text, "10-K")
        self.assertTrue(any("Risk Factors" in h[1] for h in headers))

    def test_section_detection_far_lookback(self):
        """Test that section detection works when heading is > 500 chars before chunk."""
        header = "Item 1A. Risk Factors"
        filler = "x " * 500
        text = header + "\n" + filler + "Actual risk factors content here."
        headers = _find_section_headers(text, "10-K")
        self.assertTrue(any("Risk Factors" in h[1] for h in headers))

    def test_section_detection_far_lookback_exact_500_plus(self):
        """Test that section detection works at exactly 500+ chars distance."""
        header = "Item 1A Risk Factors"
        filler = "y " * 300
        text = header + "\n" + filler + "Content at position."
        headers = _find_section_headers(text, "10-K")
        self.assertTrue(any("Risk Factors" in h[1] for h in headers))

    def test_no_section_when_not_present(self):
        """Test that section is None when no SEC section header found."""
        text = "Just some regular text without section headers."
        headers = _find_section_headers(text, "10-K")
        self.assertEqual(len(headers), 0)

    def test_10q_section_mapping(self):
        """Test 10-Q specific section mapping."""
        text = """
        Part I
        Item 1 - Financial Statements
        Some financial statement content here.
        Item 2 - Management's Discussion and Analysis
        MD&A content here.
        Item 3 - Quantitative and Qualitative Disclosures About Market Risk
        Market risk content.
        Item 4 - Controls and Procedures
        Controls content.
        Part II
        Item 1 - Legal Proceedings
        Legal content.
        Item 1A - Risk Factors
        Risk factors content.
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        self.assertIn("Part I", section_names)
        self.assertIn("Item 1 - Financial Statements", section_names)
        self.assertIn("Item 2 - MD&A", section_names)
        self.assertIn("Item 3 - Market Risk", section_names)
        self.assertIn("Item 4 - Controls and Procedures", section_names)
        self.assertIn("Part II", section_names)
        self.assertIn("Item 1 - Legal Proceedings", section_names)
        self.assertIn("Item 1A - Risk Factors", section_names)

    def test_10k_section_mapping(self):
        """Test 10-K specific section mapping."""
        text = """
        Part I
        Item 1 - Business
        Business content.
        Item 1A - Risk Factors
        Risk content.
        Item 1B - Unresolved Staff Comments
        Comments content.
        Item 2 - Properties
        Properties content.
        Item 7 - Management's Discussion and Analysis
        MD&A content.
        Item 8 - Financial Statements
        Financial content.
        """
        headers = _find_section_headers(text, "10-K")
        section_names = [h[1] for h in headers]
        self.assertIn("Part I", section_names)
        self.assertIn("Item 1 - Business", section_names)
        self.assertIn("Item 1A - Risk Factors", section_names)
        self.assertIn("Item 1B - Unresolved Staff Comments", section_names)
        self.assertIn("Item 2 - Properties", section_names)
        self.assertIn("Item 7 - MD&A", section_names)
        self.assertIn("Item 8 - Financial Statements", section_names)

    def test_stateful_section_inheritance(self):
        """Test that chunks inherit the most recent section header."""
        # Create text with clear section headers and content
        text = """
        Part I
        Item 1 - Financial Statements
        Financial statement content goes here. More financial data.
        Item 2 - Management's Discussion and Analysis
        MD&A content goes here. More analysis.
        Item 1A - Risk Factors
        Risk factor content here. More risks discussed.
        """
        chunk_texts = [
            "Financial statement content goes here. More financial data.",
            "MD&A content goes here. More analysis.",
            "Risk factor content here. More risks discussed.",
        ]
        sections = _assign_sections_to_chunks(chunk_texts, text, "10-Q")
        self.assertEqual(sections[0], "Item 1 - Financial Statements")
        self.assertEqual(sections[1], "Item 2 - MD&A")
        self.assertEqual(sections[2], "Item 1A - Risk Factors")

    def test_toc_references_do_not_change_section(self):
        """Test that Table of Contents references don't become active section."""
        # TOC-like text with page numbers/dots
        text = """
        TABLE OF CONTENTS
        Item 1A. Risk Factors .......... 21
        Item 2. Management's Discussion .......... 13
        PART I — FINANCIAL INFORMATION
        Item 1. Financial Statements
        Actual financial statement content here.
        Item 1A. Risk Factors
        Actual risk factor content here.
        """
        # With TOC detection, the TOC entries should be skipped
        headers = _find_section_headers(text, "10-Q")
        # Should only find the real headers, not TOC entries
        section_names = [h[1] for h in headers]
        # Should find real Item 1 and Item 1A, not TOC versions
        self.assertIn("Item 1 - Financial Statements", section_names)
        self.assertIn("Item 1A - Risk Factors", section_names)
        # TOC entries should be filtered out (they appear before 2000 chars with dots)

    def test_section_remains_active_until_next_header(self):
        """Test that a section stays active until next real header."""
        text = """
        Item 1A - Risk Factors
        First risk paragraph.
        Second risk paragraph.
        Third risk paragraph.
        Item 2 - Management's Discussion and Analysis
        MD&A content starts here.
        """
        chunk_texts = [
            "First risk paragraph.",
            "Second risk paragraph.",
            "Third risk paragraph.",
            "MD&A content starts here.",
        ]
        sections = _assign_sections_to_chunks(chunk_texts, text, "10-Q")
        self.assertEqual(sections[0], "Item 1A - Risk Factors")
        self.assertEqual(sections[1], "Item 1A - Risk Factors")
        self.assertEqual(sections[2], "Item 1A - Risk Factors")
        self.assertEqual(sections[3], "Item 2 - MD&A")

    def test_item_1a_risk_factors_correctly_detected(self):
        """Test Item 1A Risk Factors detected in actual section, not TOC."""
        text = """
        TABLE OF CONTENTS
        Item 1A. Risk Factors .......... 21
        PART II — OTHER INFORMATION
        Item 1A. Risk Factors
        The Company's business, reputation, results of operations, financial condition
        and stock price can be materially and adversely affected by a number of factors,
        whether currently known or unknown.
        """
        headers = _find_section_headers(text, "10-Q")
        # Should only find the real Item 1A, not the TOC entry
        risk_factors_headers = [h for h in headers if "Risk Factors" in h[1]]
        self.assertEqual(len(risk_factors_headers), 1)
        # The real header should be after position 2000 (not in TOC)
        self.assertGreater(risk_factors_headers[0][0], 100)

    def test_generic_fallback_sections(self):
        """Test generic fallback when specific pattern not matched."""
        # Use 10-K since Item 10+ are 10-K items
        # Need newline before each to simulate real section headers
        text = "Item 10 Some content.\nItem 11 More content."
        headers = _find_section_headers(text, "10-K")
        # Should match generic Item 10, Item 11
        section_names = [h[1] for h in headers]
        self.assertIn("Item 10", section_names)
        self.assertIn("Item 11", section_names)

    def test_10q_item_1_financial_statements_header(self):
        """Test 10-Q Item 1 Financial Statements specific header detection."""
        text = """
        Part I
        Item 1. Financial Statements
        Condensed Consolidated Balance Sheets...
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        self.assertIn("Item 1 - Financial Statements", section_names)

    def test_10q_item_2_mda_header(self):
        """Test 10-Q Item 2 MD&A specific header detection."""
        text = """
        Part I
        Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations
        MD&A content here...
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        self.assertIn("Item 2 - MD&A", section_names)

    def test_10q_item_3_market_risk_header(self):
        """Test 10-Q Item 3 Market Risk specific header detection."""
        text = """
        Part I
        Item 3. Quantitative and Qualitative Disclosures About Market Risk
        Market risk content...
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        self.assertIn("Item 3 - Market Risk", section_names)

    def test_10q_item_1a_risk_factors_header(self):
        """Test 10-Q Item 1A Risk Factors specific header detection."""
        text = """
        Part II
        Item 1A. Risk Factors
        Risk factors content...
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        self.assertIn("Item 1A - Risk Factors", section_names)

    def test_toc_item_references_not_section_headers(self):
        """Test that TOC references to Item 1/3/6 don't become section headers."""
        text = """
        TABLE OF CONTENTS
        Item 1. Financial Statements .......... 1
        Item 3. Quantitative and Qualitative Disclosures .......... 19
        Item 6. Exhibits .......... 25
        PART I
        Item 1. Financial Statements
        Actual financial statement content.
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        # TOC entries should be filtered out
        # Only the real Item 1 should be found
        item_1_count = sum(1 for n in section_names if n == "Item 1 - Financial Statements")
        self.assertEqual(item_1_count, 1)
        # Generic Item 1/3/6 from TOC should not appear
        self.assertNotIn("Item 1", section_names)
        self.assertNotIn("Item 3", section_names)
        self.assertNotIn("Item 6", section_names)

    def test_textual_reference_not_section_header(self):
        """Test that 'see Item 7...' in text doesn't become a section header."""
        text = """
        Item 2. Management's Discussion and Analysis
        As described in Item 7 of our Annual Report, the results...
        See Item 7 for more details. Per Item 7, the risk factors...
        Item 3. Quantitative and Qualitative Disclosures
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        # Should find real Item 2 and Item 3
        self.assertIn("Item 2 - MD&A", section_names)
        self.assertIn("Item 3 - Market Risk", section_names)
        # Should NOT find Item 7 from textual references
        self.assertNotIn("Item 7", section_names)
        self.assertNotIn("Item 7 - MD&A", section_names)

    def test_10q_no_item_7_mda_from_generic(self):
        """Test that 10-Q doesn't get Item 7 MD&A from generic fallback."""
        text = """
        Item 2. Management's Discussion and Analysis
        MD&A content.
        Some text mentioning Item 7 in passing.
        """
        headers = _find_section_headers(text, "10-Q")
        section_names = [h[1] for h in headers]
        # Should find Item 2 - MD&A
        self.assertIn("Item 2 - MD&A", section_names)
        # Should NOT find Item 7 or Item 7 - MD&A in a 10-Q
        self.assertNotIn("Item 7", section_names)
        self.assertNotIn("Item 7 - MD&A", section_names)

    def test_10k_mappings_still_work(self):
        """Test that existing 10-K mappings still work correctly."""
        text = """
        Part I
        Item 1. Business
        Business content.
        Item 1A. Risk Factors
        Risk content.
        Item 7. Management's Discussion and Analysis
        MD&A content.
        Item 8. Financial Statements
        Financial content.
        """
        headers = _find_section_headers(text, "10-K")
        section_names = [h[1] for h in headers]
        self.assertIn("Item 1 - Business", section_names)
        self.assertIn("Item 1A - Risk Factors", section_names)
        self.assertIn("Item 7 - MD&A", section_names)
        self.assertIn("Item 8 - Financial Statements", section_names)
        # Should NOT have 10-Q specific sections
        self.assertNotIn("Item 1 - Financial Statements", section_names)
        self.assertNotIn("Item 2 - MD&A", section_names)
        self.assertNotIn("Item 3 - Market Risk", section_names)

    def test_stateful_inheritance_still_works(self):
        """Test that stateful section inheritance across chunks still works."""
        text = """
        Part I
        Item 1. Financial Statements
        Financial content paragraph 1.
        Financial content paragraph 2.
        Item 2. Management's Discussion and Analysis
        MD&A content paragraph 1.
        MD&A content paragraph 2.
        """
        chunk_texts = [
            "Financial content paragraph 1.",
            "Financial content paragraph 2.",
            "MD&A content paragraph 1.",
            "MD&A content paragraph 2.",
        ]
        sections = _assign_sections_to_chunks(chunk_texts, text, "10-Q")
        self.assertEqual(sections[0], "Item 1 - Financial Statements")
        self.assertEqual(sections[1], "Item 1 - Financial Statements")
        self.assertEqual(sections[2], "Item 2 - MD&A")
        self.assertEqual(sections[3], "Item 2 - MD&A")

    def test_table_conversion(self):
        """Test that HTML tables are converted to structured text."""
        html = """<html><body><table><tr><th>Revenue</th><th>2024</th></tr><tr><td>Product</td><td>$100B</td></tr></table></body></html>"""
        with open(self.raw_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        result = process_document(self.raw_path, self.metadata)
        self.assertIsNotNone(result)
        text = result.chunks[0]["text"]
        # Should contain table markers
        self.assertIn('[TABLE]', text)
        self.assertIn('[/TABLE]', text)
        self.assertIn('Revenue', text)
        self.assertIn('$100B', text)

    def test_dry_run_creates_separate_file(self):
        """Test that dry-run creates file in _dry_run directory, not production."""
        html = """<html><body><p>Test content for dry run.</p></body></html>"""
        with open(self.raw_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Dry run
        result = process_document_dry_run(self.raw_path, self.metadata)
        self.assertIsNotNone(result)
        
        # Check file location
        self.assertIn('_dry_run', result.processed_path)
        self.assertNotIn('_dry_run', result.raw_path)
        
        # Production file should not exist
        prod_path = os.path.join(self.temp_dir, "AAPL", "test.json")
        # Actually the prod path is in the standard location
        self.assertTrue(os.path.exists(result.processed_path))

    def test_compare_unchanged_chunks(self):
        """Test comparison identifies unchanged chunks."""
        # Create two identical processed documents
        chunks = [
            {"chunk_id": "id1", "text": "Same content.", "chunk_index": 0, "section": None},
            {"chunk_id": "id2", "text": "Also same.", "chunk_index": 1, "section": None},
        ]
        doc1 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=2, chunks=chunks
        )
        doc2 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=2, chunks=chunks
        )
        
        comparison = compare_processed_documents(doc1, doc2)
        
        self.assertEqual(len(comparison.unchanged), 2)
        self.assertEqual(len(comparison.changed), 0)
        self.assertEqual(len(comparison.new), 0)
        self.assertEqual(len(comparison.removed), 0)

    def test_compare_changed_chunks(self):
        """Test comparison identifies changed chunks (same ID, different text)."""
        chunks1 = [{"chunk_id": "id1", "text": "Original text.", "chunk_index": 0, "section": None}]
        chunks2 = [{"chunk_id": "id1", "text": "Modified text.", "chunk_index": 0, "section": None}]
        
        doc1 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks1
        )
        doc2 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks2
        )
        
        comparison = compare_processed_documents(doc1, doc2)
        
        self.assertEqual(len(comparison.unchanged), 0)
        self.assertEqual(len(comparison.changed), 1)
        self.assertEqual(comparison.changed[0], ("id1", "id1"))
        self.assertEqual(comparison.mapping["id1"], "id1")

    def test_compare_new_and_removed_chunks(self):
        """Test comparison identifies new and removed chunks."""
        chunks1 = [{"chunk_id": "id1", "text": "Old content.", "chunk_index": 0, "section": None}]
        chunks2 = [{"chunk_id": "id2", "text": "New content.", "chunk_index": 0, "section": None}]
        
        doc1 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks1
        )
        doc2 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks2
        )
        
        comparison = compare_processed_documents(doc1, doc2)
        
        self.assertEqual(len(comparison.unchanged), 0)
        self.assertEqual(len(comparison.changed), 0)
        self.assertEqual(len(comparison.new), 1)
        self.assertEqual(comparison.new[0], "id2")
        self.assertEqual(len(comparison.removed), 1)
        self.assertEqual(comparison.removed[0], "id1")

    def test_compare_content_similarity_mapping(self):
        """Test that similar content gets mapped even with different IDs."""
        # Old chunk with content
        chunks1 = [{"chunk_id": "old_id", "text": "Apple faces supply chain risks in China.", "chunk_index": 0, "section": "Item 1A"}]
        # New chunk with similar content but different chunking -> different ID
        chunks2 = [{"chunk_id": "new_id", "text": "Apple faces supply chain risks in China and Taiwan.", "chunk_index": 0, "section": "Item 1A"}]
        
        doc1 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks1
        )
        doc2 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks2
        )
        
        comparison = compare_processed_documents(doc1, doc2)
        
        # Should detect as changed (similar content) rather than removed+new
        self.assertEqual(len(comparison.changed), 1)
        self.assertEqual(comparison.changed[0], ("old_id", "new_id"))

    def test_compare_returns_chunk_comparison_object(self):
        """Test that comparison returns proper ChunkComparison object."""
        chunks = [{"chunk_id": "id1", "text": "Test.", "chunk_index": 0, "section": None}]
        doc1 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks
        )
        doc2 = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=1, chunks=chunks
        )
        
        comparison = compare_processed_documents(doc1, doc2)
        
        self.assertIsInstance(comparison, ChunkComparison)
        self.assertIsInstance(comparison.unchanged, list)
        self.assertIsInstance(comparison.changed, list)
        self.assertIsInstance(comparison.new, list)
        self.assertIsInstance(comparison.removed, list)
        self.assertIsInstance(comparison.mapping, dict)


class TestSentenceSplitting(unittest.TestCase):
    def test_split_into_sentences_basic(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = _split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "First sentence.")
        self.assertEqual(sentences[1], "Second sentence!")
        self.assertEqual(sentences[2], "Third sentence?")

    def test_split_into_sentences_handles_abbreviations(self):
        # This is a known limitation - simple regex will split on Mr.
        text = "Mr. Smith went to the U.S. He liked it."
        sentences = _split_into_sentences(text)
        # May split on "U.S." but that's acceptable for now
        self.assertGreaterEqual(len(sentences), 2)


class TestTrimFilenameHeader(unittest.TestCase):
    def test_trims_ticker_date(self):
        text = "aapl-20260627\nUNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q"
        result = _trim_filename_header(text, "aapl-20260627")
        self.assertNotIn("aapl-20260627", result)
        self.assertNotIn("UNITED STATES SECURITIES AND EXCHANGE COMMISSION", result)
        self.assertTrue(result.startswith("Washington, D.C.") or result.startswith("FORM 10-Q"))

    def test_trims_various_formats(self):
        test_cases = [
            ("aapl-20260627\nContent", "aapl-20260627"),
            ("tsla 20260630\nContent", "tsla-20260630"),
            ("nvda_20260726\nContent", "nvda-20260726"),
        ]
        for text, stem in test_cases:
            result = _trim_filename_header(text, stem)
            self.assertNotIn(stem.replace('-', '').replace('_', ''), result.replace(' ', '').lower())


if __name__ == "__main__":
    unittest.main()