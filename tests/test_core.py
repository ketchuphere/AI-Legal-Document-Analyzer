"""
tests/test_core.py
==================
Unit and integration tests for the AI Document Analyzer.
Run with:  pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_CONTRACT = """
CONSULTING SERVICES AGREEMENT

This Consulting Services Agreement ("Agreement") is entered into as of January 1, 2025,
by and between Acme Corp ("Client"), a Delaware corporation, and John Doe ("Consultant").

1. SERVICES. Consultant agrees to provide software development services as requested.

2. PAYMENT. Client shall pay Consultant $150/hour, invoiced monthly, due net 30.

3. TERM. This Agreement commences January 1, 2025 and terminates December 31, 2025,
   unless earlier terminated.

4. TERMINATION. Either party may terminate with 14 days written notice.

5. CONFIDENTIALITY. Consultant shall not disclose Client's proprietary information.

6. GOVERNING LAW. This Agreement is governed by the laws of Delaware.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Text splitter tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTextSplitter:
    def test_short_document_single_chunk(self):
        from src.utils.text_splitter import split_text
        text   = "A" * 100
        chunks = split_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_document_multiple_chunks(self):
        from src.utils.text_splitter import split_text
        text   = "word " * 2000   # 10000 chars
        chunks = split_text(text, chunk_size=3000, overlap=200)
        assert len(chunks) > 1
        # Every chunk should be non-empty
        assert all(len(c) > 0 for c in chunks)

    def test_overlap_present(self):
        from src.utils.text_splitter import split_text
        # Build text with identifiable paragraphs
        text   = ("Para one content.\n\n" * 100)
        chunks = split_text(text, chunk_size=500, overlap=50)
        # There should be content shared between consecutive chunks (overlap)
        assert len(chunks) >= 2

    def test_estimate_tokens(self):
        from src.utils.text_splitter import estimate_tokens
        assert estimate_tokens("a" * 400) == 100


# ──────────────────────────────────────────────────────────────────────────────
# Document loader tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_basic_metadata_word_count(self):
        from src.utils.document_loader import DocumentLoader
        meta = DocumentLoader.basic_metadata("hello world " * 100)
        assert meta["word_count"] == "200"

    def test_basic_metadata_doc_type_lease(self):
        from src.utils.document_loader import DocumentLoader
        text = "The landlord and tenant agree on this lease and rent for the property."
        meta = DocumentLoader.basic_metadata(text)
        assert meta["doc_type"] == "Lease Agreement"

    def test_basic_metadata_doc_type_nda(self):
        from src.utils.document_loader import DocumentLoader
        text = "This Non-Disclosure Agreement (NDA) covers all confidential information."
        meta = DocumentLoader.basic_metadata(text)
        assert meta["doc_type"] == "NDA"

    def test_basic_metadata_year_extraction(self):
        from src.utils.document_loader import DocumentLoader
        text = "This agreement is dated January 2024 and is valid."
        meta = DocumentLoader.basic_metadata(text)
        assert meta["doc_date"] == "2024"

    def test_load_txt(self):
        from src.utils.document_loader import DocumentLoader
        loader = DocumentLoader()
        mock   = MagicMock()
        mock.name = "contract.txt"
        mock.getvalue.return_value = b"Hello legal world"
        text = loader.load(mock)
        assert text == "Hello legal world"

    def test_unsupported_format_raises(self):
        from src.utils.document_loader import DocumentLoader
        loader = DocumentLoader()
        mock   = MagicMock()
        mock.name = "contract.xls"
        mock.getvalue.return_value = b""
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(mock)


# ──────────────────────────────────────────────────────────────────────────────
# LLM client tests (mocked)
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMClient:
    @patch("anthropic.Anthropic")
    def test_call_returns_text(self, mock_anthropic_cls):
        from src.utils.llm_client import LLMClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="  Test response  ")]
        )

        llm    = LLMClient(api_key="test-key", model="claude-sonnet-4-20250514")
        result = llm.call("Test prompt")
        assert result == "Test response"   # whitespace stripped

    @patch("anthropic.Anthropic")
    def test_call_with_history(self, mock_anthropic_cls):
        from src.utils.llm_client import LLMClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Answer")]
        )

        llm    = LLMClient(api_key="test-key")
        result = llm.call_with_history([
            {"role": "user",      "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user",      "content": "Q2"},
        ])
        assert result == "Answer"


# ──────────────────────────────────────────────────────────────────────────────
# Agent node tests (mocked LLM)
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentNodes:
    def _mock_llm(self, return_value: str):
        """Return a patched LLMClient that always returns `return_value`."""
        mock = MagicMock()
        mock.call.return_value = return_value
        return mock

    def test_summarize_node(self):
        from src.agents.workflow import summarize_node
        with patch("src.agents.workflow._client") as mock_client_fn:
            mock_client_fn.return_value = self._mock_llm("• Point 1\n• Point 2")
            state  = {"original_text": SAMPLE_CONTRACT, "model": "test"}
            result = summarize_node(state)
            assert "summary" in result
            assert len(result["summary"]) > 0

    def test_analyze_risks_node_valid_json(self):
        from src.agents.workflow import analyze_risks_node
        risks_json = json.dumps([{
            "title": "Termination Risk",
            "why": "Short notice period",
            "severity": "Med",
            "likelihood": "Med",
            "mitigation": "Extend notice to 30 days"
        }])
        with patch("src.agents.workflow._client") as mock_client_fn:
            mock_client_fn.return_value = self._mock_llm(risks_json)
            state  = {"original_text": SAMPLE_CONTRACT, "model": "test"}
            result = analyze_risks_node(state)
            assert "risks_structured" in result
            assert isinstance(result["risks_structured"], list)
            assert len(result["risks_structured"]) == 1
            assert result["risks_structured"][0]["title"] == "Termination Risk"

    def test_analyze_risks_node_bad_json_fallback(self):
        from src.agents.workflow import analyze_risks_node
        with patch("src.agents.workflow._client") as mock_client_fn:
            mock_client_fn.return_value = self._mock_llm("Not valid JSON at all")
            state  = {"original_text": SAMPLE_CONTRACT, "model": "test"}
            result = analyze_risks_node(state)
            # Should not raise; falls back to empty list
            assert result["risks_structured"] == []

    def test_extract_clauses_node(self):
        from src.agents.workflow import extract_clauses_node
        clauses_json = json.dumps({
            "parties": "Acme Corp and John Doe",
            "effective_date": "January 1, 2025",
            "governing_law": "Delaware",
        })
        with patch("src.agents.workflow._client") as mock_client_fn:
            mock_client_fn.return_value = self._mock_llm(clauses_json)
            state  = {"original_text": SAMPLE_CONTRACT, "model": "test"}
            result = extract_clauses_node(state)
            assert result["clauses"]["parties"] == "Acme Corp and John Doe"

    def test_compliance_check_node(self):
        from src.agents.workflow import compliance_check_node
        comp_json = json.dumps([
            {"check": "Parties clearly identified", "status": "PASS", "note": "Both parties named"},
            {"check": "Payment terms clear",        "status": "PASS", "note": "$150/hr net 30"},
        ])
        with patch("src.agents.workflow._client") as mock_client_fn:
            mock_client_fn.return_value = self._mock_llm(comp_json)
            state  = {"original_text": SAMPLE_CONTRACT, "model": "test"}
            result = compliance_check_node(state)
            assert len(result["compliance_items"]) == 2
            assert result["compliance_items"][0]["status"] == "PASS"


# ──────────────────────────────────────────────────────────────────────────────
# Export tests
# ──────────────────────────────────────────────────────────────────────────────

class TestExport:
    def _sample_result(self):
        return {
            "doc_name":          "test_contract.pdf",
            "summary":           "• Point 1\n• Point 2",
            "risks_structured":  [{"title": "Risk A", "severity": "High", "likelihood": "Med", "mitigation": "Mitigate X"}],
            "clauses":           {"parties": "A and B", "governing_law": "Delaware"},
            "compliance_items":  [{"check": "Parties", "status": "PASS", "note": "Clear"}],
            "suggestions":       "Add indemnification clause.",
            "final_report":      "# Report\n\nContent here.",
            "doc_metadata":      {"word_count": "100", "page_count": "2", "doc_type": "Contract", "doc_date": "2025"},
        }

    def test_export_docx_returns_bytes(self):
        from src.utils.export import export_report_docx
        result = self._sample_result()
        docx_bytes = export_report_docx(result)
        assert isinstance(docx_bytes, bytes)
        assert len(docx_bytes) > 0
        # DOCX files start with PK (ZIP magic bytes)
        assert docx_bytes[:2] == b"PK"


# ──────────────────────────────────────────────────────────────────────────────
# Q&A chain tests
# ──────────────────────────────────────────────────────────────────────────────

class TestQAChain:
    @patch("src.agents.qa_chain.LLMClient")
    def test_ask_returns_string(self, mock_llm_cls):
        from src.agents.qa_chain import QAChain
        mock_llm = MagicMock()
        mock_llm.call_with_history.return_value = "The payment is $150/hr."
        mock_llm_cls.return_value = mock_llm

        chain  = QAChain(doc_text=SAMPLE_CONTRACT, model="test", api_key="key")
        answer = chain.ask("What is the payment rate?")
        assert answer == "The payment is $150/hr."

    @patch("src.agents.qa_chain.LLMClient")
    def test_ask_handles_error(self, mock_llm_cls):
        from src.agents.qa_chain import QAChain
        mock_llm = MagicMock()
        mock_llm.call_with_history.side_effect = Exception("API down")
        mock_llm_cls.return_value = mock_llm

        chain  = QAChain(doc_text=SAMPLE_CONTRACT, model="test", api_key="key")
        answer = chain.ask("What is the payment rate?")
        assert "error" in answer.lower()
