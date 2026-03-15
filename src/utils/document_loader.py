"""
src/utils/document_loader.py
============================
Handles loading and text extraction from PDF, TXT, and DOCX files.
Returns a clean unicode string ready for the analysis pipeline.
"""

import logging
import os
import tempfile

import pymupdf4llm
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents from Streamlit UploadedFile objects.

    Supported formats:
        - .pdf  → pymupdf4llm (preserves layout as Markdown)
        - .txt  → raw UTF-8 decode
        - .docx → python-docx paragraph extraction
    """

    SUPPORTED = {"pdf", "txt", "docx"}

    # ──────────────────────────────────────────────────────────────────────
    def load(self, uploaded_file) -> str:
        """
        Extract text from an uploaded file.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Extracted text as a unicode string.

        Raises:
            ValueError: If the file extension is not supported.
        """
        suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: .{suffix}")

        data = uploaded_file.getvalue()
        logger.info("Loading document '%s' (%d bytes)", uploaded_file.name, len(data))

        if suffix == "txt":
            return self._load_txt(data)
        elif suffix == "pdf":
            return self._load_pdf(data)
        elif suffix == "docx":
            return self._load_docx(data)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_txt(data: bytes) -> str:
        """Decode raw bytes as UTF-8, replacing errors."""
        text = data.decode("utf-8", errors="replace").strip()
        logger.debug("TXT: %d chars extracted", len(text))
        return text

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_pdf(data: bytes) -> str:
        """Convert PDF bytes to Markdown text via pymupdf4llm."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            text = pymupdf4llm.to_markdown(tmp_path)
            logger.debug("PDF: %d chars extracted", len(text))
            return text.strip()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_docx(data: bytes) -> str:
        """Extract paragraphs from a DOCX file using python-docx."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            doc   = DocxDocument(tmp_path)
            lines = [p.text for p in doc.paragraphs if p.text.strip()]
            text  = "\n".join(lines)
            logger.debug("DOCX: %d paragraphs / %d chars extracted", len(lines), len(text))
            return text
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def basic_metadata(text: str, filename: str = "") -> dict:
        """
        Compute lightweight metadata from extracted text.

        Returns a dict with: word_count, page_count (approx), doc_type, doc_date.
        """
        words = len(text.split())
        pages = max(1, round(words / 400))   # rough heuristic

        # Very simple document-type classifier
        lower = text.lower()
        if any(k in lower for k in ["lease", "rent", "landlord", "tenant"]):
            doc_type = "Lease Agreement"
        elif any(k in lower for k in ["employment", "employee", "employer", "salary"]):
            doc_type = "Employment Contract"
        elif any(k in lower for k in ["service agreement", "services", "client", "vendor"]):
            doc_type = "Service Agreement"
        elif any(k in lower for k in ["non-disclosure", "nda", "confidential"]):
            doc_type = "NDA"
        elif any(k in lower for k in ["purchase", "sale", "buyer", "seller"]):
            doc_type = "Purchase Agreement"
        else:
            doc_type = "Legal Document"

        # Attempt to find a date mention (YYYY format)
        import re
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        doc_date   = year_match.group(0) if year_match else "—"

        return {
            "word_count": f"{words:,}",
            "page_count": str(pages),
            "doc_type":   doc_type,
            "doc_date":   doc_date,
        }
