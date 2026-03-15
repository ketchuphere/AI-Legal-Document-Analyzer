"""
src/agents/workflow.py
======================
Defines and compiles the LangGraph multi-agent analysis workflow.

Pipeline (sequential):
    condense → summarize → analyze_risks → extract_clauses
        → compliance_check → suggest_improvements → compile_report

Each node receives the full AgentState, enriches it with its output,
and returns the updated keys. LangGraph merges updates automatically.
"""

import json
import logging
from textwrap import dedent
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.config.settings import Settings
from src.utils.llm_client import LLMClient
from src.utils.text_splitter import split_text
from src.utils.document_loader import DocumentLoader

logger   = logging.getLogger(__name__)
settings = Settings()


# ──────────────────────────────────────────────────────────────────────────────
# Shared state schema
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # Inputs
    original_text:  str
    doc_name:       str
    model:          str
    chunk_chars:    int
    chunk_overlap:  int

    # Intermediate
    condensed_text: str
    doc_metadata:   dict

    # Analysis outputs
    summary:             str
    risks:               str
    risks_structured:    list    # list of dicts
    clauses:             dict
    compliance_items:    list    # list of dicts
    suggestions:         str

    # Final
    final_report:   str


# ──────────────────────────────────────────────────────────────────────────────
# Helper — build a client from state
# ──────────────────────────────────────────────────────────────────────────────

def _client(state: AgentState) -> LLMClient:
    return LLMClient(
        api_key=settings.ANTHROPIC_API_KEY,
        model=state.get("model", settings.DEFAULT_MODEL),
        max_tokens=settings.MAX_TOKENS_PER_CALL,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node 1: Condense (map-reduce for long documents)
# ──────────────────────────────────────────────────────────────────────────────

def condense_node(state: AgentState) -> Dict[str, Any]:
    """
    Map-reduce text condensation.

    - Splits the document into manageable chunks.
    - Summarises each chunk individually (map).
    - Consolidates all chunk summaries into a single condensed text (reduce).
    - Also computes basic document metadata (word count, type, etc.).
    """
    text   = state["original_text"]
    llm    = _client(state)
    chunks = split_text(
        text,
        chunk_size=state.get("chunk_chars", settings.DEFAULT_CHUNK_CHARS),
        overlap=state.get("chunk_overlap", settings.DEFAULT_CHUNK_OVERLAP),
    )
    logger.info("Condense: %d chunks from %d chars", len(chunks), len(text))

    if len(chunks) == 1:
        # Short document — no condensation needed
        condensed = text
    else:
        # MAP: summarise each chunk
        partials = []
        for i, chunk in enumerate(chunks, 1):
            prompt = dedent(f"""
            You are an expert document analyst.
            Summarise chunk {i}/{len(chunks)} of a legal/business document.
            Preserve: parties, obligations, dates, payment terms, termination, liability,
            IP, confidentiality, governing law, dispute resolution.
            Output concise bullet points only.

            Chunk:
            {chunk}
            """)
            partials.append(llm.call(prompt, temperature=settings.SUMMARY_TEMPERATURE))

        # REDUCE: merge all chunk summaries
        reduce_prompt = dedent(f"""
        You are an expert document analyst.
        Consolidate these chunk summaries into ONE coherent, non-redundant condensed document.
        Use these headings:
        - Parties & Purpose
        - Key Obligations
        - Money (fees, payment terms, penalties)
        - Term, Renewal, Termination
        - Liability, Indemnity, Insurance
        - IP & Confidentiality
        - Disputes & Governing Law
        - Important Deadlines & Dates

        Chunk summaries:
        {"---".join(partials)}
        """)
        condensed = llm.call(reduce_prompt, temperature=settings.SUMMARY_TEMPERATURE)

    # Metadata
    meta = DocumentLoader.basic_metadata(text, state.get("doc_name", ""))

    return {"condensed_text": condensed, "doc_metadata": meta}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2: Summarize
# ──────────────────────────────────────────────────────────────────────────────

def summarize_node(state: AgentState) -> Dict[str, Any]:
    """Generate a concise executive summary (5–12 bullet points)."""
    text   = state.get("condensed_text") or state["original_text"]
    llm    = _client(state)
    prompt = dedent(f"""
    You are an expert document analyst and legal assistant.
    Write a concise executive summary of this document (5–12 bullet points maximum).
    Focus on: parties, main purpose, key commitments, important conditions, and practical meaning.
    Use clear, plain language. Each bullet should be 1–2 sentences.

    Document:
    {text}
    """)
    summary = llm.call(prompt, temperature=settings.SUMMARY_TEMPERATURE)
    logger.info("Summary generated (%d chars)", len(summary))
    return {"summary": summary}


# ──────────────────────────────────────────────────────────────────────────────
# Node 3: Risk Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_risks_node(state: AgentState) -> Dict[str, Any]:
    """
    Identify legal/business risks and return both a markdown string
    and a structured list of risk dicts for the UI cards.
    """
    text   = state.get("condensed_text") or state["original_text"]
    llm    = _client(state)

    # Ask for JSON output so we can render structured risk cards
    prompt = dedent(f"""
    You are an expert legal risk analyst.
    Identify all key legal and business risks in this document.

    Return a JSON array (and ONLY the JSON array — no prose before or after) where each item has:
    {{
      "title":       "<short risk name>",
      "why":         "<why this matters>",
      "severity":    "High" | "Med" | "Low",
      "likelihood":  "High" | "Med" | "Low",
      "mitigation":  "<one-line suggested mitigation>"
    }}

    Document:
    {text}
    """)

    raw = llm.call(prompt, temperature=settings.RISKS_TEMPERATURE)

    # Parse JSON, fall back gracefully
    risks_structured: list = []
    risks_text = raw
    try:
        # Strip code fences if present
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        risks_structured = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Risk JSON parse failed (%s). Using raw text.", e)
        risks_structured = []

    return {"risks": risks_text, "risks_structured": risks_structured}


# ──────────────────────────────────────────────────────────────────────────────
# Node 4: Clause Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_clauses_node(state: AgentState) -> Dict[str, Any]:
    """
    Extract key clause values as a structured JSON dictionary.
    Returns an empty dict if extraction fails.
    """
    text   = state.get("condensed_text") or state["original_text"]
    llm    = _client(state)
    prompt = dedent(f"""
    You are a legal data extraction specialist.
    From the document below, extract the following fields.
    Return ONLY a valid JSON object (no prose, no markdown fences).

    Fields to extract:
    - parties           (string: names of all parties)
    - effective_date    (string: when the agreement starts)
    - termination       (string: termination conditions/notice period)
    - payment           (string: payment terms, amounts, schedule)
    - confidentiality   (string: confidentiality obligations)
    - ip_ownership      (string: who owns intellectual property)
    - governing_law     (string: governing jurisdiction)
    - dispute           (string: dispute resolution mechanism)
    - warranties        (string: warranties given)
    - liability_cap     (string: limitation of liability clause)

    For each field, if not found write null.

    Document:
    {text}
    """)

    raw = llm.call(prompt, temperature=settings.CLAUSES_TEMPERATURE)
    clauses: dict = {}
    try:
        clean   = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        clauses = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Clause JSON parse failed (%s). Returning empty dict.", e)

    return {"clauses": clauses}


# ──────────────────────────────────────────────────────────────────────────────
# Node 5: Compliance Check
# ──────────────────────────────────────────────────────────────────────────────

def compliance_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Check the document against a standard compliance checklist.
    Returns a list of {check, status, note} dicts.
    """
    text   = state.get("condensed_text") or state["original_text"]
    llm    = _client(state)
    prompt = dedent(f"""
    You are a legal compliance specialist.
    Evaluate this document against the checklist below.
    For each item output PASS, FAIL, or WARN.

    Return ONLY a JSON array (no prose). Each item:
    {{
      "check":  "<checklist item name>",
      "status": "PASS" | "FAIL" | "WARN",
      "note":   "<brief explanation>"
    }}

    Checklist:
    1. Parties clearly identified
    2. Effective date specified
    3. Term/duration defined
    4. Payment terms clear
    5. Termination clause present
    6. Confidentiality clause present
    7. Intellectual property ownership addressed
    8. Limitation of liability present
    9. Governing law specified
    10. Dispute resolution mechanism present
    11. Signature / execution block present
    12. Force majeure clause present

    Document:
    {text}
    """)

    raw = llm.call(prompt, temperature=settings.COMPLIANCE_TEMPERATURE)
    items: list = []
    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        items = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Compliance JSON parse failed (%s). Returning empty list.", e)

    return {"compliance_items": items}


# ──────────────────────────────────────────────────────────────────────────────
# Node 6: Suggest Improvements
# ──────────────────────────────────────────────────────────────────────────────

def suggest_improvements_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate specific, clause-level improvement suggestions organised by topic.
    """
    text   = state.get("condensed_text") or state["original_text"]
    clauses = state.get("clauses", {})
    llm    = _client(state)

    context = f"Known gaps from clause extraction: {json.dumps(clauses, indent=2)}" if clauses else ""

    prompt = dedent(f"""
    You are a senior legal counsel reviewing a document for a client.
    Provide specific improvement suggestions at the clause level.
    Organise your suggestions under these headings:
    - Payment & Fees
    - Termination & Exit
    - Limitation of Liability
    - Indemnification
    - Confidentiality & Data
    - Intellectual Property
    - Dispute Resolution
    - Warranties & Representations
    - Missing Clauses (any important provisions completely absent)

    For each suggestion, state: what to add/change and why it matters.

    {context}

    Document:
    {text}
    """)

    suggestions = llm.call(prompt, temperature=settings.SUGGESTIONS_TEMPERATURE)
    return {"suggestions": suggestions}


# ──────────────────────────────────────────────────────────────────────────────
# Node 7: Compile Final Report
# ──────────────────────────────────────────────────────────────────────────────

def compile_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Assemble all analysis outputs into a single Markdown report.
    """
    doc_name = state.get("doc_name", "Unnamed Document")
    meta     = state.get("doc_metadata", {})

    # Format structured risks as markdown table
    risks_md = state.get("risks", "")
    structured = state.get("risks_structured", [])
    if structured:
        risks_md = "| # | Risk | Severity | Likelihood | Mitigation |\n"
        risks_md += "|---|------|----------|------------|------------|\n"
        for i, r in enumerate(structured, 1):
            risks_md += (
                f"| {i} | {r.get('title','—')} | {r.get('severity','—')} "
                f"| {r.get('likelihood','—')} | {r.get('mitigation','—')} |\n"
            )

    # Format clauses as markdown table
    clauses    = state.get("clauses", {})
    clauses_md = ""
    if clauses:
        clauses_md = "| Clause | Extracted Value |\n|--------|----------------|\n"
        for k, v in clauses.items():
            clauses_md += f"| {k.replace('_',' ').title()} | {v or '_Not specified_'} |\n"

    # Format compliance checklist
    comp_items = state.get("compliance_items", [])
    comp_md    = ""
    if comp_items:
        for item in comp_items:
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(item.get("status",""), "❓")
            comp_md += f"- {icon} **{item.get('check','')}**: {item.get('note','')}\n"

    from datetime import datetime
    report = dedent(f"""
    # 📄 AI Document Analysis Report

    > **Disclaimer:** This report is AI-assisted and does not constitute legal advice.
    > Consult a qualified attorney before acting on any information herein.

    **Document:** {doc_name}
    **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
    **Words:** {meta.get("word_count","—")} | **Pages (approx):** {meta.get("page_count","—")}
    **Detected Type:** {meta.get("doc_type","—")}

    ---

    ## 📝 Executive Summary

    {state.get("summary", "_No summary generated._")}

    ---

    ## ⚠️ Risk Analysis

    {risks_md or "_No risks identified._"}

    ---

    ## 📑 Key Clause Extraction

    {clauses_md or "_No clause data extracted._"}

    ---

    ## ✅ Compliance Checklist

    {comp_md or "_No compliance data available._"}

    ---

    ## 💡 Improvement Suggestions

    {state.get("suggestions", "_No suggestions generated._")}

    ---

    *Report generated by AI Document Analyzer — powered by Anthropic Claude*
    """).strip()

    return {"final_report": report}


# ──────────────────────────────────────────────────────────────────────────────
# Graph factory
# ──────────────────────────────────────────────────────────────────────────────

def build_workflow():
    """
    Build and compile the LangGraph StateGraph.

    Graph edges (all sequential):
        condense → summarize → analyze_risks → extract_clauses
            → compliance_check → suggest_improvements → compile_report → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("condense",             condense_node)
    graph.add_node("summarize",            summarize_node)
    graph.add_node("analyze_risks",        analyze_risks_node)
    graph.add_node("extract_clauses",      extract_clauses_node)
    graph.add_node("compliance_check",     compliance_check_node)
    graph.add_node("suggest_improvements", suggest_improvements_node)
    graph.add_node("compile_report",       compile_report_node)

    # Wire the pipeline
    graph.set_entry_point("condense")
    graph.add_edge("condense",             "summarize")
    graph.add_edge("summarize",            "analyze_risks")
    graph.add_edge("analyze_risks",        "extract_clauses")
    graph.add_edge("extract_clauses",      "compliance_check")
    graph.add_edge("compliance_check",     "suggest_improvements")
    graph.add_edge("suggest_improvements", "compile_report")
    graph.add_edge("compile_report",       END)

    return graph.compile()
