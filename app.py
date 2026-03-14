"""
AI Document Analyzer — Main Streamlit Application
===================================================
Entry point for the web interface.
Orchestrates document upload, analysis pipeline, Q&A chat, and report export.
"""

import hashlib
import os
import time
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.config.settings import Settings
from src.utils.document_loader import DocumentLoader
from src.utils.export import export_report_docx
from src.agents.workflow import build_workflow
from src.agents.qa_chain import QAChain


#Bootstrap
load_dotenv()
settings = Settings()

st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title   { font-size: 2.2rem; font-weight: 700; }
    .sub-title    { font-size: 1rem; color: #888; margin-top: -0.5rem; }
    .metric-card  { padding: 1rem; border-radius: 10px;
                    border: 1px solid rgba(255,255,255,0.12);
                    background: rgba(255,255,255,0.03); }
    .risk-high    { color: #ff4b4b; font-weight: 600; }
    .risk-med     { color: #ffa600; font-weight: 600; }
    .risk-low     { color: #21c354; font-weight: 600; }
    .chat-bubble  { padding: 0.6rem 1rem; border-radius: 12px;
                    margin-bottom: 0.4rem; max-width: 85%; }
    .chat-user    { background: rgba(99,102,241,0.15); margin-left: auto; }
    .chat-bot     { background: rgba(255,255,255,0.06); }
    div[data-testid="stTabs"] button { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────
def _init_session():
    """Initialise default session-state keys once per session."""
    defaults = {
        "results_cache":   {},   # hash → analysis result dict
        "qa_history":      [],   # list of {"role": ..., "content": ...}
        "qa_chain":        None, # QAChain instance bound to current doc
        "current_hash":    None,
        "doc_text":        "",
        "doc_name":        "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar() -> object | None:
    """Render sidebar controls; return uploaded file object or None."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.subheader("📂 Upload Document")
        uploaded = st.file_uploader(
            "Supported: PDF, TXT, DOCX",
            type=["pdf", "txt", "docx"],
            help="Upload a legal, business, or any document for AI analysis.",
        )

        st.divider()
        st.subheader("🤖 Model Settings")
        model_choice = st.selectbox(
            "Claude Model",
            options=list(settings.AVAILABLE_MODELS.keys()),
            index=0,
            help="Larger models give richer analysis at higher cost.",
        )
        st.session_state["model"] = settings.AVAILABLE_MODELS[model_choice]

        st.divider()
        st.subheader("🔧 Advanced Options")
        st.session_state["max_chunk_chars"] = st.number_input(
            "Max chars per chunk", value=settings.DEFAULT_CHUNK_CHARS, step=500,
            help="Lower this for small-context models."
        )
        st.session_state["chunk_overlap"] = st.number_input(
            "Chunk overlap", value=settings.DEFAULT_CHUNK_OVERLAP, step=50
        )
        st.session_state["enable_qa"] = st.toggle(
            "Enable Document Q&A Chat", value=True,
            help="Ask follow-up questions after analysis."
        )

        st.divider()
        st.caption("📋 AI Document Analyzer v2.0")
        st.caption("⚠️ Not legal advice. Consult a qualified attorney.")

    return uploaded


# ──────────────────────────────────────────────
# Core analysis pipeline
# ──────────────────────────────────────────────
def run_analysis(doc_text: str, doc_name: str) -> dict:
    """
    Execute the full multi-agent LangGraph workflow.

    Pipeline:
        condense → summarize → risk_analysis → clause_extraction
            → compliance_check → suggest_improvements → compile_report

    Returns the final state dict from the graph.
    """
    workflow = build_workflow()

    with st.status("🔍 Analysing document…", expanded=True) as status:
        st.write("📖 Extracting and condensing text…")
        time.sleep(0.3)   # UX breathing room

        st.write("✍️  Generating executive summary…")
        state_in = {
            "original_text":  doc_text,
            "doc_name":       doc_name,
            "model":          st.session_state["model"],
            "chunk_chars":    int(st.session_state["max_chunk_chars"]),
            "chunk_overlap":  int(st.session_state["chunk_overlap"]),
        }

        result = workflow.invoke(state_in)

        st.write("⚠️  Identifying risks…")
        st.write("📑  Extracting key clauses…")
        st.write("✅  Running compliance checks…")
        st.write("💡  Generating improvement suggestions…")
        st.write("📊  Compiling final report…")
        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    return result


# ──────────────────────────────────────────────
# Result display helpers
# ──────────────────────────────────────────────
def _confidence_badge(score: float) -> str:
    colour = "#21c354" if score >= 0.75 else ("#ffa600" if score >= 0.5 else "#ff4b4b")
    return (f'<span style="background:{colour};color:#fff;padding:2px 8px;'
            f'border-radius:12px;font-size:0.8rem;">{score:.0%} confidence</span>')


def display_summary(result: dict):
    st.markdown("### 📝 Executive Summary")
    st.info(result.get("summary", "No summary generated."))
    # Key metadata metrics
    meta = result.get("doc_metadata", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Words",    meta.get("word_count",  "—"))
    c2.metric("📃 Pages",    meta.get("page_count",  "—"))
    c3.metric("🗓️ Date",     meta.get("doc_date",    "—"))
    c4.metric("⚖️  Type",    meta.get("doc_type",    "—"))


def display_risks(result: dict):
    st.markdown("### ⚠️ Risk Analysis")
    risks = result.get("risks_structured", [])
    if not risks:
        st.markdown(result.get("risks", "No risks identified."))
        return

    # Aggregate severity counts
    sev_counts = {"High": 0, "Med": 0, "Low": 0}
    for r in risks:
        s = r.get("severity", "Low")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Severity", sev_counts["High"])
    c2.metric("🟡 Med Severity",  sev_counts["Med"])
    c3.metric("🟢 Low Severity",  sev_counts["Low"])

    st.divider()
    for i, risk in enumerate(risks, 1):
        sev   = risk.get("severity", "Low")
        color = {"High": "🔴", "Med": "🟡", "Low": "🟢"}.get(sev, "⚪")
        with st.expander(f"{color} Risk {i}: {risk.get('title', 'Unnamed Risk')}"):
            st.markdown(f"**Why it matters:** {risk.get('why', '—')}")
            st.markdown(f"**Severity:** `{sev}` | **Likelihood:** `{risk.get('likelihood','—')}`")
            st.markdown(f"**Mitigation:** {risk.get('mitigation', '—')}")


def display_clauses(result: dict):
    st.markdown("### 📑 Key Clause Extraction")
    clauses = result.get("clauses", {})
    if not clauses:
        st.info("No structured clause data extracted.")
        return

    label_map = {
        "parties":          "👥 Parties",
        "effective_date":   "📅 Effective Date",
        "termination":      "🚫 Termination",
        "payment":          "💰 Payment Terms",
        "confidentiality":  "🔒 Confidentiality",
        "ip_ownership":     "💡 IP Ownership",
        "governing_law":    "⚖️  Governing Law",
        "dispute":          "🏛️  Dispute Resolution",
        "warranties":       "📋 Warranties",
        "liability_cap":    "🛡️  Liability Cap",
    }

    cols = st.columns(2)
    for idx, (key, label) in enumerate(label_map.items()):
        val = clauses.get(key, "Not found")
        with cols[idx % 2]:
            with st.container(border=True):
                st.caption(label)
                st.write(val if val else "_Not specified_")


def display_compliance(result: dict):
    st.markdown("### ✅ Compliance Checklist")
    items = result.get("compliance_items", [])
    if not items:
        st.info("No compliance data available.")
        return

    pass_count = sum(1 for i in items if i.get("status") == "PASS")
    fail_count = sum(1 for i in items if i.get("status") == "FAIL")
    warn_count = sum(1 for i in items if i.get("status") == "WARN")

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Passed", pass_count)
    c2.metric("❌ Failed", fail_count)
    c3.metric("⚠️  Warnings", warn_count)

    st.divider()
    for item in items:
        status = item.get("status", "WARN")
        icon   = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(status, "❓")
        st.markdown(f"{icon} **{item.get('check')}** — {item.get('note','')}")


def display_suggestions(result: dict):
    st.markdown("### 💡 Improvement Suggestions")
    st.markdown(result.get("suggestions", "No suggestions generated."))


def display_full_report(result: dict):
    st.markdown(result.get("final_report", "Report unavailable."))

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Markdown",
            data=result.get("final_report", ""),
            file_name="document_analysis.md",
            mime="text/markdown",
        )
    with col2:
        docx_bytes = export_report_docx(result)
        st.download_button(
            "📝 Download DOCX",
            data=docx_bytes,
            file_name="document_analysis.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def display_qa_chat(doc_text: str):
    """Render the interactive Q&A chatbot panel."""
    st.markdown("### 💬 Ask Questions About This Document")

    # Initialise QA chain once per document
    if st.session_state["qa_chain"] is None:
        st.session_state["qa_chain"] = QAChain(
            doc_text=doc_text,
            model=st.session_state["model"],
            api_key=settings.ANTHROPIC_API_KEY,
        )

    # Render conversation history
    for msg in st.session_state["qa_history"]:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

    # New user input
    if prompt := st.chat_input("Ask anything about this document…"):
        st.session_state["qa_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = st.session_state["qa_chain"].ask(
                    question=prompt,
                    history=st.session_state["qa_history"][:-1],
                )
            st.markdown(answer)
        st.session_state["qa_history"].append({"role": "assistant", "content": answer})


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    _init_session()
    uploaded = render_sidebar()

    # ── Hero header ──
    st.markdown('<p class="main-title">📄 AI Document Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Upload any document — get an instant AI-powered analysis: '
        "summary, risks, clauses, compliance, and improvement suggestions.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── No file yet ──
    if not uploaded:
        st.info("👈 Upload a PDF, TXT, or DOCX document from the sidebar to get started.")

        with st.expander("ℹ️ How it works"):
            col1, col2, col3 = st.columns(3)
            col1.markdown("**1️⃣ Upload**\n\nDrop any document into the sidebar uploader.")
            col2.markdown("**2️⃣ Analyse**\n\nThe AI pipeline runs 6 specialised agents.")
            col3.markdown("**3️⃣ Review & Export**\n\nExplore results and download a full report.")
        return

    # ── File hash for caching ──
    raw_bytes = uploaded.getvalue()
    doc_hash  = hashlib.sha256(raw_bytes).hexdigest()

    # Reset Q&A chat when document changes
    if st.session_state["current_hash"] != doc_hash:
        st.session_state["current_hash"] = doc_hash
        st.session_state["qa_history"]   = []
        st.session_state["qa_chain"]     = None

    # ── Load document text ──
    if st.session_state.get("doc_hash_loaded") != doc_hash:
        loader = DocumentLoader()
        with st.spinner("📖 Reading document…"):
            doc_text = loader.load(uploaded)
        st.session_state["doc_text"]       = doc_text
        st.session_state["doc_name"]       = uploaded.name
        st.session_state["doc_hash_loaded"] = doc_hash
    else:
        doc_text = st.session_state["doc_text"]

    # ── Preview expander ──
    with st.expander("📄 Document Preview (first 60 lines)"):
        preview = "\n".join(doc_text.splitlines()[:60])
        st.markdown(f"```\n{preview}\n```")

    # ── Analyse button ──
    if st.button("🔍 Analyse Document", type="primary", use_container_width=True):
        if not settings.ANTHROPIC_API_KEY:
            st.error("❌ ANTHROPIC_API_KEY not set. Add it to your `.env` file.")
            return
        result = run_analysis(doc_text, uploaded.name)
        st.session_state["results_cache"][doc_hash] = result

    # ── Display results ──
    if doc_hash in st.session_state["results_cache"]:
        result = st.session_state["results_cache"][doc_hash]

        tabs = st.tabs([
            "📝 Summary",
            "⚠️ Risks",
            "📑 Clauses",
            "✅ Compliance",
            "💡 Suggestions",
            "📊 Full Report",
            *(["💬 Q&A Chat"] if st.session_state.get("enable_qa") else []),
        ])

        with tabs[0]: display_summary(result)
        with tabs[1]: display_risks(result)
        with tabs[2]: display_clauses(result)
        with tabs[3]: display_compliance(result)
        with tabs[4]: display_suggestions(result)
        with tabs[5]: display_full_report(result)
        if st.session_state.get("enable_qa"):
            with tabs[6]: display_qa_chat(doc_text)


if __name__ == "__main__":
    main()
