"""
src/agents/qa_chain.py
======================
Document-grounded Q&A chatbot.

Maintains the full document as context and answers user questions
by referencing specific parts of the document.
Supports multi-turn conversation history.
"""

import logging
from textwrap import dedent
from typing import List, Optional

from src.utils.llm_client import LLMClient
from src.config.settings import Settings

logger   = logging.getLogger(__name__)
settings = Settings()

# Context window budget for the document (characters)
# Keeps the system prompt + doc + history within Claude's context.
_MAX_DOC_CHARS = 40_000


class QAChain:
    """
    Stateless document Q&A chain.

    The full document text is injected into the system prompt so Claude
    always has the source material available. Conversation history is
    passed on each turn for multi-turn coherence.

    Args:
        doc_text:  Full extracted text of the uploaded document.
        model:     Claude model identifier string.
        api_key:   Anthropic API key.
    """

    def __init__(self, doc_text: str, model: str, api_key: Optional[str] = None):
        # Truncate very large documents to stay within context budget
        self.doc_text = doc_text[:_MAX_DOC_CHARS]
        if len(doc_text) > _MAX_DOC_CHARS:
            logger.warning(
                "Document truncated from %d to %d chars for Q&A context.",
                len(doc_text), _MAX_DOC_CHARS
            )

        self._llm = LLMClient(
            api_key=api_key or settings.ANTHROPIC_API_KEY,
            model=model,
            max_tokens=1024,
        )
        self._system = dedent(f"""
        You are an expert document analyst and legal assistant.
        You have been given a document to analyse. Answer the user's questions
        accurately and concisely based on the document content below.

        Rules:
        - Only answer based on what is in the document.
        - If information is not in the document, say so clearly.
        - Quote relevant passages when helpful.
        - Do not provide legal advice; note when professional consultation is recommended.
        - Be concise but thorough.

        ━━━━━━━━━━━━━━━━━━━━━ DOCUMENT ━━━━━━━━━━━━━━━━━━━━━
        {self.doc_text}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """).strip()

    def ask(self, question: str, history: Optional[List[dict]] = None) -> str:
        """
        Answer a user question about the document.

        Args:
            question: The user's question string.
            history:  Previous conversation turns as list of
                      {"role": "user"|"assistant", "content": "…"} dicts.

        Returns:
            The assistant's answer as a plain string.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": question})

        try:
            answer = self._llm.call_with_history(
                messages=messages,
                system=self._system,
                temperature=settings.QA_TEMPERATURE,
            )
            logger.debug("Q&A answer (%d chars)", len(answer))
            return answer
        except Exception as e:
            logger.error("Q&A chain error: %s", e)
            return f"⚠️ An error occurred: {e}"
