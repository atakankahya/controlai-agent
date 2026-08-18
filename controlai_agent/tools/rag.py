"""Local document retrieval tool for ControlAI Agent."""

from __future__ import annotations

from typing import Any

from controlai_agent.registry import registry
from controlai_rag.index import ControlRAGIndex

rag_index = ControlRAGIndex()


@registry.register(
    name="search_control_references",
    description="Search local engineering textbooks, MathWorks manuals, MIT/Stanford notes, and user-provided documentation for control theory theorems, formulas, or syntax.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or concept phrase (e.g., 'Discrete Algebraic Riccati Equation CARE vs DARE', 'LQR robustness gain margin')",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
                "description": "Number of reference passages to retrieve",
            },
            "source_filter": {
                "type": "string",
                "description": "Optional substring filter on source file path",
            },
        },
        "required": ["query"],
    },
)
def search_control_references(
    query: str,
    top_k: int = 3,
    source_filter: str | None = None,
) -> dict[str, Any]:
    hits = rag_index.search(query=query, top_k=top_k, source_filter=source_filter)
    if not hits:
        return {
            "query": query,
            "results_found": 0,
            "message": "No matching reference passages found in local index.",
        }

    formatted_passages = []
    for hit in hits:
        formatted_passages.append({
            "citation": f"[{hit['filename']}:{hit.get('page') or 'sec'}]",
            "source_path": hit["source"],
            "relevance_score": hit["score"],
            "content": hit["text"][:600],
        })

    return {
        "query": query,
        "results_found": len(formatted_passages),
        "passages": formatted_passages,
    }
