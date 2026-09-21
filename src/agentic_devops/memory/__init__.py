"""Layer 4: memory.

Episodic — what happened, turn by turn, resumable after a crash.
Semantic — what the organisation knows, retrievable by symptom.
"""

from .fts import FTSKnowledgeIndex
from .knowledge import (
    Document,
    HybridIndex,
    Kind,
    KnowledgeIndex,
    SearchHit,
    query_terms,
    render_hits,
)
from .learn import add_postmortem, add_runbook, record_run, run_to_document
from .recall import KnowledgeRecall
from .transcripts import (
    Checkpoint,
    TranscriptStore,
    prepare_resume,
    serialize_messages,
)

__all__ = [
    "Checkpoint",
    "Document",
    "FTSKnowledgeIndex",
    "HybridIndex",
    "Kind",
    "KnowledgeIndex",
    "KnowledgeRecall",
    "SearchHit",
    "TranscriptStore",
    "add_postmortem",
    "add_runbook",
    "prepare_resume",
    "query_terms",
    "record_run",
    "render_hits",
    "run_to_document",
    "serialize_messages",
]
