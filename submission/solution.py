"""Submission entry point.

    build() -> KnowledgeSystem   with ingest(snapshot), answer(question, as_of),
                                 conflicts()

Pure standard library. No network access and no API keys: every answer is
derived from the snapshots at ingest time, which is also what keeps answers
reproducible and under the latency budget.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_now import KnowledgeNow  # noqa: E402


def build() -> KnowledgeNow:
    return KnowledgeNow()
