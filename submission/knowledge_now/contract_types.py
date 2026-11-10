"""The types that cross the harness boundary.

The harness adds its own `harness/` directory to `sys.path` before importing a
submission, so `contract` is normally importable. We still define a structurally
identical fallback, because the harness accepts any object with these fields and
a submission that cannot be imported scores nothing.
"""
from __future__ import annotations

try:  # pragma: no cover - depends on how the harness invokes us
    from contract import Answer, Conflict, Page, Snapshot, Source  # type: ignore
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass(frozen=True)
    class Page:  # type: ignore[no-redef]
        url: str
        title: str
        markdown: str
        fetched_at: str

    @dataclass(frozen=True)
    class Snapshot:  # type: ignore[no-redef]
        snapshot_id: str
        taken_on: str
        pages: list

    @dataclass
    class Source:  # type: ignore[no-redef]
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:  # type: ignore[no-redef]
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:  # type: ignore[no-redef]
        topic: str
        values: list = field(default_factory=list)
        sources: list = field(default_factory=list)
        resolution: Optional[str] = None
        reason: Optional[str] = None

__all__ = ["Answer", "Conflict", "Page", "Snapshot", "Source"]
