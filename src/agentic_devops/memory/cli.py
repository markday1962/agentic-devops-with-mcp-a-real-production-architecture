"""``knowledge`` — curate what the agent remembers.

The index is only as good as what is in it, and the one thing that must stay
human-controlled is which documents are trusted. ``knowledge verify`` is that
control: agent-written documents arrive unverified and stay that way until
somebody reads one and says otherwise.
"""

from __future__ import annotations

import argparse
import sys

from .fts import FTSKnowledgeIndex
from .knowledge import Document, Kind
from .learn import add_postmortem, add_runbook
from .transcripts import TranscriptStore


def _print(document: Document, *, verbose: bool = False) -> None:
    mark = "✓" if document.verified else "?"
    print(
        f"{document.id}  {mark} {document.kind.value:<11} "
        f"{(document.service or '-'):<18} {document.title}"
    )
    if verbose:
        print(f"    recorded: {document.created_at.isoformat()}")
        if document.thread_id:
            print(f"    thread:   {document.thread_id}")
        for line in document.body.splitlines():
            print(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knowledge", description=__doc__)
    parser.add_argument("--db", default="knowledge.db", help="path to the knowledge index")
    parser.add_argument(
        "--transcripts", default="transcripts.db", help="path to the transcript store"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="query the index the way the agent does")
    search.add_argument("query")
    search.add_argument("--kind", choices=[k.value for k in Kind], action="append")
    search.add_argument("--service")
    search.add_argument("--limit", type=int, default=5)

    listing = sub.add_parser("list", help="list documents, newest first")
    listing.add_argument("--limit", type=int, default=50)

    show = sub.add_parser("show", help="print one document in full")
    show.add_argument("id")

    postmortem = sub.add_parser("add-postmortem", help="file a human-written postmortem")
    postmortem.add_argument("--service", required=True)
    postmortem.add_argument("--title", required=True)
    postmortem.add_argument("--root-cause", required=True)
    postmortem.add_argument("--resolution", required=True)
    postmortem.add_argument("--incident-id")

    runbook = sub.add_parser("add-runbook", help="file a runbook")
    runbook.add_argument("--service", required=True)
    runbook.add_argument("--title", required=True)
    runbook.add_argument("--step", action="append", required=True, dest="steps")

    verify = sub.add_parser("verify", help="mark an agent-written document as trusted")
    verify.add_argument("id")
    verify.add_argument("--undo", action="store_true", help="mark it untrusted again")

    remove = sub.add_parser("rm", help="delete a document")
    remove.add_argument("id")

    runs = sub.add_parser("transcripts", help="list stored run transcripts")
    runs.add_argument("--resumable", action="store_true", help="only interrupted runs")
    runs.add_argument("--service")
    runs.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)

    if args.command == "transcripts":
        store = TranscriptStore(args.transcripts)
        try:
            entries = (
                store.resumable(limit=args.limit)
                if args.resumable
                else store.recent(service=args.service, limit=args.limit)
            )
            if not entries:
                print("no transcripts")
            for entry in entries:
                print(
                    f"{entry.thread_id:<28} {entry.status:<10} turns={entry.turns:<3} "
                    f"{(entry.service or '-'):<18} {entry.updated_at.isoformat()}"
                )
        finally:
            store.close()
        return 0

    index = FTSKnowledgeIndex(args.db)
    try:
        if args.command == "search":
            kinds = [Kind(value) for value in args.kind] if args.kind else None
            hits = index.search(
                args.query, limit=args.limit, kinds=kinds, service=args.service
            )
            if not hits:
                print("no matches")
            for hit in hits:
                print(f"[{hit.score:7.3f}] ", end="")
                _print(hit.document)
            return 0

        if args.command == "list":
            for document in index.all(limit=args.limit):
                _print(document)
            return 0

        if args.command == "show":
            document = index.get(args.id)
            if document is None:
                print(f"no document {args.id}", file=sys.stderr)
                return 1
            _print(document, verbose=True)
            return 0

        if args.command == "add-postmortem":
            _print(
                add_postmortem(
                    index,
                    service=args.service,
                    title=args.title,
                    root_cause=args.root_cause,
                    resolution=args.resolution,
                    incident_id=args.incident_id,
                ),
                verbose=True,
            )
            return 0

        if args.command == "add-runbook":
            _print(
                add_runbook(
                    index, service=args.service, title=args.title, steps=args.steps
                ),
                verbose=True,
            )
            return 0

        if args.command == "verify":
            if not index.set_verified(args.id, not args.undo):
                print(f"no document {args.id}", file=sys.stderr)
                return 1
            _print(index.get(args.id))
            return 0

        if args.command == "rm":
            if not index.delete(args.id):
                print(f"no document {args.id}", file=sys.stderr)
                return 1
            print(f"deleted {args.id}")
            return 0
    finally:
        index.close()

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
