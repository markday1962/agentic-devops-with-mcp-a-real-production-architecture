"""``approvals`` — inspect and resolve pending approvals from a terminal.

The out-of-band path for when Slack is down, and the one ConsoleNotifier points
reviewers at during local development.
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import ApprovalConflict, ApprovalNotFound
from .store import SQLiteApprovalStore


def _print_request(request, *, verbose: bool = False) -> None:
    line = (
        f"{request.id}  {request.status.value:<9} {request.risk.value:<6} "
        f"{request.tool_name:<22} {request.requested_at.isoformat()}"
    )
    if request.decided_by:
        line += f"  by {request.decided_by}"
    print(line)
    if verbose:
        print(f"    args:    {json.dumps(request.args)}")
        print(f"    expires: {request.expires_at.isoformat()}")
        if request.thread_id:
            print(f"    thread:  {request.thread_id}")
        if request.decision_note:
            print(f"    note:    {request.decision_note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="approvals", description=__doc__)
    parser.add_argument("--db", default="approvals.db", help="path to the approval store")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pending", help="list requests awaiting a decision")

    history = sub.add_parser("history", help="show the audit trail")
    history.add_argument("--thread", help="limit to one agent thread")
    history.add_argument("--limit", type=int, default=50)

    show = sub.add_parser("show", help="show one request in full")
    show.add_argument("id")

    resolve = sub.add_parser("resolve", help="approve or reject a request")
    resolve.add_argument("id")
    decision = resolve.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    resolve.add_argument("--by", required=True, help="who is making this call")
    resolve.add_argument("--note")

    args = parser.parse_args(argv)
    store = SQLiteApprovalStore(args.db)

    try:
        if args.command == "pending":
            requests = store.list_pending()
            if not requests:
                print("nothing awaiting approval")
            for request in requests:
                _print_request(request, verbose=True)
            return 0

        if args.command == "history":
            for request in store.history(thread_id=args.thread, limit=args.limit):
                _print_request(request)
            return 0

        if args.command == "show":
            try:
                _print_request(store.get(args.id), verbose=True)
            except ApprovalNotFound as exc:
                print(exc, file=sys.stderr)
                return 1
            return 0

        if args.command == "resolve":
            try:
                request = store.resolve(
                    args.id,
                    approved=args.approve,
                    decided_by=args.by,
                    note=args.note,
                )
            except ApprovalNotFound as exc:
                print(exc, file=sys.stderr)
                return 1
            except ApprovalConflict as exc:
                print(exc, file=sys.stderr)
                return 1
            _print_request(request, verbose=True)
            return 0
    finally:
        store.close()

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
