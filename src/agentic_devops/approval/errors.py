"""Exceptions raised by the approval gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import ApprovalRequest


class ApprovalError(Exception):
    """Base class for approval-gate failures."""


class ApprovalNotFound(ApprovalError):
    """No approval request exists with the given id."""


class ApprovalConflict(ApprovalError):
    """A state transition was attempted that the request's status does not allow.

    Raised when a reviewer clicks Approve on a request that already timed out,
    when two reviewers race, or when an approval is spent twice. Carries the
    request as it actually stands so the caller can report the real outcome.
    """

    def __init__(self, message: str, request: "ApprovalRequest") -> None:
        super().__init__(message)
        self.request = request


class ArgumentMismatch(ApprovalError):
    """Execution arguments differ from the ones a human approved."""

    def __init__(self, message: str, request: "ApprovalRequest") -> None:
        super().__init__(message)
        self.request = request


class ApprovalDenied(ApprovalError):
    """A guarded action was not approved (rejected, timed out, or unnotifiable)."""

    def __init__(self, message: str, request: "ApprovalRequest") -> None:
        super().__init__(message)
        self.request = request


class NotificationFailed(ApprovalError):
    """The request could not be put in front of a human."""
