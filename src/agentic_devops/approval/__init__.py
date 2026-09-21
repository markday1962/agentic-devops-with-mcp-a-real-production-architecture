"""Layer 2: the approval gate.

Every write an agent proposes stops here until a named human says yes, inside a
deadline, for those exact arguments, once.
"""

from .errors import (
    ApprovalConflict,
    ApprovalDenied,
    ApprovalError,
    ApprovalNotFound,
    ArgumentMismatch,
    NotificationFailed,
)
from .gate import DEFAULT_TIMEOUT, ApprovalGate, Authorization, denial_message
from .models import (
    ApprovalRequest,
    ApprovalStatus,
    RiskLevel,
    fingerprint_args,
    utcnow,
)
from .notifier import ConsoleNotifier, Notifier, NullNotifier, SlackNotifier
from .policy import ApprovalPolicy
from .store import ApprovalStore, SQLiteApprovalStore

__all__ = [
    "ApprovalConflict",
    "ApprovalDenied",
    "ApprovalError",
    "ApprovalGate",
    "ApprovalNotFound",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalStore",
    "ArgumentMismatch",
    "Authorization",
    "ConsoleNotifier",
    "DEFAULT_TIMEOUT",
    "Notifier",
    "NotificationFailed",
    "NullNotifier",
    "RiskLevel",
    "SQLiteApprovalStore",
    "SlackNotifier",
    "denial_message",
    "fingerprint_args",
    "utcnow",
]
