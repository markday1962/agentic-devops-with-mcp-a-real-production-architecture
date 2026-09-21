"""What is safe to put in a span.

The article's tracer records ``agent.task_goal[:500]`` and
``json.dumps(args)[:500]``. Both are incident data. Tool arguments carry
hostnames, customer identifiers, connection strings and — when someone passes
a credential to a tool that takes one — secrets; incident titles carry customer
names. Traces leave the cluster for a backend with its own access control,
retention and vendor, so the default here is to record the *shape* of a call
rather than its contents: which tool, which argument names, how many, how long,
what happened.

``record_payloads=True`` turns values back on for debugging, and everything
that goes out then passes through :class:`Redactor` first. Redaction is a
backstop, not a guarantee — it catches the credential formats we know about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

REDACTED = "«redacted»"

#: Argument names whose values are never recorded, matched as substrings so
#: `aws_secret_access_key` and `dbPassword` are both covered.
SECRET_KEY_HINTS: frozenset[str] = frozenset({
    "token",
    "password",
    "passwd",
    "secret",
    "credential",
    "authorization",
    "auth",
    "apikey",
    "api_key",
    "private_key",
    "privatekey",
    "session",
    "cookie",
    "signature",
    "passphrase",
})

#: Value shapes that are credentials wherever they appear.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                      # AWS access key id
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),                      # AWS temporary key id
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),                # OpenAI/Anthropic-style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),            # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),         # Slack tokens
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:[^\s/@]+@"),  # creds in a URL
)


@dataclass(frozen=True, slots=True)
class Redactor:
    key_hints: frozenset[str] = SECRET_KEY_HINTS
    patterns: tuple[re.Pattern[str], ...] = SECRET_PATTERNS
    #: Span attributes are not a log sink; long values are truncated.
    max_length: int = 256

    def is_secret_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(hint in lowered for hint in self.key_hints)

    def scrub(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub(REDACTED, text)
        return text

    def value(self, key: str, value: Any) -> Any:
        if self.is_secret_key(key):
            return REDACTED
        if isinstance(value, str):
            cleaned = self.scrub(value)
            if len(cleaned) > self.max_length:
                return cleaned[: self.max_length] + f"…(+{len(cleaned) - self.max_length})"
            return cleaned
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return self.mapping(value)
        if isinstance(value, (list, tuple)):
            return [self.value(key, item) for item in value]
        return self.value(key, str(value))

    def mapping(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return {key: self.value(key, item) for key, item in values.items()}

    def text(self, text: str) -> str:
        cleaned = self.scrub(text)
        if len(cleaned) > self.max_length:
            return cleaned[: self.max_length] + f"…(+{len(cleaned) - self.max_length})"
        return cleaned


def argument_shape(args: Mapping[str, Any]) -> dict[str, Any]:
    """What a call looked like, without saying what it was about.

    Enough to debug a malformed tool call — which arguments were present, how
    big they were — with nothing in it that could identify a customer or leak
    a credential. This is what gets recorded by default.
    """
    return {
        "arg_count": len(args),
        "arg_names": ",".join(sorted(args)) if args else "",
        "arg_types": ",".join(
            f"{key}:{type(value).__name__}" for key, value in sorted(args.items())
        ),
    }


def flatten(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Span attributes must be flat scalars; nested dicts become dotted keys."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            out.update(flatten(name, value))
        elif isinstance(value, (list, tuple)):
            out[name] = ",".join(str(item) for item in value)
        else:
            out[name] = value
    return out


def allowed_attributes(values: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Drop Nones — the OTel SDK warns on them rather than ignoring them."""
    return {key: value for key, value in values if value is not None}
