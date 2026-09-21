"""Which tool calls need a human, and how alarming they are.

The article's ``_assess_risk`` took the call arguments and then ignored them, so
``delete_resource(namespace='scratch')`` and ``patch_resource(namespace='kube-system',
replicas=0)`` scored the same. Here the tool name sets a floor and escalation
rules raise it based on what the call actually touches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .models import RiskLevel

#: Tools that change infrastructure. Everything not listed is read-only and runs
#: unguarded — an agent that needs approval to read logs is an agent nobody uses.
DEFAULT_WRITE_TOOLS: frozenset[str] = frozenset({
    "apply_manifest",
    "patch_resource",
    "delete_resource",
    "scale_deployment",
    "restart_deployment",
    "create_pull_request",
    "merge_pull_request",
    "create_issue",
    "terminate_instance",
    "modify_security_group",
    "update_dns_record",
})

DEFAULT_HIGH_RISK: frozenset[str] = frozenset({
    "delete_resource",
    "terminate_instance",
    "merge_pull_request",
    "modify_security_group",
    "update_dns_record",
})

DEFAULT_MEDIUM_RISK: frozenset[str] = frozenset({
    "apply_manifest",
    "patch_resource",
    "scale_deployment",
    "restart_deployment",
    "create_pull_request",
})

#: Namespaces where a mistake is an outage rather than an inconvenience.
PROTECTED_NAMESPACES: frozenset[str] = frozenset({
    "kube-system",
    "kube-public",
    "production",
    "prod",
    "platform-tools",
})

PROTECTED_ENVIRONMENTS: frozenset[str] = frozenset({"production", "prod"})

#: An escalation rule reports the floor it wants for a call, or None to abstain.
EscalationRule = Callable[[str, Mapping[str, Any]], RiskLevel | None]


def escalate_protected_namespace(tool_name: str, args: Mapping[str, Any]) -> RiskLevel | None:
    namespace = args.get("namespace") or args.get("ns")
    if isinstance(namespace, str) and namespace.lower() in PROTECTED_NAMESPACES:
        return RiskLevel.HIGH
    return None


def escalate_production_environment(tool_name: str, args: Mapping[str, Any]) -> RiskLevel | None:
    env = args.get("environment") or args.get("env") or args.get("cluster")
    if isinstance(env, str) and env.lower() in PROTECTED_ENVIRONMENTS:
        return RiskLevel.HIGH
    return None


def escalate_scale_to_zero(tool_name: str, args: Mapping[str, Any]) -> RiskLevel | None:
    replicas = args.get("replicas")
    if isinstance(replicas, int) and not isinstance(replicas, bool) and replicas == 0:
        return RiskLevel.HIGH
    return None


def escalate_bulk_operation(tool_name: str, args: Mapping[str, Any]) -> RiskLevel | None:
    """``--all`` and bare label selectors turn one mistake into many."""
    if args.get("all") is True or args.get("all_namespaces") is True:
        return RiskLevel.HIGH
    for key in ("label_selector", "selector"):
        if key not in args:
            continue
        selector = args[key]
        # An empty selector matches everything, and is falsy — so test for the
        # key's presence rather than its truthiness.
        if isinstance(selector, str) and selector.strip() in {"", "*"}:
            return RiskLevel.HIGH
    return None


DEFAULT_ESCALATIONS: tuple[EscalationRule, ...] = (
    escalate_protected_namespace,
    escalate_production_environment,
    escalate_scale_to_zero,
    escalate_bulk_operation,
)

_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    write_tools: frozenset[str] = DEFAULT_WRITE_TOOLS
    high_risk: frozenset[str] = DEFAULT_HIGH_RISK
    medium_risk: frozenset[str] = DEFAULT_MEDIUM_RISK
    escalations: tuple[EscalationRule, ...] = DEFAULT_ESCALATIONS
    #: Tools nobody may run through the agent, no matter who approves.
    forbidden_tools: frozenset[str] = field(default_factory=frozenset)
    #: Tools known to be safe to run unattended, regardless of what any server
    #: claims about them.
    read_only_tools: frozenset[str] = field(default_factory=frozenset)
    #: What to do with a tool nobody has classified: gate it. A DevOps agent
    #: picks up tools from MCP servers it did not ship with, and the cost of
    #: waving an unrecognised one through is unbounded.
    unknown_tools_require_approval: bool = True

    def requires_approval(self, tool_name: str, *, read_only_hint: bool | None = None) -> bool:
        """Decide whether a call must stop for a human.

        ``read_only_hint`` comes from the MCP tool's own annotations. It can
        clear a tool the operator has not classified, but it can never clear
        one on ``write_tools`` — a server declaring its delete tool read-only
        does not make it so.
        """
        if tool_name in self.write_tools:
            return True
        if tool_name in self.read_only_tools:
            return False
        if read_only_hint is True:
            return False
        return self.unknown_tools_require_approval

    def is_forbidden(self, tool_name: str) -> bool:
        return tool_name in self.forbidden_tools

    def assess(self, tool_name: str, args: Mapping[str, Any]) -> RiskLevel:
        if tool_name in self.high_risk:
            level = RiskLevel.HIGH
        elif tool_name in self.medium_risk:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        for rule in self.escalations:
            candidate = rule(tool_name, args)
            if candidate is not None and _ORDER[candidate] > _ORDER[level]:
                level = candidate
        return level

    def with_write_tools(self, names: Iterable[str]) -> "ApprovalPolicy":
        from dataclasses import replace as _replace

        return _replace(self, write_tools=frozenset(names))
