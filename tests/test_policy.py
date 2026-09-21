from __future__ import annotations

from agentic_devops.approval import ApprovalPolicy, RiskLevel

policy = ApprovalPolicy()


def test_unclassified_tools_are_gated_by_default():
    """Fail closed: a tool nobody has classified is not assumed harmless."""
    assert policy.requires_approval("get_pod_logs")


def test_mcp_read_only_hint_clears_an_unclassified_tool():
    assert not policy.requires_approval("get_pod_logs", read_only_hint=True)
    assert policy.requires_approval("get_pod_logs", read_only_hint=False)


def test_operator_allowlist_clears_a_tool():
    permissive = ApprovalPolicy(read_only_tools=frozenset({"get_pod_logs"}))
    assert not permissive.requires_approval("get_pod_logs")


def test_a_server_cannot_declare_its_delete_tool_read_only():
    """Hints may clear an unknown tool; they may not clear a known write."""
    assert policy.requires_approval("delete_resource", read_only_hint=True)


def test_unknown_tools_can_be_allowed_explicitly():
    lax = ApprovalPolicy(unknown_tools_require_approval=False)
    assert not lax.requires_approval("get_pod_logs")


def test_write_tools_need_approval():
    assert policy.requires_approval("delete_resource")
    assert policy.requires_approval("merge_pull_request")


def test_tool_name_sets_the_risk_floor():
    assert policy.assess("delete_resource", {"namespace": "scratch"}) is RiskLevel.HIGH
    assert policy.assess("apply_manifest", {"namespace": "scratch"}) is RiskLevel.MEDIUM
    assert policy.assess("create_issue", {"title": "x"}) is RiskLevel.LOW


def test_protected_namespace_escalates_to_high():
    assert policy.assess("patch_resource", {"namespace": "kube-system"}) is RiskLevel.HIGH
    assert policy.assess("apply_manifest", {"namespace": "Production"}) is RiskLevel.HIGH


def test_scaling_to_zero_escalates():
    low_stakes = {"namespace": "scratch", "replicas": 3}
    outage = {"namespace": "scratch", "replicas": 0}
    assert policy.assess("scale_deployment", low_stakes) is RiskLevel.MEDIUM
    assert policy.assess("scale_deployment", outage) is RiskLevel.HIGH


def test_bulk_operations_escalate():
    assert policy.assess("delete_resource", {"all": True}) is RiskLevel.HIGH
    assert policy.assess("patch_resource", {"label_selector": ""}) is RiskLevel.HIGH


def test_replicas_true_is_not_treated_as_zero():
    # bool is an int subclass; `replicas=False` must not read as scale-to-zero
    assert policy.assess("scale_deployment", {"replicas": False}) is RiskLevel.MEDIUM


def test_escalation_never_lowers_risk():
    assert policy.assess("terminate_instance", {"namespace": "scratch"}) is RiskLevel.HIGH


def test_forbidden_tools_are_reported():
    strict = ApprovalPolicy(forbidden_tools=frozenset({"delete_resource"}))
    assert strict.is_forbidden("delete_resource")
    assert not strict.is_forbidden("apply_manifest")
