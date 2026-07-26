"""Components shared by the architecture implementations."""

from .handoff import (
    FileHandoffTransport,
    RecoveryCommand,
    build_escalation_request,
    validate_escalation_request,
    validate_recovery_response,
)

__all__ = [
    "FileHandoffTransport",
    "RecoveryCommand",
    "build_escalation_request",
    "validate_escalation_request",
    "validate_recovery_response",
]
