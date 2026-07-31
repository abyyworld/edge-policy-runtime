"""edge-policy-runtime — edge policy runtime with signed OTA updates and telemetry."""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import AgentConfig, DeviceAgent
from .ota import OTAClient
from .policy import Policy
from .runtime import InferenceRuntime
from .telemetry import TelemetryClient

__all__ = [
    "AgentConfig",
    "DeviceAgent",
    "InferenceRuntime",
    "OTAClient",
    "Policy",
    "TelemetryClient",
    "__version__",
]
