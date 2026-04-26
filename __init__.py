"""
ARC-India OpenEnv Package.

simulation and agents work standalone (gymnasium + numpy only).
models and client require openenv — loaded lazily.
"""

from .simulation import ARCIndiaSimulation
from .agents import AgentOrchestrator

__all__ = ["ARCIndiaSimulation", "AgentOrchestrator"]

try:
    from .models import ARCAction, ARCObservation, ARCState
    from .client import ARCClient
    __all__ += ["ARCAction", "ARCObservation", "ARCState", "ARCClient"]
except ImportError:
    pass