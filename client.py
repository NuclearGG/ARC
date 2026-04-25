"""
ARC/client.py
OpenEnv SyncEnvClient for ARC-India.
Used by RL training scripts and tests to interact with the running server.

Usage:
  from ARC.client import ARCClient
  from ARC.models import ARCAction

  with ARCClient(base_url="http://localhost:8000") as client:
      obs  = client.reset()
      obs  = client.step(ARCAction(action_type="signal",
                                    target_zone="Z4",
                                    parameters={"phase": "NS_GREEN"}))
      state = client.get_state()
"""

from openenv.core import SyncEnvClient
from .models import ARCAction, ARCObservation


class ARCClient(SyncEnvClient):
    """
    Synchronous HTTP client for the ARC-India OpenEnv server.
    Wraps SyncEnvClient with ARC-specific action/observation types.
    """

    action_cls      = ARCAction
    observation_cls = ARCObservation
