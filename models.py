"""
ARC/models.py
Pydantic models for ARC-India OpenEnv environment.
Action, Observation, State — inheriting from openenv.core base classes.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


# ─────────────────────────────────────────────────────────────────────────────
# ACTION
# action_type : "signal" | "reroute" | "emergency" | "comply" | "observe"
# target_zone : "Z0" … "Z8"  (None = global / no-op)
# parameters  : payload varies by action_type
#   signal    → {"phase": "NS_GREEN"}
#   reroute   → {"vehicle_ids": ["V0","V3"]}
#   emergency → {"vehicle_id": "AMB-01", "cleared": True}
#   comply    → {"vehicle_ids": ["V7"]}
# ─────────────────────────────────────────────────────────────────────────────

class ARCAction(Action):
    agent_id:    str            = Field(default="global",  description="Which agent is acting")
    action_type: str            = Field(default="observe", description="signal|reroute|emergency|comply|observe")
    target_zone: Optional[str]  = Field(default=None,      description="Zone ID e.g. Z4")
    parameters:  Dict[str, Any] = Field(default_factory=dict, description="Action payload")


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION
# Inherits: done (bool), reward (float|None), metadata (dict)
# ─────────────────────────────────────────────────────────────────────────────

class ARCObservation(Observation):
    # ASCII grid  [row][col] → "." | "C" | "B" | "U" | "W"
    grid_snapshot:      List[List[str]]          = Field(default_factory=list)

    # {"Z0": {"phase": "NS_GREEN", "queue_len": 3, "avg_wait": 2.1, "violations": 0}}
    zone_metrics:       Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # [{"type": "accident", "zone": "Z3", "severity": 0.8, "ttl": 7}]
    active_events:      List[Dict[str, Any]]      = Field(default_factory=list)

    # {"Z0": 2, "Z4": 5}
    pedestrian_density: Dict[str, int]            = Field(default_factory=dict)

    # {"vehicle_id": "AMB-01", "zone": "Z4", "severity": 0.9, "eta": 3} | None
    emergency_status:   Optional[Dict[str, Any]]  = Field(default=None)

    # Itemised reward breakdown for transparency
    reward_breakdown:   Dict[str, float]          = Field(default_factory=dict)

    message:            str                       = Field(default="")


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# Inherits: episode_id (str|None), step_count (int)
# ─────────────────────────────────────────────────────────────────────────────

class ARCState(State):
    seed:       int  = Field(default=42)
    difficulty: str  = Field(default="medium")
    grid_width:  int = Field(default=10)
    grid_height: int = Field(default=10)

    vehicles:    Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pedestrians: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    obstacles:   Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # {"Z0": "NS_GREEN", ...}
    signal_phases: Dict[str, str] = Field(default_factory=dict)

    active_events: List[Dict[str, Any]] = Field(default_factory=list)

    total_throughput:           int   = Field(default=0)
    total_collisions:           int   = Field(default=0)
    total_violations:           int   = Field(default=0)
    total_pedestrian_incidents: int   = Field(default=0)
    emergency_successes:        int   = Field(default=0)
    emergency_failures:         int   = Field(default=0)
    cumulative_reward:          float = Field(default=0.0)