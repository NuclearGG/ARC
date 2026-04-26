"""
ARC-India: agents.py
Five-agent architecture for the ARC-India environment.

Agents:
  1. PerceptionAgent        — reads raw observation, builds world model
  2. LocalControlAgent      — manages one intersection/zone
  3. GlobalCoordinatorAgent — system-wide routing & resource allocation
  4. BACSAgent              — Behavior-Aware Compliance System
  5. EmergencyDecisionAgent — detects & routes emergency vehicles

Each agent exposes a .decide(observation) -> TrafficAction method.
During RL training, the LLM replaces these heuristic policies.

FIXES APPLIED:
  BUG-10  GlobalCoordinatorAgent hardcoded zone list as a string literal instead
          of importing ZONE_IDS from simulation. If the zone layout ever changes
          the agent silently operates on a stale list — and LocalControlAgent
          instances for missing zones would never be created.
          Fixed: import ZONE_IDS from ARCE.simulation.
"""

import os
import sys
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# BUG-10 FIX: import the single source-of-truth zone list from simulation
# Guard against the case where this file is imported before sys.path is set
# (e.g. during server startup) by inserting the project root here too.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ARCE.simulation import ZONE_IDS   # single source of truth


# ---------------------------------------------------------------------------
# Shared world model (built by PerceptionAgent, read by others)
# ---------------------------------------------------------------------------

@dataclass
class WorldModel:
    congested_zones: List[str]
    violated_zones:  List[str]
    ped_heavy_zones: List[str]
    emergency_event: Optional[Dict[str, Any]]
    step: int = 0


# ---------------------------------------------------------------------------
# 1. Perception Agent
# ---------------------------------------------------------------------------

class PerceptionAgent:
    """
    Transforms raw observation into a structured WorldModel.
    In RL, this becomes the LLM's context window / prompt.
    """
    name = "perception"

    def perceive(self, obs: Dict) -> WorldModel:
        zone_metrics = obs.get("zone_metrics", {})
        ped_density  = obs.get("pedestrian_density", {})

        congested = [z for z, m in zone_metrics.items() if m.get("queue_len", 0) > 5]
        violated  = [z for z, m in zone_metrics.items() if m.get("violations", 0) > 0]
        ped_heavy = [z for z, cnt in ped_density.items() if cnt >= 3]

        emergency = obs.get("emergency_status")

        return WorldModel(
            congested_zones=congested,
            violated_zones=violated,
            ped_heavy_zones=ped_heavy,
            emergency_event=emergency,
        )

    def to_prompt(self, world: WorldModel, obs: Dict) -> str:
        """
        Converts WorldModel to natural-language prompt for LLM agent.
        This is what gets fed to the model during GRPO training.
        """
        lines = [
            "=== ARC-India Traffic System ===",
            f"Step: {world.step}",
            f"Congested zones: {world.congested_zones or 'None'}",
            f"Violation zones: {world.violated_zones or 'None'}",
            f"High pedestrian zones: {world.ped_heavy_zones or 'None'}",
        ]
        if world.emergency_event:
            ev = world.emergency_event
            lines.append(
                f"🚨 EMERGENCY: {ev.get('vehicle_id','?')} in zone {ev.get('zone','?')} "
                f"(severity={ev.get('severity',0):.1f}, ETA={ev.get('eta','?')} steps)"
            )
        lines.append("\nZone signal phases:")
        for z, m in obs.get("zone_metrics", {}).items():
            lines.append(
                f"  {z}: phase={m['phase']}, queue={m['queue_len']}, wait={m['avg_wait']}s"
            )
        lines.append(
            "\nAvailable actions: signal | reroute | emergency | comply | observe"
        )
        lines.append("Respond with: ACTION|zone|parameters (e.g. signal|Z3|phase=NS_GREEN)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Local Control Agent (one per zone)
# ---------------------------------------------------------------------------

class LocalControlAgent:
    """Manages signal timing for a single zone."""
    name = "local"

    def __init__(self, zone_id: str):
        self.zone_id = zone_id

    def decide(self, world: WorldModel) -> Optional[Dict]:
        """Returns an action dict, or None if nothing to do."""
        if self.zone_id in world.congested_zones:
            return {
                "agent_id":    f"local_{self.zone_id}",
                "action_type": "signal",
                "target_zone": self.zone_id,
                "parameters":  {"phase": "NS_GREEN"},
            }
        if self.zone_id in world.ped_heavy_zones:
            return {
                "agent_id":    f"local_{self.zone_id}",
                "action_type": "signal",
                "target_zone": self.zone_id,
                "parameters":  {"phase": "PEDESTRIAN_CROSS"},
            }
        return None


# ---------------------------------------------------------------------------
# 3. Global Coordinator Agent
# ---------------------------------------------------------------------------

class GlobalCoordinatorAgent:
    """Coordinates across all zones — resolves conflicts between locals."""
    name = "global"

    def __init__(self):
        # BUG-10 FIX: use imported ZONE_IDS instead of a hardcoded literal
        self._local_agents = {z: LocalControlAgent(z) for z in ZONE_IDS}

    def decide(self, world: WorldModel, obs: Dict) -> Dict:
        """
        Pick the highest-priority action across all zones.
        Priority: emergency > congestion > pedestrian > idle.
        """
        if world.emergency_event:
            return {
                "agent_id":    "global",
                "action_type": "observe",
                "target_zone": None,
                "parameters":  {},
            }

        zone_metrics = obs.get("zone_metrics", {})
        if world.congested_zones:
            worst = max(
                world.congested_zones,
                key=lambda z: zone_metrics.get(z, {}).get("queue_len", 0),
            )
            local_action = self._local_agents[worst].decide(world)
            if local_action:
                return local_action

        return {
            "agent_id":    "global",
            "action_type": "observe",
            "target_zone": None,
            "parameters":  {},
        }


# ---------------------------------------------------------------------------
# 4. BACS — Behavior-Aware Compliance System
# ---------------------------------------------------------------------------

class BACSAgent:
    """
    Detects non-compliant vehicles and attempts enforcement.
    Mirrors real-world adaptive compliance nudging.
    """
    name = "bacs"

    def decide(self, world: WorldModel, obs: Dict) -> Optional[Dict]:
        if not world.violated_zones:
            return None
        zone_metrics = obs.get("zone_metrics", {})
        worst = max(
            world.violated_zones,
            key=lambda z: zone_metrics.get(z, {}).get("violations", 0),
        )
        return {
            "agent_id":    "bacs",
            "action_type": "comply",
            "target_zone": worst,
            "parameters":  {"vehicle_ids": []},
        }


# ---------------------------------------------------------------------------
# 5. Emergency Decision Agent
# ---------------------------------------------------------------------------

class EmergencyDecisionAgent:
    """
    Detects emergency events, estimates severity, and clears corridors.
    """
    name = "emergency"

    def decide(self, world: WorldModel) -> Optional[Dict]:
        if not world.emergency_event:
            return None
        ev = world.emergency_event
        if ev.get("severity", 0) >= 0.5:
            return {
                "agent_id":    "emergency",
                "action_type": "emergency",
                "target_zone": ev.get("zone"),
                "parameters":  {
                    "vehicle_id": ev.get("vehicle_id", ""),
                    "cleared":    True,
                },
            }
        return None


# ---------------------------------------------------------------------------
# Agent Orchestrator — combines all agents into one decision
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Runs all agents in priority order and returns the winning action.
    Priority: Emergency > BACS > Global > Observe
    """

    def __init__(self):
        self.perception  = PerceptionAgent()
        self.emergency   = EmergencyDecisionAgent()
        self.bacs        = BACSAgent()
        self.global_coord = GlobalCoordinatorAgent()

    def act(self, obs: Dict, step: int = 0) -> Dict:
        world      = self.perception.perceive(obs)
        world.step = step

        action = self.emergency.decide(world)
        if action:
            return action

        action = self.bacs.decide(world, obs)
        if action:
            return action

        return self.global_coord.decide(world, obs)

    def get_prompt(self, obs: Dict, step: int = 0) -> str:
        """LLM-facing prompt for GRPO training."""
        world      = self.perception.perceive(obs)
        world.step = step
        return self.perception.to_prompt(world, obs)