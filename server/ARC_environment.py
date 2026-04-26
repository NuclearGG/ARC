"""
ARCE/server/ARC_environment.py
OpenEnv Environment subclass — wraps ARCIndiaSimulation (gymnasium.Env).

OpenEnv routes:
  POST /reset  → reset()  → ARCObservation
  POST /step   → step()   → ARCObservation
  GET  /state  → state    → ARCState
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
from openenv.core.env_server import Environment

from ARCE.models import ARCAction, ARCObservation, ARCState
from ARCE.simulation import ARCIndiaSimulation


class ARCEnvironment(Environment):
    """
    OpenEnv server wrapper around ARCIndiaSimulation.

    The gymnasium sim handles all physics and RL logic.
    This class translates between OpenEnv's Pydantic protocol
    and the gymnasium step/reset interface.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._sim = ARCIndiaSimulation(difficulty="medium")

    # ── OpenEnv required interface ────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ARCObservation:
        """Start a new episode. Returns first observation."""
        import random
        actual_seed = seed if seed is not None else random.randint(0, 99999)

        # Pass difficulty override if provided
        options = {}
        if "difficulty" in kwargs:
            options["difficulty"] = kwargs["difficulty"]

        _gym_obs, info = self._sim.reset(seed=actual_seed, options=options or None)
        rich = self._sim._rich_obs(step_reward=0.0)

        return ARCObservation(
            done               = False,
            reward             = 0.0,
            grid_snapshot      = rich["grid_snapshot"],
            zone_metrics       = rich["zone_metrics"],
            active_events      = rich["active_events"],
            pedestrian_density = rich["pedestrian_density"],
            emergency_status   = rich["emergency_status"],
            reward_breakdown   = rich["reward_breakdown"],
            message            = f"Episode started. seed={actual_seed} difficulty={self._sim.difficulty}",
            metadata           = info,
        )

    def step(
        self,
        action: ARCAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ARCObservation:
        """Apply action, advance simulation one tick."""
        arc_action = {
            "agent_id":    action.agent_id,
            "action_type": action.action_type,
            "target_zone": action.target_zone,
            "parameters":  action.parameters,
        }

        rich, reward, done, info = self._sim.step_from_arc_action(arc_action)

        return ARCObservation(
            done               = done,
            reward             = reward,
            grid_snapshot      = rich["grid_snapshot"],
            zone_metrics       = rich["zone_metrics"],
            active_events      = rich["active_events"],
            pedestrian_density = rich["pedestrian_density"],
            emergency_status   = rich["emergency_status"],
            reward_breakdown   = rich["reward_breakdown"],
            message            = (
                f"step={info['step']}  reward={reward:+.3f}  "
                f"cumulative={info['cumulative_reward']:+.2f}  done={done}"
            ),
            metadata = info,
        )

    @property
    def state(self) -> ARCState:
        """Full simulation state — exposed at GET /state."""
        s = self._sim.get_full_state()
        return ARCState(
            episode_id                 = s["episode_id"],
            step_count                 = s["step_count"],
            difficulty                 = s["difficulty"],
            grid_width                 = s["grid_width"],
            grid_height                = s["grid_height"],
            vehicles                   = s["vehicles"],
            pedestrians                = s["pedestrians"],
            obstacles                  = s["obstacles"],
            signal_phases              = s["signal_phases"],
            active_events              = s["active_events"],
            total_throughput           = s["total_throughput"],
            total_collisions           = s["total_collisions"],
            total_violations           = s["total_violations"],
            total_pedestrian_incidents = s["total_pedestrian_incidents"],
            emergency_successes        = s["emergency_successes"],
            emergency_failures         = s["emergency_failures"],
            cumulative_reward          = s["cumulative_reward"],
        )