"""
ARC/simulation.py
Gymnasium-based simulation engine for ARC-India.

ARCIndiaSimulation subclasses gymnasium.Env — the standard RL interface.
Observation space : Dict (zone metrics, grid, events)
Action space      : Dict (action_type, target_zone, parameters)

This class is the source-of-truth simulation. It is wrapped by
ARC_environment.py for the OpenEnv HTTP server.
"""

import math
import uuid
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_PHASES    = ["NS_GREEN", "EW_GREEN", "ALL_RED", "PEDESTRIAN_CROSS"]
SIGNAL_PHASE_IDX = {p: i for i, p in enumerate(SIGNAL_PHASES)}
EVENT_TYPES      = ["accident", "congestion", "road_repair", "emergency"]

ZONE_IDS = ["Z0","Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8"]
ZONE_POSITIONS: Dict[str, Tuple[int,int]] = {
    "Z0":(2,2),"Z1":(5,2),"Z2":(8,2),
    "Z3":(2,5),"Z4":(5,5),"Z5":(8,5),
    "Z6":(2,8),"Z7":(5,8),"Z8":(8,8),
}

DIFFICULTY_CONFIG = {
    "easy":   {"n_vehicles":10, "n_pedestrians": 5,  "event_prob":0.02, "violation_prob":0.05, "max_steps":100},
    "medium": {"n_vehicles":25, "n_pedestrians":15,  "event_prob":0.05, "violation_prob":0.15, "max_steps":200},
    "hard":   {"n_vehicles":50, "n_pedestrians":30,  "event_prob":0.10, "violation_prob":0.30, "max_steps":400},
}

DIR_DELTA = {"N":(0,-1),"S":(0,1),"E":(1,0),"W":(-1,0)}
DIRECTIONS = ["N","S","E","W"]
VEHICLE_TYPES = ["car","bike","bus"]


# ─────────────────────────────────────────────────────────────────────────────
# Entity factories
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_vehicle(vid: str, cfg: dict, rng: random.Random,
                   grid_w: int, grid_h: int) -> Dict:
    weights = ["car","car","car","bike","bike","bus"]
    return {
        "id":        vid,
        "type":      rng.choice(weights),
        "x":         rng.randint(0, grid_w - 1),
        "y":         rng.randint(0, grid_h - 1),
        "speed":     round(rng.uniform(0.5, 2.0), 2),
        "direction": rng.choice(DIRECTIONS),
        "dest":      [rng.randint(0, grid_w-1), rng.randint(0, grid_h-1)],
        "compliant": rng.random() > cfg["violation_prob"],
        "stopped":   False,
        "wait_steps":0,
    }

def _spawn_pedestrian(pid: str, rng: random.Random,
                      grid_w: int, grid_h: int) -> Dict:
    return {
        "id":   pid,
        "x":    rng.randint(0, grid_w - 1),
        "y":    rng.randint(0, grid_h - 1),
        "dest": [rng.randint(0, grid_w-1), rng.randint(0, grid_h-1)],
        "safe": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium Environment
# ─────────────────────────────────────────────────────────────────────────────

class ARCIndiaSimulation(gym.Env):
    """
    ARC-India: Mixed Urban Mobility Simulation

    Gymnasium environment (gymnasium.Env) simulating Indian urban traffic:
    vehicles, pedestrians, signal control, dynamic events, emergency response.

    Observation space: Dict of structured traffic data
    Action space:      Dict specifying agent_id, action_type, zone, parameters

    Designed to wrap into OpenEnv's Environment for HTTP serving.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    def __init__(self, difficulty: str = "medium", render_mode: Optional[str] = None):
        super().__init__()
        self.difficulty  = difficulty
        self.render_mode = render_mode
        self.cfg         = DIFFICULTY_CONFIG[difficulty]
        self.grid_w      = 10
        self.grid_h      = 10

        # ── Gymnasium spaces ──────────────────────────────────────────────

        # Action space: encoded as MultiDiscrete for gym compatibility
        # [action_type_idx, zone_idx, param_phase_idx]
        # Training wrappers can use this directly; OpenEnv uses dict actions.
        self.action_space = spaces.Dict({
            "action_type_idx": spaces.Discrete(5),    # 0=observe,1=signal,2=reroute,3=emergency,4=comply
            "zone_idx":        spaces.Discrete(9),    # Z0–Z8
            "phase_idx":       spaces.Discrete(4),    # NS_GREEN,EW_GREEN,ALL_RED,PED_CROSS
        })

        # Observation space: flat vectors per zone + global counters
        self.observation_space = spaces.Dict({
            # Per zone: [queue_len(0–50), avg_wait(0–100), violations(0–20), phase_idx(0–3)]
            "zone_features": spaces.Box(
                low=0.0, high=100.0, shape=(9, 4), dtype=np.float32
            ),
            # Global: [throughput, collisions, violations, ped_incidents, n_events, step_frac]
            "global_features": spaces.Box(
                low=0.0, high=1.0, shape=(6,), dtype=np.float32
            ),
            # Emergency: [active(0/1), severity(0–1), eta_frac(0–1)]
            "emergency_features": spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            ),
        })

        # Internal state (populated by reset)
        self._rng:         random.Random      = random.Random(0)
        self.vehicles:     Dict[str, Dict]    = {}
        self.pedestrians:  Dict[str, Dict]    = {}
        self.obstacles:    Dict[str, Dict]    = {}
        self.signal_phases: Dict[str, str]    = {}
        self.active_events: List[Dict]        = []

        self.episode_id:               str   = ""
        self.step_count:               int   = 0
        self.total_throughput:         int   = 0
        self.total_collisions:         int   = 0
        self.total_violations:         int   = 0
        self.total_pedestrian_incidents:int  = 0
        self.emergency_successes:      int   = 0
        self.emergency_failures:       int   = 0
        self.cumulative_reward:        float = 0.0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment. Returns (gym_obs, info).
        Accepts optional difficulty override via options={"difficulty": "hard"}.
        """
        super().reset(seed=seed)

        if options and "difficulty" in options:
            self.difficulty = options["difficulty"]
            self.cfg        = DIFFICULTY_CONFIG[self.difficulty]

        actual_seed = seed if seed is not None else random.randint(0, 99999)
        self._rng = random.Random(actual_seed)

        self.episode_id  = str(uuid.uuid4())
        self.step_count  = 0
        self.cumulative_reward = 0.0

        # Reset counters
        self.total_throughput           = 0
        self.total_collisions           = 0
        self.total_violations           = 0
        self.total_pedestrian_incidents = 0
        self.emergency_successes        = 0
        self.emergency_failures         = 0
        self.active_events              = []
        self.obstacles                  = {}

        # Signals
        self.signal_phases = {z: self._rng.choice(SIGNAL_PHASES) for z in ZONE_IDS}

        # Spawn entities
        n_v = self.cfg["n_vehicles"]
        n_p = self.cfg["n_pedestrians"]
        self.vehicles = {
            f"V{i}": _spawn_vehicle(f"V{i}", self.cfg, self._rng, self.grid_w, self.grid_h)
            for i in range(n_v)
        }
        self.pedestrians = {
            f"P{i}": _spawn_pedestrian(f"P{i}", self._rng, self.grid_w, self.grid_h)
            for i in range(n_p)
        }

        info = {"episode_id": self.episode_id, "difficulty": self.difficulty, "seed": actual_seed}
        return self._gym_obs(), info

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Gymnasium step: action → (obs, reward, terminated, truncated, info)

        action dict keys:
          action_type_idx (int) : 0=observe,1=signal,2=reroute,3=emergency,4=comply
          zone_idx        (int) : 0–8 → Z0–Z8
          phase_idx       (int) : 0–3 → SIGNAL_PHASES

        For OpenEnv integration use step_from_arc_action() with a dict action.
        """
        self.step_count += 1

        # Decode gym action
        arc_action = self._decode_gym_action(action)

        # Apply action + advance world
        action_bonus = self._apply_arc_action(arc_action)
        self._move_vehicles()
        self._move_pedestrians()
        self._update_events()

        col, viol, ped = self._check_safety()
        self.total_collisions           += col
        self.total_violations           += viol
        self.total_pedestrian_incidents += ped

        tp = self._count_throughput()
        self.total_throughput += tp

        reward = action_bonus + self._compute_reward(tp, col, viol, ped)
        self.cumulative_reward += reward

        max_steps  = self.cfg["max_steps"]
        terminated = False                         # no natural terminal state
        truncated  = self.step_count >= max_steps

        info = {
            "step":              self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "throughput":        self.total_throughput,
            "collisions":        self.total_collisions,
            "violations":        self.total_violations,
            "ped_incidents":     self.total_pedestrian_incidents,
            "emergency_ok":      self.emergency_successes,
        }
        return self._gym_obs(), reward, terminated, truncated, info

    def step_from_arc_action(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        Dict-action step used by ARC_environment.py (OpenEnv server).
        action keys: agent_id, action_type, target_zone, parameters
        Returns: (rich_obs_dict, reward, done, info)
        """
        self.step_count += 1

        action_bonus = self._apply_arc_action(action)
        self._move_vehicles()
        self._move_pedestrians()
        self._update_events()

        col, viol, ped = self._check_safety()
        self.total_collisions           += col
        self.total_violations           += viol
        self.total_pedestrian_incidents += ped

        tp = self._count_throughput()
        self.total_throughput += tp

        reward = action_bonus + self._compute_reward(tp, col, viol, ped)
        self.cumulative_reward += reward

        done = self.step_count >= self.cfg["max_steps"]
        info = {
            "step": self.step_count,
            "cumulative_reward": self.cumulative_reward,
        }
        return self._rich_obs(step_reward=reward), reward, done, info

    def render(self) -> Optional[str]:
        """ANSI text render of current grid state."""
        if self.render_mode != "ansi":
            return None
        grid = [["." for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        for v in self.vehicles.values():
            sym = {"car":"C","bike":"B","bus":"U"}.get(v["type"],"V")
            grid[v["y"]][v["x"]] = sym
        for p in self.pedestrians.values():
            if 0 <= p["y"] < self.grid_h and 0 <= p["x"] < self.grid_w:
                grid[p["y"]][p["x"]] = "W"
        lines = ["  " + " ".join(str(c) for c in range(self.grid_w))]
        for i, row in enumerate(grid):
            lines.append(f"{i} " + " ".join(row))
        # Signal legend
        lines.append("\nSignals: " + "  ".join(f"{z}:{self.signal_phases[z][:2]}" for z in ZONE_IDS))
        return "\n".join(lines)

    def close(self):
        pass

    # ── Gym obs builder ───────────────────────────────────────────────────────

    def _gym_obs(self) -> Dict:
        """Returns observation matching self.observation_space."""
        zone_feats = np.zeros((9, 4), dtype=np.float32)
        for i, z in enumerate(ZONE_IDS):
            zx, zy = ZONE_POSITIONS[z]
            nearby = [v for v in self.vehicles.values()
                      if abs(v["x"]-zx) <= 2 and abs(v["y"]-zy) <= 2]
            queue   = sum(1 for v in nearby if v["stopped"])
            avg_wait= sum(v.get("wait_steps",0) for v in nearby) / max(1, len(nearby))
            viols   = sum(1 for v in nearby if not v["compliant"])
            ph_idx  = SIGNAL_PHASE_IDX.get(self.signal_phases[z], 0)
            zone_feats[i] = [
                min(queue, 50),
                min(avg_wait, 100),
                min(viols, 20),
                ph_idx,
            ]

        max_steps = self.cfg["max_steps"]
        global_feats = np.array([
            min(self.total_throughput,   500) / 500,
            min(self.total_collisions,    50) / 50,
            min(self.total_violations,   200) / 200,
            min(self.total_pedestrian_incidents, 100) / 100,
            min(len(self.active_events),  10) / 10,
            self.step_count / max_steps,
        ], dtype=np.float32)

        emg = next((e for e in self.active_events if e["type"] == "emergency"), None)
        emg_feats = np.array([
            1.0 if emg else 0.0,
            float(emg["severity"]) if emg else 0.0,
            min(emg.get("eta", 10), 10) / 10 if emg else 0.0,
        ], dtype=np.float32)

        return {
            "zone_features":      zone_feats,
            "global_features":    global_feats,
            "emergency_features": emg_feats,
        }

    # ── Rich obs for OpenEnv ──────────────────────────────────────────────────

    def _rich_obs(self, step_reward: float = 0.0) -> Dict:
        """Returns the full human-readable observation for OpenEnv."""
        # ASCII grid
        grid = [["." for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        for v in self.vehicles.values():
            sym = {"car":"C","bike":"B","bus":"U"}.get(v["type"],"V")
            grid[v["y"]][v["x"]] = sym
        for p in self.pedestrians.values():
            if 0 <= p["y"] < self.grid_h and 0 <= p["x"] < self.grid_w:
                grid[p["y"]][p["x"]] = "W"

        # Zone metrics
        zone_metrics: Dict[str, Dict] = {}
        for z in ZONE_IDS:
            zx, zy = ZONE_POSITIONS[z]
            nearby = [v for v in self.vehicles.values()
                      if abs(v["x"]-zx) <= 2 and abs(v["y"]-zy) <= 2]
            zone_metrics[z] = {
                "phase":     self.signal_phases[z],
                "queue_len": sum(1 for v in nearby if v["stopped"]),
                "avg_wait":  round(sum(v.get("wait_steps",0) for v in nearby)
                                   / max(1, len(nearby)), 2),
                "violations":sum(1 for v in nearby if not v["compliant"]),
            }

        # Pedestrian density
        ped_density = {z: 0 for z in ZONE_IDS}
        for p in self.pedestrians.values():
            best = min(ZONE_IDS, key=lambda z: math.hypot(
                p["x"] - ZONE_POSITIONS[z][0], p["y"] - ZONE_POSITIONS[z][1]))
            ped_density[best] += 1

        emergency = next((e for e in self.active_events if e["type"] == "emergency"), None)

        return {
            "grid_snapshot":      grid,
            "zone_metrics":       zone_metrics,
            "active_events":      list(self.active_events),
            "pedestrian_density": ped_density,
            "emergency_status":   emergency,
            "reward_breakdown": {
                "step_reward":       step_reward,
                "throughput_bonus":  self.total_throughput   * 1.0,
                "collision_penalty":-self.total_collisions   * 3.0,
                "violation_penalty":-self.total_violations   * 0.5,
                "ped_penalty":      -self.total_pedestrian_incidents * 2.0,
                "emergency_bonus":   self.emergency_successes * 2.0,
            },
        }

    # ── Gym action decoder ────────────────────────────────────────────────────

    def _decode_gym_action(self, action: Dict) -> Dict:
        """Converts gym MultiDiscrete action → ARC dict action."""
        atype_map  = ["observe","signal","reroute","emergency","comply"]
        atype      = atype_map[int(action.get("action_type_idx", 0))]
        zone       = ZONE_IDS[int(action.get("zone_idx", 4))]
        phase      = SIGNAL_PHASES[int(action.get("phase_idx", 0))]
        return {
            "agent_id":    "gym_agent",
            "action_type": atype,
            "target_zone": zone,
            "parameters":  {"phase": phase},
        }

    # ── World mechanics ───────────────────────────────────────────────────────

    def _apply_arc_action(self, action: Dict) -> float:
        bonus  = 0.0
        atype  = action.get("action_type", "observe")
        zone   = action.get("target_zone")
        params = action.get("parameters", {})

        if atype == "signal" and zone in self.signal_phases:
            phase = params.get("phase", "NS_GREEN")
            if phase in SIGNAL_PHASES:
                self.signal_phases[zone] = phase
                bonus += 0.1

        elif atype == "reroute":
            for vid in params.get("vehicle_ids", []):
                if vid in self.vehicles:
                    self.vehicles[vid]["direction"] = self._rng.choice(DIRECTIONS)
                    bonus += 0.05

        elif atype == "emergency":
            vid     = params.get("vehicle_id", "")
            cleared = params.get("cleared", False)
            if cleared and vid:
                self.emergency_successes += 1
                bonus += 2.0
                self.active_events = [
                    e for e in self.active_events
                    if not (e["type"] == "emergency" and e.get("vehicle_id") == vid)
                ]
            else:
                self.emergency_failures += 1
                bonus -= 1.0

        elif atype == "comply":
            for vid in params.get("vehicle_ids", []):
                if vid in self.vehicles and not self.vehicles[vid]["compliant"]:
                    self.vehicles[vid]["compliant"] = True
                    bonus += 0.2

        return bonus

    def _move_vehicles(self):
        for v in self.vehicles.values():
            if v["stopped"]:
                v["wait_steps"] += 1
                continue
            phase = self._nearest_phase(v["x"], v["y"])
            if phase == "ALL_RED" and v["compliant"]:
                v["stopped"]     = True
                v["wait_steps"] += 1
                continue
            dx, dy = DIR_DELTA.get(v["direction"], (0,0))
            v["x"] = max(0, min(self.grid_w-1, v["x"] + dx))
            v["y"] = max(0, min(self.grid_h-1, v["y"] + dy))
            v["stopped"] = False
            if self._rng.random() < 0.05:                # chaotic lane change
                v["direction"] = self._rng.choice(DIRECTIONS)

    def _move_pedestrians(self):
        for p in self.pedestrians.values():
            tx, ty = p["dest"]
            dx = (1 if tx > p["x"] else -1) if tx != p["x"] else 0
            dy = (1 if ty > p["y"] else -1) if ty != p["y"] else 0
            if self._rng.random() < 0.7:
                p["x"] = max(0, min(self.grid_w-1, p["x"] + dx))
                p["y"] = max(0, min(self.grid_h-1, p["y"] + dy))
            if [p["x"], p["y"]] == [tx, ty]:
                p["dest"] = [self._rng.randint(0, self.grid_w-1),
                             self._rng.randint(0, self.grid_h-1)]

    def _update_events(self):
        self.active_events = [e for e in self.active_events if e["ttl"] > 0]
        for e in self.active_events:
            e["ttl"] -= 1
        if self._rng.random() < self.cfg["event_prob"]:
            etype = self._rng.choice(EVENT_TYPES)
            zone  = self._rng.choice(ZONE_IDS)
            ev: Dict[str,Any] = {
                "type":     etype,
                "zone":     zone,
                "severity": round(self._rng.uniform(0.3, 1.0), 2),
                "ttl":      self._rng.randint(5, 20),
            }
            if etype == "emergency":
                ev["vehicle_id"] = f"AMB-{self._rng.randint(1,9):02d}"
                ev["eta"]        = self._rng.randint(3, 10)
            self.active_events.append(ev)

    def _check_safety(self) -> Tuple[int, int, int]:
        collisions = violations = ped_incidents = 0
        occupancy: Dict[Tuple, List] = {}
        for vid, v in self.vehicles.items():
            occupancy.setdefault((v["x"], v["y"]), []).append(vid)
        for pos, ids in occupancy.items():
            if len(ids) > 1:
                collisions += len(ids) - 1

        # Pedestrian incident: only count if pedestrian is on a road cell
        # (road cells are x ∈ {2,5,8} or y ∈ {2,5,8}), not the full grid.
        # This prevents spurious incidents from random initial spawn overlap.
        road_xs = {2, 5, 8}
        road_ys = {2, 5, 8}
        for p in self.pedestrians.values():
            on_road = p["x"] in road_xs or p["y"] in road_ys
            if on_road and (p["x"], p["y"]) in occupancy:
                ped_incidents += 1
                p["safe"] = False
            else:
                p["safe"] = True

        for v in self.vehicles.values():
            if not v["compliant"] and self._nearest_phase(v["x"], v["y"]) == "ALL_RED":
                violations += 1
        return collisions, violations, ped_incidents

    def _count_throughput(self) -> int:
        count = 0
        for v in self.vehicles.values():
            if [v["x"], v["y"]] == v["dest"]:
                count += 1
                v["dest"] = [self._rng.randint(0, self.grid_w-1),
                             self._rng.randint(0, self.grid_h-1)]
        return count

    def _compute_reward(self, tp: int, col: int, viol: int, ped: int) -> float:
        """
        Rebalanced reward function.

        Design goals:
        - Positive baseline: vehicles moving = positive reward every step
        - Penalties proportional but not overwhelming
        - Long-wait penalty capped so it cannot dominate
        - No hard clipping here — reward functions in train_grpo normalise
        """
        n_v   = max(1, len(self.vehicles))
        n_p   = max(1, len(self.pedestrians))

        # ── All components are per-vehicle normalised so reward scale is
        #    comparable across easy/medium/hard difficulties. ──

        # Flow: fraction of vehicles currently moving → [0, +1.0]
        moving = sum(1 for v in self.vehicles.values() if not v["stopped"])
        r  = (moving / n_v) * 1.0

        # Throughput bonus: per-vehicle proportion that arrived this step
        r += (tp / n_v) * 1.0

        # Speed bonus: normalised avg speed (speed range ~0.5–2.0, norm by 2)
        avg_spd = sum(v["speed"] for v in self.vehicles.values()) / n_v
        r += (avg_spd / 2.0) * 0.2

        # Collision penalty: per-vehicle rate, capped at 1.0
        r -= min(col / n_v, 1.0) * 2.0

        # Violation penalty: per-vehicle rate, capped at 1.0
        r -= min(viol / n_v, 1.0) * 0.5

        # Pedestrian incident: per-pedestrian rate, capped at 1.0
        r -= min(ped / n_p, 1.0) * 0.5

        # Long-wait penalty: fraction of vehicles waiting too long, small weight
        long_wait = sum(1 for v in self.vehicles.values() if v.get("wait_steps", 0) > 8)
        r -= (long_wait / n_v) * 0.3

        return round(r, 4)

    def _nearest_phase(self, x: int, y: int) -> str:
        best = min(ZONE_IDS, key=lambda z: math.hypot(
            x - ZONE_POSITIONS[z][0], y - ZONE_POSITIONS[z][1]))
        return self.signal_phases.get(best, "NS_GREEN")

    # ── State export (for OpenEnv /state endpoint) ────────────────────────────

    def get_full_state(self) -> Dict:
        return {
            "episode_id":               self.episode_id,
            "step_count":               self.step_count,
            "difficulty":               self.difficulty,
            "grid_width":               self.grid_w,
            "grid_height":              self.grid_h,
            "vehicles":                 self.vehicles,
            "pedestrians":              self.pedestrians,
            "obstacles":                self.obstacles,
            "signal_phases":            self.signal_phases,
            "active_events":            self.active_events,
            "total_throughput":         self.total_throughput,
            "total_collisions":         self.total_collisions,
            "total_violations":         self.total_violations,
            "total_pedestrian_incidents":self.total_pedestrian_incidents,
            "emergency_successes":      self.emergency_successes,
            "emergency_failures":       self.emergency_failures,
            "cumulative_reward":        self.cumulative_reward,
        }