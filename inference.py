"""
ARC-India: GRPO Inference Script
Loads trained LoRA weights and runs the agent using HF TRL for tool-calling inference.

FIXES APPLIED:
  BUG-1  sys.path pointed at ARC/ dir itself, so 'from ARC.x' looked for ARC/ARC/x.
         Fixed: go up one more level to the project root.
  BUG-2  Optional missing from typing import → NameError on reset() signature.
  BUG-3  TRLAutoModelForCausalLM does not exist in trl → ImportError.
         Removed; FastLanguageModel from unsloth is sufficient.
  BUG-4  Hardcoded .to("cuda") crashes on CPU machines.
         Fixed: device = "cuda" if torch.cuda.is_available() else "cpu"
  BUG-5  episode_reward never updated — main loop called sim.step_from_arc_action()
         directly, bypassing env._step_env(). Fixed: route through env._step_env().
  BUG-6  Simulation double-stepped each iteration AND env._obs_rich / env._step
         went out of sync. Fixed: single step through env methods only.
  BUG-9  sim.reset() return value (gym_obs, info) never unpacked.
         Fixed: _gym_obs, _info = self.sim.reset(seed=actual_seed)
"""

import os
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
import sys
import argparse
from typing import Dict, List, Optional       # BUG-2 FIX: added Optional

# BUG-1 FIX: __file__ is ARC/inference.py; dirname(__file__) = ARC/
#            We need the project root (parent of ARC/) on sys.path so that
#            'from ARC.simulation import ...' resolves to ARC/simulation.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from unsloth import FastLanguageModel    # BUG-3 FIX: removed TRLAutoModelForCausalLM
    import torch
except ImportError:
    print("Error: Please install unsloth and torch:  pip install unsloth torch")
    sys.exit(1)

from ARC.simulation import ARCIndiaSimulation, ZONE_IDS, SIGNAL_PHASES
from ARC.agents import AgentOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Environment wrapper  (mirrors train.py ARCTRLEnv, fully method-complete)
# ─────────────────────────────────────────────────────────────────────────────

class ARCTRLEnv:
    """
    TRL-compatible Environment Wrapper for Inference.
    Matches the structure used during GRPO training.
    """
    def __init__(self, difficulty: str = "medium", seed: int = 42):
        self.sim          = ARCIndiaSimulation(difficulty=difficulty)
        self.orchestrator = AgentOrchestrator()
        self._seed        = seed
        self._step        = 0
        self._obs_rich: Dict = {}
        self.episode_reward  = 0.0

    def reset(self, seed: Optional[int] = None) -> str:
        actual_seed = seed if seed is not None else self._seed
        # BUG-9 FIX: sim.reset() returns (gym_obs, info) — unpack properly
        _gym_obs, _info = self.sim.reset(seed=actual_seed)
        self._obs_rich  = self.sim._rich_obs()
        self._step      = 0
        self.episode_reward = 0.0
        return self.orchestrator.get_prompt(self._obs_rich, self._step)

    # ── Tool methods exposed to LLM ──────────────────────────────────────────

    def signal(self, zone: str, phase: str) -> str:
        return self._step_env("signal", zone, {"phase": phase}, "llm_local")

    def reroute(self, zone: str, vehicle_ids: str) -> str:
        ids = [v.strip() for v in vehicle_ids.split(",") if v.strip()]
        return self._step_env("reroute", zone, {"vehicle_ids": ids}, "llm_global")

    def emergency_clear(self, vehicle_id: str, zone: str) -> str:
        return self._step_env(
            "emergency", zone,
            {"vehicle_id": vehicle_id, "cleared": True},
            "llm_emergency"
        )

    def enforce_compliance(self, zone: str) -> str:
        return self._step_env("comply", zone, {"vehicle_ids": []}, "llm_bacs")

    def observe(self) -> str:
        return self.orchestrator.get_prompt(self._obs_rich, self._step)

    # ── Internal step (single code-path for all tool calls) ──────────────────

    def _step_env(self, atype: str, zone: str, params: Dict, agent: str) -> str:
        action = {
            "agent_id":    agent,
            "action_type": atype,
            "target_zone": zone,
            "parameters":  params,
        }
        # step_from_arc_action returns 4-tuple: (rich_obs, reward, done, info)
        self._obs_rich, r, done, _ = self.sim.step_from_arc_action(action)
        self._step         += 1
        self.episode_reward += r          # BUG-5 FIX: reward now accumulated correctly
        return self.orchestrator.get_prompt(self._obs_rich, self._step)


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model_path: str, difficulty: str, max_steps: int):
    print(f"Loading model with Unsloth from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # BUG-4 FIX: don't assume CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = ARCTRLEnv(difficulty=difficulty)

    # Tool registry — used for fallback heuristic routing below
    tool_map = {
        "signal":    lambda z, p: env.signal(z, p),
        "reroute":   lambda z, v: env.reroute(z, v),
        "emergency": lambda vid, z: env.emergency_clear(vid, z),
        "comply":    lambda z, _: env.enforce_compliance(z),
        "observe":   lambda *_: env.observe(),
    }

    print(f"Starting inference loop  [{difficulty} difficulty, {max_steps} steps]...")
    current_prompt = env.reset()

    for i in range(max_steps):
        inputs = tokenizer([current_prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                use_cache=True,
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"\n[Step {i}] Agent Output:\n{response[:200]}...")

        # ── Parse model output for tool call (format: ACTION|zone|params) ──
        # Falls back to orchestrator heuristic if parse fails.
        # BUG-5/6 FIX: ALL stepping goes through env._step_env() so that
        # episode_reward, _step, and _obs_rich stay in sync.
        parsed = False
        for line in response.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2:
                atype = parts[0].strip().lower()
                zone  = parts[1].strip() if len(parts) > 1 else "Z4"
                param = parts[2].strip() if len(parts) > 2 else ""
                if atype == "signal" and param:
                    current_prompt = env.signal(zone, param)
                    parsed = True; break
                elif atype == "reroute":
                    current_prompt = env.reroute(zone, param)
                    parsed = True; break
                elif atype == "emergency" and param:
                    current_prompt = env.emergency_clear(param, zone)
                    parsed = True; break
                elif atype == "comply":
                    current_prompt = env.enforce_compliance(zone)
                    parsed = True; break
                elif atype == "observe":
                    current_prompt = env.observe()
                    parsed = True; break

        if not parsed:
            # Heuristic fallback — still routed through env so state stays consistent
            heuristic_action = env.orchestrator.act(env._obs_rich, i)
            current_prompt = env._step_env(
                heuristic_action["action_type"],
                heuristic_action.get("target_zone") or "Z4",
                heuristic_action.get("parameters", {}),
                heuristic_action.get("agent_id", "heuristic"),
            )

        # Check done via sim directly (env._step tracks this)
        if env._step >= env.sim.cfg["max_steps"]:
            print("Episode complete: max steps reached.")
            break

    state = env.sim.get_full_state()
    print("\n" + "=" * 40)
    print("  HF TRL INFERENCE SUMMARY")
    print("=" * 40)
    print(f"  Steps run    : {env._step}")
    print(f"  Total Reward : {env.episode_reward:.4f}")
    print(f"  Throughput   : {state['total_throughput']}")
    print(f"  Collisions   : {state['total_collisions']}")
    print(f"  Violations   : {state['total_violations']}")
    print(f"  Ped Incidents: {state['total_pedestrian_incidents']}")
    print(f"  Emg Successes: {state['emergency_successes']}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-India Inference")
    parser.add_argument("--model_path", required=True, help="Path to LoRA weights dir")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    run_inference(args.model_path, args.difficulty, args.steps)