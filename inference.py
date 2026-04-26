"""
ARC-India Hybrid Inference (HF + OpenAI fallback)
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional

os.environ["UNSLOTH_DISABLE_TRITON"] = "1"

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch
from unsloth import FastLanguageModel

from openai import OpenAI

from ARCE.simulation import ARCIndiaSimulation
from ARCE.agents import AgentOrchestrator


# ─────────────────────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def openai_decide(prompt: str) -> Dict:
    """
    Fallback planner that returns structured tool action JSON.
    """
    system = """
You are a traffic control decision system.

Return ONLY valid JSON:
{
  "action_type": "signal|reroute|emergency|comply|observe",
  "zone": "Z1-ZN",
  "parameters": {}
}
"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {
            "action_type": "observe",
            "zone": "Z4",
            "parameters": {}
        }


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

class ARCTRLEnv:
    def __init__(self, difficulty="medium", seed=42):
        self.sim = ARCIndiaSimulation(difficulty=difficulty)
        self.orchestrator = AgentOrchestrator()

        self._step = 0
        self._seed = seed
        self._obs_rich = {}
        self.episode_reward = 0.0

    def reset(self, seed: Optional[int] = None):
        seed = seed or self._seed
        self.sim.reset(seed=seed)
        self._obs_rich = self.sim._rich_obs()
        self._step = 0
        self.episode_reward = 0.0
        return self.orchestrator.get_prompt(self._obs_rich, self._step)

    def _step_env(self, action: Dict):
        self._obs_rich, r, done, _ = self.sim.step_from_arc_action(action)
        self._step += 1
        self.episode_reward += r
        return self.orchestrator.get_prompt(self._obs_rich, self._step)

    def observe(self):
        return self.orchestrator.get_prompt(self._obs_rich, self._step)


# ─────────────────────────────────────────────────────────────
# HF parsing (robust JSON extractor)
# ─────────────────────────────────────────────────────────────

def parse_model_output(text: str):
    """
    Try to extract structured action from HF model.
    Expected format:
    ACTION|ZONE|PARAMS
    """
    try:
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                return {
                    "action_type": parts[0].strip().lower(),
                    "zone": parts[1].strip(),
                    "parameters": {"raw": parts[2]} if len(parts) > 2 else {}
                }
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# Inference Loop
# ─────────────────────────────────────────────────────────────

def run_inference(model_path: str, difficulty: str, steps: int):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = ARCTRLEnv(difficulty=difficulty)

    prompt = env.reset()

    print(f"\nRunning hybrid inference on {difficulty}...\n")

    for i in range(steps):

        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                use_cache=True,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"\n[HF STEP {i}]")
        print(text[:200])

        action = parse_model_output(text)

        # ─────────────────────────────────────────────
        # HF SUCCESS PATH
        # ─────────────────────────────────────────────
        if action and action["action_type"] in {
            "signal", "reroute", "emergency", "comply", "observe"
        }:
            prompt = env._step_env(action)

        # ─────────────────────────────────────────────
        # OPENAI FALLBACK PATH
        # ─────────────────────────────────────────────
        else:
            print("⚠️ Falling back to OpenAI planner...")
            action = openai_decide(prompt)
            prompt = env._step_env(action)

        if env._step >= env.sim.cfg["max_steps"]:
            break

    print("\n──────── FINAL REPORT ────────")
    print("Steps:", env._step)
    print("Reward:", env.episode_reward)
    print("State:", env.sim.get_full_state())


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--steps", type=int, default=50)

    args = parser.parse_args()

    run_inference(args.model_path, args.difficulty, args.steps)