---
title: ARCE Server
emoji: 🛣️
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# ARC (Automated Roadway Control)

# ARC (formerly ARC-India)

## 🚦 Problem Motivation

Urban mobility systems in India operate under extreme uncertainty:
- Mixed traffic (cars, bikes, buses, pedestrians)
- Weak or ignored lane discipline
- Dynamic and unpredictable human behavior
- Frequent violations and informal road usage

Traditional traffic systems fail because they assume structured, rule-based environments.

### ARC reframes this problem as:
> A multi-agent learning problem under chaos and uncertainty.

Instead of fixed rules, agents learn how to dynamically coordinate shared road space.

---

## 🌍 Environment Overview

ARC is built on top of **OpenEnv (latest release)** and simulates a realistic multi-agent traffic system.

### Environment includes:
- Vehicles (cars, bikes, buses)
- Pedestrians sharing road space
- Obstacles and dynamic road conditions
- Random real-world events:
  - Accidents
  - Congestion spikes
  - Road blockages
  - Rule violations

### Key idea:
The environment does NOT enforce strict traffic rules — agents must learn behavior.

---

## 🤖 How the System Works

ARC uses a **multi-agent reinforcement learning setup**:

### Agent Types:
- Perception Agent → understands global state
- Local Control Agents → manage zones/intersections
- Global Coordinator → optimizes system-wide flow
- Emergency Agent → prioritizes critical events
- Compliance Model → handles unpredictable behavior

### Learning Objective:
Agents optimize:
- Traffic flow
- Safety
- Fairness
- Emergency response efficiency

---

## 📊 Training & Results

We train using:
- HuggingFace TRL (GRPO)
- Unsloth for optimized fine-tuning

### Rewards track:
- Reduced congestion
- Lower collision rates
- Improved throughput
- Stable long-horizon coordination

### Evidence:
- Training loss curves
- Reward progression plots
- Multi-seed comparisons

📈 W&B Dashboard:
https://wandb.ai/nucorp/arc-india-grpo

---

## 🤗 Live Environment

Interactive simulation available here:

👉 https://huggingface.co/spaces/nucleargg/ARCE

Features:
- Real-time multi-agent traffic simulation
- Scenario switching (normal / emergency / congestion)
- Live visualization of agent decisions

---

## 📌 Additional Materials

All supplementary resources are linked below:

| Type | Link |
|------|------|
| Training Notebook (TRL / Unsloth) | https://www.kaggle.com/code/nuclearggind/arctrainer |
| Experiment Tracking (W&B) | https://wandb.ai/nucorp/arc-india-grpo |
| GitHub Repository | https://github.com/NuclearGG/ARC |
| Hugging Face Space (Env) | https://huggingface.co/spaces/nucleargg/ARCE |

---

## Materials
- Blog / Writeup: *[Add HF blog link here]*


> Note: No large video files are included in this repository. All media is referenced via external URLs only, as required.

---

## 📌 Summary

ARC demonstrates that:
> Traffic systems can be learned, not just programmed.

By modeling urban mobility as a **multi-agent reinforcement learning problem**, ARC enables adaptive coordination in chaotic real-world environments.

---


The environment models:
- Agent interaction in dynamic systems
- Multi-step decision making
- Real-time state evolution via actions
- Scalable simulation via Docker + HF Spaces

This Space serves as the **official deployed environment for ARC-India (now ARC).**

---

## 🌍 Environment Goal

ARC is designed to test how agents behave under:
- Dynamic environments
- Multi-agent interactions
- Sparse and delayed rewards
- Real-world inspired system constraints

It is a minimal but extensible OpenEnv-compatible environment.

---

## ⚙️ Quick Start (Python Client)

```python
from ARC import ArcAction, ArcEnv

# Create environment from Docker image
env = ArcEnv.from_docker_image("ARC-env:latest")

# Reset environment
result = env.reset()
print(result.observation.echoed_message)

# Step through environment
for msg in ["Hello", "World", "ARC"]:
    result = env.step(ArcAction(message=msg))
    print(result.observation.echoed_message)

env.close()