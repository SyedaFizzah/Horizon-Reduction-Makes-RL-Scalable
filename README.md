# Horizon Reduction Makes RL Scalable — SHARSA vs GC-IQL

**Course:** CT-469 Reinforcement Learning  
**Assignment:** Research Paper Analysis & Implementation  
**Paper:** "Horizon Reduction Makes RL Scalable" — Park et al., 2025  
**Environment:** OGBench  
**Agents Compared:** SHARSA vs GC-IQL  

---

## What This Project Is About

This project is a study of a research paper that asks a simple but important question:

> *Why do most Reinforcement Learning (RL) algorithms stop improving even when you give them more data?*

The paper's answer is: **the horizon is too long.**

### What is a "horizon" in RL?

Imagine an agent trying to solve a maze where the goal is very far away. At every step, it has to decide if its move is helping or not.

In standard reinforcement learning, the agent only knows if it did well after reaching the goal, so it has to trace that success all the way back through many steps. The longer the path, the harder it becomes to figure out which early moves actually mattered. This weakening of feedback over long sequences is called the **curse of horizon**.

### What does SHARSA do differently?

Instead of trying to reach the final goal in one go, SHARSA breaks the journey into short pieces:

- It picks a **subgoal** that is only 25 steps ahead
- It reaches that subgoal first
- Then picks the next subgoal 25 steps further
- And so on until it reaches the final goal

---

## The Two Agents That We Compare

### GC-IQL (Baseline — the standard approach)
- Full name: Goal-Conditioned Implicit Q-Learning
- Uses **1-step TD backup** — learns from one step at a time
- Has a **flat policy** — directly tries to reach the final goal
- Horizon: full trajectory (~1000 steps)
- Struggles because errors accumulate over long horizons

### SHARSA (The proposed method)
- Full name: Subgoal-conditioned Hindsight Actor SARSA
- Uses **n-step SARSA** — looks 25 steps ahead instead of 1
- Has a **hierarchical policy** — high-level picks subgoal, low-level reaches it
- Horizon: only 25 steps
- Uses **flow-based behavioral cloning** — a generative model for actions
- Uses **rejection sampling** — generates 32 candidate subgoals, picks the best one

---

## Repository Structure

```
horizon-reduction/
│
├── agents/
│   ├── sharsa.py      ← SHARSA agent (the proposed method)
│   ├── gciql.py       ← GC-IQL agent (the baseline we compare against)
|
├── envs/              ← Environment wrappers (do not modify)
├── utils/             ← Helper functions — networks, datasets, logging
├── assets/            ← Images used in the paper's README
│
│── analysis of horizon reduction.ipynb
├── main.py            ← THE MAIN FILE — runs training for any agent
├── requirements.txt   ← All Python libraries needed
└── README.md          ← This file
```

**You only need to interact with `main.py` and the `agents/` folder.**  
Everything else runs automatically in the background.

---

## Setup Instructions

### Step 1 — Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

**Important:** Before doing anything, change your runtime to GPU:  
`Runtime → Change runtime type → T4 GPU → Save`

### Step 2 — Check your GPU

```python
!nvidia-smi
!free -h
```

You should see a Tesla T4 GPU and at least 10GB of free RAM.

### Step 3 — Install system dependencies

```python
!apt-get update -qq
!apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

import os
os.environ['MUJOCO_GL'] = 'osmesa'
print('MUJOCO_GL =', os.environ['MUJOCO_GL'])
```

### Step 4 — Clone the repository

```python
!git clone https://github.com/seohongpark/horizon-reduction.git
%cd horizon-reduction
!ls
```

### Step 5 — Install Python dependencies

```python
!pip install -r requirements.txt -q
!pip install -q "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install --upgrade wandb -q

import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())   # should show GPU
```

### Step 6 — Mount Google Drive (to save your logs)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/rl-logs/iql',    exist_ok=True)
os.makedirs('/content/drive/MyDrive/rl-logs/sharsa', exist_ok=True)
print('Drive mounted ✓')
```

### Step 7 — Login to Weights & Biases (for live tracking)

```python
import wandb
wandb.login()
```

---

## How to Reproduce Results

### Patch main.py to run fewer steps (required for Colab)

The paper trains for 5,000,000 steps. On Colab's free tier we use 5,000 steps to stay within compute limits. The relative comparison between agents is still valid.

```python
with open("main.py", "r") as f:
    code = f.read()

code = code.replace("flags.DEFINE_integer('offline_steps', 5000000,", "flags.DEFINE_integer('offline_steps', 5000,")
code = code.replace("flags.DEFINE_integer('eval_interval', 250000,",  "flags.DEFINE_integer('eval_interval', 1000,")
code = code.replace("flags.DEFINE_integer('log_interval', 10000,",    "flags.DEFINE_integer('log_interval', 100,")
code = code.replace("flags.DEFINE_integer('save_interval', 5000000,", "flags.DEFINE_integer('save_interval', 5000,")

with open("main.py", "w") as f:
    f.write(code)

print("Patched to 5k steps")
```

### Run GC-IQL (baseline)

```python
import os
os.makedirs('/content/logs', exist_ok=True)

!python main.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --offline_steps=5000 \
    --log_interval=100 \
    --eval_interval=1000 \
    --save_interval=5000 \
    --agent=agents/gciql.py \
    --agent.alpha=1 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.actor_geom_sample=True \
    2>&1 | tee /content/logs/iql.log

print('IQL done')
```

### Run SHARSA (proposed method)

```python
!python main.py \
    --env_name=pointmaze-medium-navigate-v0 \
    --offline_steps=5000 \
    --log_interval=100 \
    --eval_interval=1000 \
    --save_interval=5000 \
    --agent=agents/sharsa.py \
    --agent.q_agg=min \
    --agent.subgoal_steps=25 \
    --agent.actor_p_trajgoal=0.5 \
    --agent.actor_p_randomgoal=0.5 \
    --agent.actor_geom_sample=True \
    2>&1 | tee /content/logs/sharsa.log

print('SHARSA done')
```

> **Note:** The dataset for pointmaze auto-downloads on first run — no manual download needed.  
> Each run takes approximately 5–10 minutes on a T4 GPU.

### Plot results

```python
import re, numpy as np, matplotlib.pyplot as plt

def parse_log(path):
    steps, success = [], []
    try:
        with open(path) as f:
            for line in f:
                s = re.search(r'step[:\s=]+(\d+)', line, re.I)
                r = re.search(r'overall_success[:\s=]+([\d.]+)', line, re.I)
                if s and r:
                    steps.append(int(s.group(1)))
                    success.append(float(r.group(1)))
    except FileNotFoundError:
        print(f'Log not found: {path}')
    return steps, success

iql_s, iql_r       = parse_log('/content/logs/iql.log')
sharsa_s, sharsa_r = parse_log('/content/logs/sharsa.log')

plt.figure(figsize=(10, 5))
if iql_r:    plt.plot(iql_s, iql_r, label='GC-IQL', color='steelblue', linewidth=2.5, marker='o')
if sharsa_r: plt.plot(sharsa_s, sharsa_r, label='SHARSA', color='darkorange', linewidth=2.5, marker='s')
plt.xlabel('Training Steps')
plt.ylabel('Success Rate')
plt.title('SHARSA vs GC-IQL — pointmaze-medium')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/content/results.png', dpi=150)
plt.show()
```

---

## What the Key Parameters Mean

| Parameter | What it does | IQL value | SHARSA value |
|---|---|---|---|
| `offline_steps` | How many training steps to run | 5000 | 5000 |
| `eval_interval` | How often to test the agent | 1000 | 1000 |
| `subgoal_steps` | How far ahead the subgoal is | N/A | 25 |
| `q_agg` | How to combine two Q-values | N/A | min |
| `actor_p_trajgoal` | Probability of using a future state as goal | 0.5 | 0.5 |
| `alpha` | How strongly to follow the dataset | 1.0 | N/A |
| `discount` | How much to value future rewards | 0.999 | 0.999 |

---

## Why We Used pointmaze Instead of puzzle-4x5

The paper evaluates on four hard environments: `cube-octuple`, `puzzle-4x5`, `puzzle-4x6`, and `humanoidmaze-giant`. These require:

- Datasets of 100M–1B transitions (gigabytes of data)
- Millions of training steps to show meaningful results
- Multiple hours of GPU time per run

On Google Colab's free tier this is not feasible. We use `pointmaze-medium-navigate-v0` instead because:

- Dataset auto-downloads (no manual wget needed)
- Shows learning signal within 5000 steps
- Is from the same OGBench benchmark suite used in the paper
- The relative comparison between IQL and SHARSA is still valid

The absolute success rates will be lower than the paper reports, but the key finding — that SHARSA learns faster than IQL due to horizon reduction — should still be visible.

---

## Key Findings from the Paper

1. Standard RL algorithms like IQL **stop improving** even with 1000x more data on hard tasks
2. The reason is the **curse of horizon** — long horizons cause Q-value errors to accumulate
3. **Horizon reduction** (shortening the effective planning distance) consistently improves performance
4. SHARSA reduces **both** the value horizon (n-step SARSA) and the policy horizon (subgoal = 25 steps)
5. SHARSA achieves the **best scaling behavior** among all 10 methods tested in the paper

---

## Limitations of This Implementation

- We train for 5,000 steps vs the paper's 5,000,000 — results show early trends only
- We use pointmaze (simple 2D maze) vs the paper's harder manipulation/locomotion tasks
- Colab's free T4 GPU is slower than the hardware used in the paper
- Results may vary slightly between runs due to random seeds

---

## References

- **Paper:** Park et al. (2025). "Horizon Reduction Makes RL Scalable." arXiv:2506.04168
- **Code:** https://github.com/seohongpark/horizon-reduction
- **OGBench:** https://github.com/seohongpark/ogbench
- **JAX:** https://github.com/google/jax
- **Weights & Biases:**  https://wandb.ai/fizzah-masroor-ned-university-of-engineering-and-technology/horizon-reduction/runs/2ucamxef
