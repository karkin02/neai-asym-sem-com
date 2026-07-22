# LeRobot Simulation Tutorial

**AUTHOR:** [Kevin Thomas](ket189@pitt.edu)

**CREATION DATE:** January 11, 2026  
**UPDATE DATE:** February 28, 2026

> **Companion resource** — this project draws heavily on ideas from
> [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403)
> by Francesco Capuano, Caroline Pascal, Adil Zouitine, Thomas Wolf, and Michel Aractingi.
> An interactive version is available on
> [HuggingFace Spaces](https://huggingface.co/spaces/lerobot/robot-learning-tutorial).
> That paper covers the full landscape of modern robot learning — from
> Reinforcement Learning and Behavioral Cloning to generalist
> Vision-Language-Action models — with ready-to-use examples implemented
> in [LeRobot](https://github.com/huggingface/lerobot).  The tutorial
> below focuses specifically on what *this* simulated environment
> implements and how you can run, understand, and extend it yourself.

---

## Table of Contents

1. [What Is Robot Learning?](#1-what-is-robot-learning)
2. [What Is Behavioral Cloning?](#2-what-is-behavioral-cloning)
3. [Project Overview](#3-project-overview)
4. [Installation](#4-installation)
5. [Environments](#5-environments)
6. [Expert Policies (the Teacher)](#6-expert-policies-the-teacher)
7. [The MLP Policy (the Student)](#7-the-mlp-policy-the-student)
8. [Training Pipeline](#8-training-pipeline)
9. [Running the Code](#9-running-the-code)
10. [Understanding the Results](#10-understanding-the-results)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Extending the Project](#12-extending-the-project)
13. [Glossary](#13-glossary)

---

## 1. What Is Robot Learning?

Classical robotics relies on *explicit* models — engineers hand-write the
physics equations of a robot's joints and use control theory to compute
exact motor commands.  This works well for structured factories but
breaks down when the world is messy, the model is inaccurate, or the
task changes.

**Robot learning** replaces those hand-coded equations with *data-driven*
methods.  Instead of telling the robot *how* to move, you show it
examples of successful movement and let it figure out the pattern via
machine learning.

Two dominant paradigms exist today:

| Paradigm                        | Core idea                                                                         | Needs a reward? | Needs demonstrations?   |
| ------------------------------- | --------------------------------------------------------------------------------- | --------------- | ----------------------- |
| **Reinforcement Learning (RL)** | The robot tries random actions and learns which ones lead to higher reward scores | Yes             | No (but demos can help) |
| **Behavioral Cloning (BC)**     | The robot watches an expert and learns to copy the expert's actions               | No              | Yes                     |

This project implements **Behavioral Cloning** — the simplest and most
practical entry point to robot learning.

---

## 2. What Is Behavioral Cloning?

Behavioral Cloning treats the problem as straightforward supervised
learning:

1. **Collect** a dataset of (observation, action) pairs from an expert
   (a human teleoperator, or a scripted policy that already knows how
   to solve the task).
2. **Train** a neural network to predict the expert's action given the
   same observation.
3. **Deploy** the trained network: feed it a live observation and
   execute whatever action it predicts.

Formally, given a dataset $\mathcal{D} = \{(o_i, a_i)\}_{i=1}^{N}$
the learning objective is:

$$\min_\theta \; \frac{1}{N}\sum_{i=1}^{N} \| f_\theta(o_i) - a_i \|^2$$

where $f_\theta$ is our neural network (an MLP in this project) and
$\theta$ are its learnable weights.

**Advantages:**
- No reward function to design — the demonstrations *are* the signal.
- Entirely offline training — no risk of a half-trained robot damaging
  hardware.
- Simple to implement and debug.

**Limitations:**
- The student can be at best as good as the teacher.
- Small prediction errors compound over time (*covariate shift*).
- Multi-modal demonstrations (multiple valid strategies) can confuse a
  simple model.

State-of-the-art methods such as **ACT** (Action Chunking with
Transformers), **Diffusion Policy**, and **pi-0** address these limits
with more expressive architectures.  See the
[HuggingFace tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)
for details.

---

## 3. Project Overview

```
lerobot/
  run_sim.py                   # CLI entry point
  env.py                       # EnvHub compatibility shim
  pyproject.toml               # Package metadata + dependencies
  TUTORIAL.md                  # You are here
  README.md                    # Quick-start reference
  lerobot_sim/
    envs/
      push_t.py                # PushT Gymnasium environment
      pick_place.py            # PickPlace environment
      reach.py                 # Reach environment
      configs.py               # Per-task config dataclasses
      factory.py               # make_sim_env() factory
    policies/
      policy.py                # BasePolicy, RandomPolicy, experts, MLPPolicy
      trainer.py               # 3-phase training pipeline
    datasets/
      dataset_manager.py       # Episode recording + HF Hub loader
    robots/
      sim_robot_arm.py         # 6-DOF simulated arm (SO-100 style)
    teleop/
      keyboard_teleop.py       # WASD / arrow-key teleoperation
      gamepad_teleop.py        # Gamepad input
    visualization/
      visualizer.py            # Pygame renderer with HUD
    utils/
      constants.py             # Colours, layout values
```

Everything runs on **pure NumPy** — there is no PyTorch dependency.
The MLP forward pass, backward pass, and Adam optimiser are all
implemented from scratch so you can inspect every gradient.

### Dependencies

| Package         | Version | Purpose                                       |
| --------------- | ------- | --------------------------------------------- |
| numpy           | 2.4.2   | Linear algebra, the MLP itself                |
| gymnasium       | 1.2.3   | Environment interface (step / reset / spaces) |
| pygame          | 2.6.1   | Live visualisation and teleoperation          |
| huggingface_hub | 1.5.0   | Dataset upload / download                     |

---

## 4. Installation

```bash
# Clone or navigate into the workspace
cd /path/to/lerobot

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install everything (including dev tools)
pip install ".[dev]"
```

Verify:

```bash
python -c "from lerobot_sim.envs.push_t import PushTSimEnv; print('OK')"
```

---

## 5. Environments

All environments are [Gymnasium](https://gymnasium.farama.org/) `Env`
subclasses.  They share a common config dataclass (`SimEnvConfig`) and
return observations as dictionaries — matching the LeRobot convention.

### 5.1 PushT (the main task)

**Goal:** a circular pusher must slide a T-shaped block until it
overlaps with a target outline.

| Property          | Value                                                                   |
| ----------------- | ----------------------------------------------------------------------- |
| Action space      | 2-D continuous: `(dx, dy)` in [-1, 1]                                   |
| Observation       | 6-D float32: `[agent_x, agent_y, block_x, block_y, target_x, target_y]` |
| Episode length    | 300 steps                                                               |
| FPS               | 10                                                                      |
| Success condition | distance(block, target) < 0.08                                          |

**How the physics work:**

1. **Agent moves:** `position += action * 0.05`, then clipped to [0, 1].
2. **Block pushed:** if the agent is closer than 0.06 to the block,
   the block is pushed 0.02 units away from the agent.
3. **Reward:** `1.0 - distance(block, target)`.  Higher is better.
4. **Success:** triggered when block–target distance drops below 0.08.

The observation is returned under the key `"agent_pos"` (following
LeRobot conventions for low-dimensional state).  On reset, the agent
always starts at `(0.5, 0.5)`, the block is randomised in `[0.2, 0.8]`,
and the target in `[0.3, 0.7]`.

### 5.2 PickPlace

A 6-DOF arm (SO-100 style) must pick up an object and place it at a
target location.  13-D observation, 7-D action (6 joint velocities +
gripper).

### 5.3 Reach

The arm must move its end-effector to a target position.  9-D
observation, 6-D action.

> **Note:** the scripted expert for PushT is carefully tuned and achieves
> a near-perfect success rate.  The PickPlace and Reach experts are
> simpler heuristics.  This tutorial focuses on PushT as the primary
> example.

---

## 6. Expert Policies (the Teacher)

Before we can train a student, we need demonstration data.  Each
environment has a **scripted expert policy** that already knows how to
solve the task.

### PushT Expert Strategy

The `PushTExpertPolicy` uses a two-phase approach:

```
Phase A  –  Reposition
    Compute the "approach" position:  0.08 units behind the block,
    opposite to the push direction (block → target).
    
    If the agent is more than 0.04 away from this approach position,
    move toward it.

Phase B  –  Push
    Once positioned behind the block, move directly toward the block.
    The environment physics handles the rest: when the agent gets close
    enough (< 0.06), the block slides toward the target.
```

A small amount of Gaussian noise (sigma = 0.02) is added to actions
to make the demonstrations more natural and diverse — this actually
helps the student generalise.

You can test the expert on its own:

```bash
python run_sim.py --task pusht --mode eval --eval-episodes 20
```

You should see 100 % success (assuming `best_policy.npz` is not present,
the code falls back to the expert for demonstration collection, but
for standalone expert testing you would call it from Python directly).

---

## 7. The MLP Policy (the Student)

The student is a simple **Multi-Layer Perceptron** (MLP) — a
feed-forward neural network with two hidden layers:

```
  state (6)
      │
      ▼
 ┌──────────┐
 │  Linear  │  state_dim (6) → hidden_dim (256)
 │  + ReLU  │
 └──────────┘
      │
      ▼
 ┌──────────┐
 │  Linear  │  hidden_dim (256) → hidden_dim (256)
 │  + ReLU  │
 └──────────┘
      │
      ▼
 ┌──────────┐
 │  Linear  │  hidden_dim (256) → action_dim (2)
 │  + tanh  │
 └──────────┘
      │
      ▼
  action (2)    values in [-1, 1]
```

### 7.1 Why tanh at the output?

Actions live in `[-1, 1]`.  The `tanh` activation naturally squashes
the output into this range, so the network never predicts an out-of-bound
action.

### 7.2 Weight Initialisation

All weights use **Xavier uniform** initialisation:

$$w_{ij} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},\; \sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}\right)$$

Biases are initialised to zero.  Xavier keeps the signal variance
roughly constant across layers, which prevents gradients from
exploding or vanishing at the start of training.

### 7.3 Forward Pass (NumPy)

```python
h1 = relu(state @ W1 + b1)      # shape: (256,)
h2 = relu(h1    @ W2 + b2)      # shape: (256,)
action = tanh(h2 @ W3 + b3)     # shape: (2,)
```

All operations are plain NumPy matrix multiplies.  No framework magic.

### 7.4 Backward Pass (Analytical Gradients)

We compute gradients by hand using the chain rule:

1.  **Output layer:**
    $\delta_{\text{out}} = \frac{2}{d} (a_{\text{pred}} - a_{\text{target}}) \odot (1 - a_{\text{pred}}^2)$
    The $(1 - a_{\text{pred}}^2)$ term is the derivative of tanh.

2.  **Hidden layer 2:**
    $\delta_{h2} = (\delta_{\text{out}} \cdot W_3^T) \odot \mathbb{1}[h2_{\text{pre}} > 0]$

3.  **Hidden layer 1:**
    $\delta_{h1} = (\delta_{h2} \cdot W_2^T) \odot \mathbb{1}[h1_{\text{pre}} > 0]$

Then each weight gradient is the outer product of the incoming
activation and the layer's delta:  $\nabla W_k = x_k^T \delta_k$.

### 7.5 Adam Optimiser

Plain SGD struggles with noisy gradients and saddle points.  We
implement the **Adam** optimiser from scratch:

For each parameter $\theta$:

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1)\, g_t$$
$$v_t = \beta_2 \, v_{t-1} + (1 - \beta_2)\, g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Learning rate ($\alpha$)            | 0.001 |
| $\beta_1$ (momentum decay)          | 0.9   |
| $\beta_2$ (variance decay)          | 0.999 |
| $\varepsilon$ (numerical stability) | 1e-8  |

Adam adapts the effective learning rate per-parameter based on the
history of gradients.  Parameters that receive large, consistent
gradients get smaller updates; parameters with small, noisy gradients
get larger updates.   This was the single biggest improvement during
development — switching from vanilla SGD to Adam took the success rate
from ~40 % to 100 %.

### 7.6 Saving & Loading

Weights are saved as NumPy `.npz` files:

```python
policy.save("./sim_output/best_policy.npz")
loaded = MLPPolicy.load("./sim_output/best_policy.npz")
```

The file stores `w1, b1, w2, b2, w3, b3` plus `state_dim` and
`action_dim` so the architecture is fully reconstructable.

---

## 8. Training Pipeline

The `Trainer` class orchestrates a three-phase pipeline.

### Phase 1 — Collect Demonstrations

```
for each demo episode (default: 20):
    reset the environment
    run the expert policy for up to 300 steps
    save every (state, action) pair
```

This produces a buffer of thousands of (state, action) transitions.
With 20 episodes of 300 steps each, you get up to 6 000 samples.

### Phase 2 — Train the Student

```
for each training step (default: 3000):
    sample a mini-batch of 32 random transitions from the buffer
    
    policy.begin_batch()          # zero the gradient accumulators
    
    for each sample in the batch:
        forward pass  →  MSE loss
        backward pass →  accumulate gradients
    
    policy.finish_batch()         # average gradients, run Adam update
    
    if this is the lowest loss so far:
        save best_policy.npz
```

Key detail: gradients are **accumulated** across the 32 samples and
applied once.  This is equivalent to a standard mini-batch gradient
descent step — much more stable than updating weights after every
single sample.

### Phase 3 — Evaluate

```
load best_policy.npz  (the checkpoint with lowest training loss)

for each eval episode (default: 5):
    reset the environment
    run the *trained* student policy for 300 steps
    record whether the task was successful

report mean reward and success rate
```

Loading the best checkpoint (not the final one) acts as a form of
early stopping — if training overshoots, the best model is still used.

---

## 9. Running the Code

### 9.1 Full Pipeline (train + evaluate)

```bash
python run_sim.py --task pusht --mode train
```

This runs all three phases.  Default settings:

| Flag              | Default        | What it controls                          |
| ----------------- | -------------- | ----------------------------------------- |
| `--task`          | `pusht`        | Which environment                         |
| `--mode`          | `train`        | `train`, `eval`, `teleop`, or `visualize` |
| `--demo-episodes` | `20`           | Expert demonstrations to collect          |
| `--train-steps`   | `3000`         | Number of gradient steps                  |
| `--eval-episodes` | `5`            | Evaluation rollouts                       |
| `--hidden-dim`    | `256`          | MLP hidden layer width                    |
| `--output-dir`    | `./sim_output` | Where checkpoints are saved               |
| `--seed`          | `42`           | Random seed                               |

With the defaults you should see **100 % success rate** on PushT.

### 9.2 Evaluate a Saved Model

```bash
python run_sim.py --task pusht --mode eval --eval-episodes 20
```

Loads `best_policy.npz` from `--output-dir` and runs 20 episodes.

### 9.3 Teleoperation

```bash
python run_sim.py --task pusht --mode teleop
```

Drive the agent with **WASD** or **arrow keys**.  Pygame renders live.

### 9.4 Visualisation

```bash
python run_sim.py --task pusht --mode visualize
```

Runs a random policy with live Pygame rendering — useful for
understanding the environment dynamics before training.

### 9.5 Higher Quality Training

For faster convergence, collect more demonstrations and train longer:

```bash
python run_sim.py --task pusht --mode train \
    --demo-episodes 50 \
    --train-steps 5000 \
    --hidden-dim 256
```

---

## 10. Understanding the Results

A typical training run looks like this:

```
============================================================
Phase 1: Collecting demonstrations …
  Demo policy: PushTExpertPolicy
============================================================
Phase 2: Training policy …
Training for 3000 steps ...
  [train] step     0 | loss = 0.452311
  [train] step    20 | loss = 0.089453
  [train] step    40 | loss = 0.031284
  ...
  [train] step  2980 | loss = 0.001547
Training complete. Final loss: 0.001423
============================================================
Phase 3: Evaluating policy …
  Loaded best checkpoint from ./sim_output/best_policy.npz
Eval (5 eps): reward=276.234, success=100.0%
```

### What to look for

| Metric        | Good sign                 | Bad sign           |
| ------------- | ------------------------- | ------------------ |
| Training loss | Drops steadily below 0.01 | Plateaus above 0.1 |
| Success rate  | 80–100 %                  | 0–20 %             |
| Mean reward   | > 250 (out of max ~300)   | < 200              |

### Troubleshooting

| Problem                            | Likely cause                          | Fix                                                          |
| ---------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| 0 % success                        | Not enough demos or training steps    | Increase `--demo-episodes 50` and `--train-steps 5000`       |
| Loss has NaN                       | Learning rate too high                | Use default lr (0.001) — it is baked into the MLPPolicy      |
| Loss wobbles but does not decrease | Batch size too small or too few demos | Collect more demos; the batch size is fixed at 32 internally |

---

## 11. Key Design Decisions

### Why NumPy instead of PyTorch?

Transparency.  Every matrix multiply, every gradient, every optimiser
step is visible in plain Python.  There is no autograd black box. This
makes the code ideal for learning and debugging.  For production work,
you would use the upstream LeRobot policies (ACT, Diffusion Policy,
pi-0) which are implemented in PyTorch.

### Why a 6-D observation?

The agent needs to know three things: *where am I*, *where is the
block*, and *where should the block go*.  Packing all six coordinates
into a single vector gives the MLP the full picture in one flat input
— no sequence modelling or attention needed.

### Why batch gradient accumulation?

Updating weights after every single sample is noisy and unstable.
Accumulating gradients over a batch of 32, averaging them, and then
doing one Adam step is mathematically equivalent to true mini-batch
gradient descent — the standard in deep learning.

### Why Adam over SGD?

Adam maintains per-parameter learning rate estimates.  In our
experiments, switching from plain SGD (lr = 0.001) to Adam (same lr)
improved the success rate from approximately 40 % to 100 % —
the most impactful single change we made during development.

### Why save the *best* checkpoint?

Training loss can oscillate or increase late in training (mild
overfitting, noise).  By always keeping the lowest-loss checkpoint and
loading it before evaluation, we get a form of early stopping for free.

---

## 12. Extending the Project

### Add a new environment

1. Create `lerobot_sim/envs/my_task.py` with a class subclassing
   `gym.Env`.
2. Add a config dataclass in `configs.py`.
3. Register it in `factory.py` and in the `_build_env_config` function
   in `run_sim.py`.
4. (Optional) Write a scripted expert in `policy.py` and register it in
   `Trainer._select_expert_policy`.

### Use a more advanced policy

The `BasePolicy` interface has two methods: `select_action(observation)`
and `reset()`.  Any LeRobot-compatible policy (ACT, Diffusion Policy)
can be wrapped to implement this interface.

### Upload datasets to HuggingFace Hub

The `SimDatasetRecorder` saves episodes in NumPy format.  Convert them
to the LeRobot Parquet + MP4 format and push to the Hub:

```python
from lerobot_sim.datasets.dataset_manager import HubDatasetLoader
# See dataset_manager.py for upload helpers
```

### Load via EnvHub

This project includes an `env.py` at the root, so LeRobot can load it
directly:

```python
from lerobot.envs.factory import make_env
envs = make_env("your-user/lerobot-sim", trust_remote_code=True)
```

---

## 13. Glossary

| Term                        | Definition                                                                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Behavioral Cloning (BC)** | Supervised learning from expert demonstrations: predict the expert's action given the same observation.                                         |
| **MLP**                     | Multi-Layer Perceptron — a feed-forward neural network with one or more hidden layers.                                                          |
| **Adam**                    | Adaptive Moment Estimation — an optimiser that maintains running estimates of gradient mean and variance to adapt per-parameter learning rates. |
| **Xavier Init**             | Weight initialisation that scales values by $\sqrt{6 / (n_{\mathrm{in}} + n_{\mathrm{out}})}$ to preserve signal variance.                      |
| **ReLU**                    | Rectified Linear Unit — $\max(0, x)$.  The simplest and most common hidden-layer activation.                                                    |
| **tanh**                    | Hyperbolic tangent — squashes values to $[-1, 1]$.  Used at the output layer to match the action range.                                         |
| **MSE**                     | Mean Squared Error — the loss function $\frac{1}{d}\sum_j (a_j^{\text{pred}} - a_j^{\text{target}})^2$.                                         |
| **Mini-batch**              | A random subset of the dataset used to estimate the gradient at each training step (here: 32 samples).                                          |
| **Covariate shift**         | When the distribution of observations seen at test time differs from training — a key challenge for BC.                                         |
| **Gymnasium**               | The standard Python API for RL environments (successor to OpenAI Gym).                                                                          |
| **LeRobot**                 | HuggingFace's open-source library for end-to-end robotics — datasets, policies, and hardware support.                                           |
| **EnvHub**                  | LeRobot feature that loads custom environments directly from a HF repository.                                                                   |

---

## Author

Kevin Thomas
- Creation Date: January 11, 2026
- Last Updated: February 28, 2026

---

*Happy learning!  If you find a bug or want to improve this tutorial,
open an issue or pull request.*
