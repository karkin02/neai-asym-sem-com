# LeRobot Simulation Environment

**AUTHOR:** [Kevin Thomas](ket189@pitt.edu)

**CREATION DATE:** January 11, 2026  
**UPDATE DATE:** February 28, 2026

Train a robot brain from zero to 100% success in 60 seconds flat — no hardware, pure NumPy, pure magic. Built on [HuggingFace LeRobot](https://huggingface.co/lerobot) standards.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[dev]"
```

## Train

```bash
python run_sim.py --task pusht --mode train
```

Collects demo episodes with a scripted expert, trains an MLP via behaviour-cloning (Adam optimiser), then evaluates. Saves `best_policy.npz` (lowest loss) and `last_policy.npz` to `--output-dir`.

Available tasks: `pusht`, `pick_place`, `reach`

| Flag              | Default        | Description                        |
| ----------------- | -------------- | ---------------------------------- |
| `--demo-episodes` | 20             | Expert demo episodes to collect    |
| `--train-steps`   | 3000           | Behaviour-cloning gradient steps   |
| `--eval-episodes` | 5              | Evaluation rollouts after training |
| `--hidden-dim`    | 256            | MLP hidden layer width             |
| `--output-dir`    | `./sim_output` | Where to save models and data      |
| `--seed`          | 42             | Random seed                        |

## Run Inference

```bash
python run_sim.py --task pusht --mode eval
```

Loads `best_policy.npz` from `--output-dir` and runs evaluation episodes. Requires `--mode train` first.

## Teleoperate

```bash
python run_sim.py --task reach --mode teleop
```

Drive the arm with WASD / arrow keys and record demonstrations.

## Visualize

```bash
python run_sim.py --task pick_place --mode visualize
```

PyGame live rendering of a random policy.

## Environments

| Task         | DOF | Action                   | Observation                                       |
| ------------ | --- | ------------------------ | ------------------------------------------------- |
| `pusht`      | 2-D | (dx, dy)                 | 6-D state (agent + block + target) + optional RGB |
| `pick_place` | 3-D | 6 joint vels + 1 gripper | 13-D state + optional 384x384 RGB                 |
| `reach`      | 3-D | 6 joint vels             | 9-D state + optional 384x384 RGB                  |

## Project Structure

```
run_sim.py                  CLI entry point
lerobot_sim/
  envs/                     Gymnasium environments + factory + configs
  policies/                 BasePolicy, RandomPolicy, Expert policies, MLPPolicy, Trainer
  datasets/                 Episode recording + HF Hub loader
  robots/                   6-DOF simulated arm (SO-100 style)
  teleop/                   Keyboard and gamepad input
  visualization/            PyGame renderer + HUD + replay
  utils/                    Constants + helpers
```

## LeRobot Compatibility

Observations use `observation.state` / `observation.image` keys. The factory returns `{suite: {task_id: VectorEnv}}` matching `lerobot.envs.factory.make_env`. Datasets are NumPy archives convertible to Parquet + MP4 for Hub upload.

### EnvHub

This project includes an `env.py` at the root so it can be loaded directly via [LeRobot EnvHub](https://huggingface.co/docs/lerobot/envhub):

```python
from lerobot.envs.factory import make_env
envs = make_env("your-user/lerobot-sim", trust_remote_code=True)
```

## Resources

- [LeRobot Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) — free hands-on course covering teleoperation, dataset recording, policy training, and inference
- [LeRobot Documentation](https://huggingface.co/docs/lerobot/index) — full API docs, installation, and hardware guides
- [LeRobot GitHub](https://github.com/huggingface/lerobot) — upstream source code

## Arduino UNO Q Deployment

The `lerobot-policy-deployment-app/` folder contains a complete [Arduino App Lab](https://www.arduino.cc/en/software/#app-lab-section) application that deploys the trained MLP policy to an [Arduino UNO Q](https://www.arduino.cc/product-uno-q). Enter state observations through a web interface, and the board drives servos in real time using the NumPy forward pass while displaying episode status on the 8 x 13 LED matrix. See [`lerobot-policy-deployment-app/README.md`](lerobot-policy-deployment-app/README.md) for full details.

## Author

Kevin Thomas
- Creation Date: January 11, 2026
- Last Updated: February 28, 2026

## License

Apache 2.0
