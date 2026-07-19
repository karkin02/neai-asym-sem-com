from env import GenesisEnv
from baseline1 import GPTVisionPlanner
from baseline2 import JPEGVisionPlanner
import time
import numpy as np
import pandas as pd

def run_episode(env, planner, name, episode):
    obs = env.reset()
    done = False
    step = 0

    log = {
        "baseline": name,
        "episode": episode,
        "bytes": [],
        "latency": [],
        "tokens": [],
        "success": False
    }

    while not done:
        start = time.time()
        action, llm_info = planner.plan(obs)
        latency = time.time() - start

        log["latency"].append(latency)
        log["bytes"].append(llm_info["bytes"])
        log["tokens"].append(llm_info["tokens"])

        obs, reward, done, info = env.step(action)
        print(name, "episode", episode, "step", step, action)
        step += 1

    log["steps"] = step
    log["success"] = reward > 0
    return log

# Create planners
b1 = GPTVisionPlanner()
b2 = JPEGVisionPlanner()

results = []
N_EPISODES = 20

for ep in range(N_EPISODES):
    env = GenesisEnv()
    results.append(
        run_episode(env, b1, "Raw RGB", ep)
    )
    env.close()

for ep in range(N_EPISODES):
    env = GenesisEnv()
    results.append(
        run_episode(env, b2, "JPEG", ep)
    )
    env.close()

rows = []

for r in results:
    rows.append({
        "baseline": r["baseline"],
        "episode": r["episode"],
        "success": r["success"],
        "avg_bytes_step": np.mean(r["bytes"]),
        "avg_latency": np.mean(r["latency"]),
        "avg_tokens": np.mean(r["tokens"]),
        "steps": r["steps"]
    })

df = pd.DataFrame(rows)
print(df.groupby("baseline").mean(numeric_only=True))


