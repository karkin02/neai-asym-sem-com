from env import GenesisEnv
from llm_planner import VisionPlanner
from img_encoder import RawEncoder, JPEGEncoder
import genesis as gs
import time
import numpy as np
import pandas as pd
import os
from openai import OpenAI

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
        print(f"{name} | Episode {episode} | Step {step} | {action}")
        step += 1

    log["steps"] = step
    log["success"] = reward > 0
    return log

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60.0, 
    max_retries=3
)

TASK = "navigation"
gs.init(backend=gs.cpu)

b1 = VisionPlanner(
    encoder=RawEncoder(),
    task=TASK
)

b2 = VisionPlanner(
    encoder=JPEGEncoder(quality=30),
    task=TASK
)

results = []
N_EPISODES = 20

for name, planner in [("Raw RGB", b1), ("JPEG", b2)]:
    for ep in range(N_EPISODES):
        env = GenesisEnv(task=TASK)

        results.append(
            run_episode(env, planner, name, ep)
        )
        env.close()

df = pd.DataFrame([
    {
        "baseline": r["baseline"],
        "episode": r["episode"],
        "success": int(r["success"]),
        "avg_bytes_step": np.mean(r["bytes"]),
        "avg_latency": np.mean(r["latency"]),
        "avg_tokens": np.mean(r["tokens"]),
        "steps": r["steps"]
    }
    for r in results
])

print("Average Results")
print(df.groupby("baseline").mean(numeric_only=True))


