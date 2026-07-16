from fastapi import FastAPI, Request
from server.planner import LLMPLanner

app = FastAPI(title="AsymSemCom Server")

# planner instantiated once at startup (load API client, not heavy model)
planner = LLMPLanner(provider="openai")


@app.post("/plan")
async def plan_action(request: Request):
    # raw JSON byyes sent by robot client
    body = await request.body()
    # task goal passed as header
    task_goal = request.headers.get("X-Task-Goal", "Explore the environment.")

    result = planner.plan(body, task_goal)
    return result


@app.get("/helath")
async def health():
    return {"status": "ok"}