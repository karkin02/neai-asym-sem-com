import json
import time
from server.prompt_templates import build_scene_prompt

MAX_TOKENS=150

class LLMPlanner:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model

        # instantiate whichever API client that is used
        if provider == "openai":
            import openai
            self.client = openai.OpenAI()
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    def plan(self, payload_bytes: bytes, task_goal: str) -> dict:
        start = time.time()

        # decode JSON payload back into dict
        payload = json.loads(payload_bytes.decode("utf-8"))
        # convert structured detections into a natural langauge prompt for LLM
        prompt = build_scene_prompt(payload, task_goal)

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
            )
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
            )
        
        action_text = response.choices[0].message.content
        token_count = response.usage.total_tokens

        latency_ms = (time.time() - start) * 1000

        return {
            "action": action_text.strip(),
            "latency_ms": round(latency_ms, 2),
            "token_count": token_count,
        }