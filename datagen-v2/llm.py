import os
import json
from openai import OpenAI
from dataclasses import dataclass, asdict
from typing import List, Dict
from threading import Lock


@dataclass
class TokenUsage:
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAILLM:
    def __init__(
        self, api_key: str = None, base_url: str = None, track_usage_at: str = None
    ):
        if track_usage_at is None:
            self.response_file = os.path.join(os.path.dirname(__file__), "response.txt")
        else:
            self.response_file = os.path.join(
                os.path.dirname(track_usage_at), "response.txt"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.track_usage_at = track_usage_at
        self.usage: Dict[str, TokenUsage] = {}
        self.lock = Lock()

    def completion(
        self, model: str, messages: List[Dict[str, str]], response_format: dict = None
    ) -> str:
        response = self.client.chat.completions.create(
            model=model, messages=messages, response_format=response_format
        )

        with self.lock:
            if model not in self.usage:
                self.usage[model] = TokenUsage()

            self.usage[model].completion_tokens += response.usage.completion_tokens
            self.usage[model].prompt_tokens += response.usage.prompt_tokens
            self.usage[model].total_tokens += response.usage.total_tokens

            with open(self.track_usage_at, "w") as fp:
                json.dump({k: asdict(v) for k, v in self.usage.items()}, fp, indent=4)

            # for debugging purposes, write the response to a file
            with open(self.response_file, "a") as fp:
                for msg in messages:
                    fp.write(f"role: {msg['role']}\n")
                    fp.write(f"content: {msg['content']}\n")
                fp.write(f"Response: {response.choices[0].message.content}\n")
                fp.write(f"Usage: {self.usage[model]}\n\n")
                fp.write("=" * 80 + "\n\n")

        return response
