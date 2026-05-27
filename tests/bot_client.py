# Reusable chatbot client for all test suites

import os
import requests
from dotenv import load_dotenv

load_dotenv()


class BotClient:

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.conversation_history = []

    def send(self, message: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:latest",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history
                ],
                "stream": False
            }
        )

        reply = response.json()["message"]["content"]

        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def reset(self):
        self.conversation_history = []

    def get_history(self) -> list:
        """Return full conversation history."""
        return self.conversation_history