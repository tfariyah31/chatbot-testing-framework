# tests/deepeval_config.py
# DeepEval configuration using OpenAI

import os
from deepeval.models import GPTModel
from dotenv import load_dotenv

load_dotenv()

# GPT-4o-mini — fast, cheap, excellent for evaluation
evaluator = GPTModel(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)