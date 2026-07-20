# Global Pytest configuration and DeepEval setup

import deepeval
from deepeval.models import OllamaModel

def pytest_configure(config):
    """Configure DeepEval to use local Ollama model."""
    pass