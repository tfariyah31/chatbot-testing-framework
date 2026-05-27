# Functional test suite for NutriBot

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot_client import BotClient

# ─── Load system prompt ───────────────────────────────────────

def load_prompt(filename: str) -> str:
    prompt_path = os.path.join(
        os.path.dirname(__file__), 
        "../../prompts", 
        filename
    )
    with open(prompt_path, "r") as f:
        return f.read()

# ─── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def nutribot():
    """Fresh NutriBot instance for each test."""
    return BotClient(system_prompt=load_prompt("nutribot_prompt.txt"))

@pytest.fixture
def nutribot_with_context():
    """NutriBot with pre-loaded conversation context."""
    bot = BotClient(system_prompt=load_prompt("nutribot_prompt.txt"))
    bot.send("Hi, I'm vegetarian and I'm allergic to nuts")
    return bot

# ─── Happy Path Tests ─────────────────────────────────────────

class TestHappyPath:

    def test_basic_greeting(self, nutribot):
        """Bot should respond helpfully to a greeting."""
        response = nutribot.send("Hello")
        assert len(response) > 0, "Response should not be empty"
        assert len(response) < 500, "Greeting should be concise"

    def test_recipe_request(self, nutribot):
        """Bot should provide a recipe when asked."""
        response = nutribot.send(
            "Can you give me a healthy dinner recipe?"
        )
        response_lower = response.lower()
        assert any(word in response_lower for word in 
                   ["ingredient", "recipe", "cook", "prepare", "cup", "tablespoon"]), (
            f"Expected recipe content but got: {response}"
        )

    def test_nutritional_info(self, nutribot):
        """Bot should provide nutritional information."""
        response = nutribot.send(
            "What are the nutritional benefits of spinach?"
        )
        response_lower = response.lower()
        assert any(word in response_lower for word in 
                   ["vitamin", "mineral", "nutrient", "iron", 
                    "calcium", "protein", "benefit"]), (
            f"Expected nutritional info but got: {response}"
        )

    def test_dietary_restriction_guidance(self, nutribot):
        """Bot should handle dietary restrictions appropriately."""
        response = nutribot.send(
            "I'm vegan. What should I eat to get enough protein?"
        )
        response_lower = response.lower()
        assert any(word in response_lower for word in 
                   ["protein", "bean", "lentil", "tofu", 
                    "tempeh", "quinoa", "chickpea"]), (
            f"Expected vegan protein sources but got: {response}"
        )

# ─── Intent Recognition Tests ─────────────────────────────────

class TestIntentRecognition:

    @pytest.mark.parametrize("utterance", [
        "Give me a recipe",
        "What can I cook tonight?",
        "I need meal ideas",
        "Suggest something healthy to eat",
        "What should I make for dinner?",
    ])
    def test_recipe_intent_variations(self, nutribot, utterance):
        """Bot should recognise recipe intent across varied phrasings."""
        response = nutribot.send(utterance)
        response_lower = response.lower()
        assert any(word in response_lower for word in 
                   ["recipe", "ingredient", "cook", "prepare", 
                    "meal", "dish", "make"]), (
            f"Failed to recognise recipe intent for: '{utterance}'\n"
            f"Got response: {response}"
        )

    @pytest.mark.parametrize("utterance", [
        "What's good about avocado?",
        "Is quinoa healthy?",
        "Tell me about the benefits of turmeric",
        "How much protein is in chicken?",
    ])
    def test_nutrition_info_intent(self, nutribot, utterance):
        """Bot should recognise nutrition info intent."""
        response = nutribot.send(utterance)
        assert len(response) > 30, (
            f"Response too short for nutrition query: '{utterance}'\n"
            f"Got: {response}"
        )

# ─── Context Retention Tests ──────────────────────────────────

class TestContextRetention:

    def test_remembers_dietary_restriction(self, nutribot):
        """Bot should remember dietary restrictions across turns."""
        nutribot.send("I'm vegetarian and I hate mushrooms")
        response = nutribot.send("Suggest a recipe for me")
        response_lower = response.lower()

        assert "meat" not in response_lower and \
               "chicken" not in response_lower and \
               "beef" not in response_lower and \
               "pork" not in response_lower, (
            f"Bot forgot vegetarian restriction. Got: {response}"
        )

    def test_context_carried_from_fixture(self, nutribot_with_context):
        """Bot should use context established in fixture."""
        response = nutribot_with_context.send(
            "What recipe would you recommend for me?"
        )
        response_lower = response.lower()
        assert "nut" not in response_lower or \
               "nut-free" in response_lower or \
               "without nut" in response_lower, (
            f"Bot ignored nut allergy context. Got: {response}"
        )