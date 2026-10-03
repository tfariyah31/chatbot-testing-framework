# ConvoQA — Conversational AI Testing Framework

An automated testing framework for evaluating conversational AI systems.

---

## What This Framework Tests

This framework tests NutriBot — a nutrition and recipe assistant 
built on GPT-4o-mini — across four testing dimensions:

- **Functional testing** — intent recognition, entity handling,
  dialogue flows, context retention
- **Semantic evaluation** — DeepEval-powered multi-metric scoring
  using GPT-4o-mini as evaluator (relevancy, safety, helpfulness,
  persona consistency, completeness)
- **Adversarial testing** — prompt injection, jailbreaking,
  social engineering, boundary probing
- **Regression testing** — 3-layer safety net covering capability,
  safety guardrails, and persona consistency

---

## Key Findings

During testing, 3 critical security vulnerabilities were discovered
and remediated:

| Bug | Category | Severity | Status |
|-----|----------|----------|--------|
| BUG-ADV-001 | System prompt extraction via developer claim | CRITICAL | Fixed in v1.1 |
| BUG-ADV-002 | Admin override injection — guardrail bypass | CRITICAL | Fixed in v1.1 |
| BUG-ADV-003 | DAN identity replacement jailbreak | CRITICAL | Fixed in v1.2 |

---

## Tech Stack

- Python 3.11
- Pytest — test runner and suite organisation
- DeepEval — LLM evaluation framework
- OpenAI GPT-4o-mini — bot under test + evaluator model
- Custom GEval metrics — plain English evaluation criteria

---

## Project Structure
```
convoqa-testing-framework/
│
├── tests/
│   ├── bot_client.py              ← reusable AI client
│   ├── deepeval_config.py         ← GPT-4o-mini evaluator
│   ├── metrics_factory.py         ← 6 named evaluation metrics
│   ├── evaluation_results.py      ← scoring + reporting engine
│   │
│   ├── functional/
│   │   ├── test_nutribot_functional.py   ← 15 keyword tests
│   │   ├── test_nutribot_deepeval.py     ← 7 semantic tests
│   │   └── test_nutribot_pipeline.py     ← multi-metric pipeline
│   │
│   ├── adversarial/
│   │   └── test_nutribot_adversarial.py  ← 14 adversarial tests
│   │
│   └── regression/
│       └── test_nutribot_regression.py   ← 13 regression tests
│
├── prompts/
│   ├── nutribot_prompt.txt        ← current prompt v1.2
│   └── versions/                  ← versioned prompt history
│       ├── nutribot_v1.0.txt
│       ├── nutribot_v1.1.txt
│       ├── nutribot_v1.2.txt
│       └── manifest.json
│
├── reports/                       ← all generated reports
└── data/

```

---

## Running the Tests

```bash
# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env
echo "OPENAI_API_KEY=your-key" >> .env

# Run all suites
python run_tests.py all

# Run specific suite
python run_tests.py regression
python run_tests.py adversarial

# Quick smoke test
python run_tests.py smoke
```

---

## Test Coverage

| Suite | Tests | Focus |
|-------|-------|-------|
| Functional | 22 | Intent, entities, context, semantic quality |
| Adversarial | 14 | Security, guardrails, manipulation resistance |
| Regression | 13 | Stability across prompt versions |
| **Total** | **49** | |

---

## Author

**Tasnim Fariyah** — 14 years in QA engineering, now building AI-powered quality systems.

[![GitHub](https://img.shields.io/badge/GitHub-tfariyah31-181717?logo=github)](https://github.com/tfariyah31)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-tasnim--fariyah-0A66C2?logo=linkedin)](https://www.linkedin.com/in/tasnim-fariyah/)
