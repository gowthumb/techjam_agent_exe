#!/usr/bin/env python3
"""Make one minimal OpenAI-compatible LLM request and print its accounting."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.llm_client import call_llm


def main() -> int:
    response = call_llm("You are a helpful assistant.", "Say OK and nothing else.")
    print("response: " + response.text)
    print("input_tokens: %d" % response.input_tokens)
    print("output_tokens: %d" % response.output_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())