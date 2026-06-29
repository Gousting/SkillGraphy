"""
Hermes Agent integration example.

Shows how to use SkillGraph as a drop-in replacement for Hermes'
build_skills_system_prompt() function.

Before (Hermes default):
    system_prompt += build_skills_system_prompt()  # ~4000 tokens, all skills

After (with SkillGraph):
    system_prompt += adapter.build_prompt(user_message)  # ~300 tokens, top-8 skills
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skillgraph.adapters import HermesAdapter


def main():
    # 1. Create Hermes adapter (defaults to ~/.hermes/skills)
    adapter = HermesAdapter(
        skills_dir=os.path.expanduser("~/.hermes/skills"),
        backend="ollama",   # or "local" for offline, "openai" for cloud
        top_k=8,
    )

    # 2. Simulate a user message
    user_message = "Help me generate a hand-drawn architecture diagram"

    # 3. Retrieve relevant skills and format as prompt
    prompt = adapter.build_prompt(user_message)
    print("=== System Prompt Fragment ===")
    print(prompt)
    print()

    # 4. Compare: retrieve vs inject all
    stats = adapter.stats
    print(f"\n=== Stats ===")
    print(f"Total skills indexed: {stats['total_skills']}")
    print(f"Skills in prompt: {adapter.top_k}")
    print(f"Token savings: ~{stats['total_skills'] - adapter.top_k} skills omitted")

    # 5. Show what was retrieved
    print(f"\n=== Retrieved Skills ===")
    skills = adapter.retrieve(user_message)
    for skill in skills:
        print(f"  {skill.name:30s} score={skill.score:.3f} source={skill.source}")


if __name__ == "__main__":
    main()