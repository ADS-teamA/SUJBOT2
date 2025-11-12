#!/usr/bin/env python3
"""
Script to migrate agents from Anthropic client to unified provider.

Replaces:
1. `from anthropic import Anthropic` → remove import
2. `self.client = Anthropic(api_key=config.api_key)` → use provider factory
3. `self.client.messages.create(...)` → `self.provider.create_message(...)`
4. `response.content[0].text` → `response.text`
"""

import re
from pathlib import Path

AGENTS_TO_MIGRATE = [
    "citation_auditor.py",
    "classifier.py",
    "gap_synthesizer.py",
    "report_generator.py",
    "risk_verifier.py",
]

PROVIDER_INIT = """        # Initialize provider (auto-detects from model name: claude/gpt/gemini)
        try:
            from src.agent.providers.factory import create_provider

            self.provider = create_provider(model=config.model)
            logger.info(f"Initialized provider for model: {config.model}")
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            raise ValueError(
                f"Failed to initialize LLM provider for model {config.model}. "
                f"Ensure API keys are configured in environment and model name is valid."
            ) from e"""


def migrate_agent(file_path: Path):
    """Migrate single agent file."""
    print(f"Migrating {file_path.name}...")

    content = file_path.read_text()
    original = content

    # Step 1: Remove Anthropic import
    content = re.sub(
        r'from anthropic import Anthropic\n',
        '',
        content
    )

    # Step 2: Replace client initialization
    content = re.sub(
        r'        # Initialize Anthropic client\n        self\.client = Anthropic\(api_key=config\.api_key\)',
        PROVIDER_INIT,
        content
    )

    # Step 3: Replace API calls - need to handle multiple patterns
    # Pattern 1: Direct call with api_params dict
    content = re.sub(
        r'response = self\.client\.messages\.create\(\*\*api_params\)',
        '''# Call LLM via unified provider
            response = self.provider.create_message(
                messages=api_params["messages"],
                tools=[],
                system=api_params.get("system", self.system_prompt),
                max_tokens=api_params.get("max_tokens", self.config.max_tokens),
                temperature=api_params.get("temperature", self.config.temperature)
            )''',
        content
    )

    # Step 4: Replace response text extraction
    content = re.sub(
        r'response\.content\[0\]\.text',
        'response.text',
        content
    )

    # Check if anything changed
    if content != original:
        file_path.write_text(content)
        print(f"  ✓ {file_path.name} migrated")
        return True
    else:
        print(f"  ⚠ {file_path.name} - no changes made")
        return False


def main():
    agents_dir = Path("src/multi_agent/agents")
    migrated = 0

    for agent_file in AGENTS_TO_MIGRATE:
        file_path = agents_dir / agent_file
        if file_path.exists():
            if migrate_agent(file_path):
                migrated += 1
        else:
            print(f"  ✗ {agent_file} not found")

    print(f"\n✓ Migrated {migrated}/{len(AGENTS_TO_MIGRATE)} agents")


if __name__ == "__main__":
    main()
