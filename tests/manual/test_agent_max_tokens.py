"""Test that AgentConfig loads AGENT_MAX_TOKENS from environment."""

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env
from dotenv import load_dotenv
load_dotenv()

def test_agent_max_tokens_loading():
    """Test AGENT_MAX_TOKENS is loaded from .env."""

    print("="*70)
    print("Testing AGENT_MAX_TOKENS Configuration")
    print("="*70)

    # Check environment variable
    env_value = os.getenv("AGENT_MAX_TOKENS")
    print(f"\n1. Environment variable AGENT_MAX_TOKENS: {env_value}")

    if not env_value:
        print("   ❌ AGENT_MAX_TOKENS not set in .env")
        return False

    # Load AgentConfig
    print("\n2. Loading AgentConfig.from_env()...")
    from src.agent.config import AgentConfig

    config = AgentConfig.from_env()
    print(f"   Config loaded: max_tokens={config.max_tokens}")

    # Verify it matches
    expected = int(env_value)
    if config.max_tokens == expected:
        print(f"\n✅ SUCCESS: AgentConfig.max_tokens = {config.max_tokens}")
        print(f"   This matches AGENT_MAX_TOKENS={expected} from .env")
        print(f"\n   Gemini will now use max_output_tokens={config.max_tokens}")
        print(f"   This should fix MAX_TOKENS errors!")
        return True
    else:
        print(f"\n❌ MISMATCH:")
        print(f"   Expected: {expected} (from .env)")
        print(f"   Got: {config.max_tokens} (from config)")
        return False

if __name__ == "__main__":
    success = test_agent_max_tokens_loading()
    sys.exit(0 if success else 1)
