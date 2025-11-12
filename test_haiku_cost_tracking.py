"""
COMPREHENSIVE COST TRACKING TEST FOR HAIKU MODELS

This test verifies that:
1. ALL LLM API calls are tracked (orchestrator + agents + tool loops)
2. Input vs Output tokens are correctly distinguished
3. Model-specific pricing is applied (Haiku: $1 input, $5 output per 1M tokens)
4. Prompt caching discounts are correctly applied (90% discount on cache reads)
5. Final total matches manual calculation

Test Strategy:
- Spy on ALL provider.create_message() calls
- Capture usage data from each call
- Manually calculate expected cost
- Compare with CostTracker total
- Verify no calls are missed

Author: Claude Code (2025-11-12)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    call_number: int
    context: str  # orchestrator, agent_name, tool_name
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int

    def calculate_cost(self) -> float:
        """
        Calculate cost for this call using official Anthropic pricing.

        Haiku 4.5 pricing (November 2025):
        - Input: $1.00 per 1M tokens
        - Output: $5.00 per 1M tokens
        - Cache reads: $0.10 per 1M tokens (10% of input price)
        - Cache writes: $1.25 per 1M tokens (1.25x input price)

        Returns:
            Cost in USD
        """
        # Haiku 4.5 pricing
        INPUT_PRICE = 1.00 / 1_000_000
        OUTPUT_PRICE = 5.00 / 1_000_000
        CACHE_READ_PRICE = 0.10 / 1_000_000  # 90% discount
        CACHE_WRITE_PRICE = 1.25 / 1_000_000  # 25% markup

        cost = (
            self.input_tokens * INPUT_PRICE +
            self.output_tokens * OUTPUT_PRICE +
            self.cache_read_tokens * CACHE_READ_PRICE +
            self.cache_creation_tokens * CACHE_WRITE_PRICE
        )

        return cost

    def __repr__(self) -> str:
        return (
            f"Call #{self.call_number} [{self.context}]: "
            f"{self.input_tokens} in, {self.output_tokens} out, "
            f"cache: {self.cache_read_tokens} read, {self.cache_creation_tokens} created "
            f"→ ${self.calculate_cost():.6f}"
        )


class LLMCallSpy:
    """
    Spy wrapper that intercepts ALL LLM calls and records usage data.

    This wraps provider.create_message() to capture every API call
    made during the multi-agent workflow execution.
    """

    def __init__(self):
        self.calls: List[LLMCallRecord] = []
        self.call_count = 0

    def wrap_provider_create_message(self, original_method):
        """
        Wrap provider.create_message() to spy on calls.

        Args:
            original_method: Original create_message method

        Returns:
            Wrapped method that records usage data
        """
        def wrapper(self_provider, *args, **kwargs):
            # Call original method
            response = original_method(self_provider, *args, **kwargs)

            # Extract context from stack (which agent/component called this)
            import traceback
            stack = traceback.extract_stack()

            # Find calling context (agent name, orchestrator, etc.)
            context = "unknown"
            for frame in reversed(stack):
                if "agent" in frame.filename.lower():
                    # Extract agent name from frame
                    if "orchestrator" in frame.filename:
                        context = "orchestrator"
                    elif "extractor" in frame.filename:
                        context = "extractor_agent"
                    elif "classifier" in frame.filename:
                        context = "classifier_agent"
                    elif "compliance" in frame.filename:
                        context = "compliance_agent"
                    elif "risk_verifier" in frame.filename:
                        context = "risk_verifier_agent"
                    elif "citation_auditor" in frame.filename:
                        context = "citation_auditor_agent"
                    elif "gap_synthesizer" in frame.filename:
                        context = "gap_synthesizer_agent"
                    elif "report_generator" in frame.filename:
                        context = "report_generator_agent"
                    elif "agent_base" in frame.filename:
                        context = "agent_autonomous_loop"
                    break

            # Record this call
            self.call_count += 1

            # Extract usage data from response
            usage = getattr(response, "usage", {})
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_read = usage.get("cache_read_tokens", 0)
                cache_creation = usage.get("cache_creation_tokens", 0)
            else:
                # Pydantic model
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                cache_read = getattr(usage, "cache_read_input_tokens", 0)
                cache_creation = getattr(usage, "cache_creation_input_tokens", 0)

            # Get model name
            model = getattr(response, "model", "unknown")

            record = LLMCallRecord(
                call_number=self.call_count,
                context=context,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation
            )

            self.calls.append(record)

            logger.info(f"📞 {record}")

            return response

        return wrapper

    def get_total_tokens(self) -> Dict[str, int]:
        """Get total tokens across all calls."""
        return {
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cache_read_tokens": sum(c.cache_read_tokens for c in self.calls),
            "cache_creation_tokens": sum(c.cache_creation_tokens for c in self.calls),
        }

    def get_total_cost(self) -> float:
        """Calculate total cost manually from all calls."""
        return sum(c.calculate_cost() for c in self.calls)

    def print_summary(self):
        """Print detailed summary of all calls."""
        print("\n" + "=" * 80)
        print("LLM CALL TRACKING SUMMARY")
        print("=" * 80)
        print(f"Total LLM calls: {self.call_count}")
        print()

        # Group by context
        by_context: Dict[str, List[LLMCallRecord]] = {}
        for call in self.calls:
            if call.context not in by_context:
                by_context[call.context] = []
            by_context[call.context].append(call)

        for context, calls in sorted(by_context.items()):
            print(f"\n{context.upper()}:")
            for call in calls:
                print(f"  {call}")

            context_cost = sum(c.calculate_cost() for c in calls)
            print(f"  Subtotal: ${context_cost:.6f}")

        print("\n" + "-" * 80)

        tokens = self.get_total_tokens()
        print(f"Total input tokens:    {tokens['input_tokens']:,}")
        print(f"Total output tokens:   {tokens['output_tokens']:,}")
        print(f"Cache read tokens:     {tokens['cache_read_tokens']:,}")
        print(f"Cache creation tokens: {tokens['cache_creation_tokens']:,}")

        print(f"\n💰 MANUAL CALCULATED COST: ${self.get_total_cost():.6f}")
        print("=" * 80 + "\n")


async def test_haiku_cost_tracking_end_to_end():
    """
    End-to-end test of Haiku cost tracking.

    This test:
    1. Sends a real query through multi-agent system
    2. Spies on ALL LLM API calls
    3. Manually calculates expected cost
    4. Compares with CostTracker reported cost
    5. Verifies accuracy within 0.1%
    """
    from src.multi_agent.runner import MultiAgentRunner
    from src.cost_tracker import get_global_tracker, reset_global_tracker
    from src.agent.providers.anthropic_provider import AnthropicProvider

    # Step 1: Setup
    print("\n🔧 STEP 1: Setup")
    print("-" * 80)

    # Reset cost tracker
    reset_global_tracker()
    tracker = get_global_tracker()

    # Create spy
    spy = LLMCallSpy()

    # Check if we have vector store
    vector_store_path = Path("vector_db")
    if not vector_store_path.exists():
        print("❌ ERROR: vector_db not found. Run indexing first:")
        print("   uv run python run_pipeline.py data/document.pdf")
        return False

    # Load config
    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ ERROR: config.json not found. Copy from config.json.example")
        return False

    with open(config_path) as f:
        config = json.load(f)

    # Verify Anthropic API key
    api_key = config.get("api_keys", {}).get("anthropic_api_key")
    if not api_key or api_key == "your-key-here":
        print("❌ ERROR: ANTHROPIC_API_KEY not set in config.json")
        return False

    print(f"✅ Vector store: {vector_store_path}")
    print(f"✅ Config loaded")
    print(f"✅ API key: {api_key[:10]}...")

    # Step 2: Patch AnthropicProvider.create_message to spy on calls
    print("\n🔍 STEP 2: Install spy wrapper")
    print("-" * 80)

    original_create_message = AnthropicProvider.create_message
    AnthropicProvider.create_message = spy.wrap_provider_create_message(original_create_message)

    print("✅ Spy installed on AnthropicProvider.create_message()")

    # Step 3: Execute query
    print("\n🚀 STEP 3: Execute test query")
    print("-" * 80)

    # Use a query that will trigger multiple agents
    # This should hit: orchestrator → extractor → classifier → compliance → report_generator
    test_query = "What are the key privacy requirements for handling user data?"

    print(f"Query: {test_query}")
    print()

    try:
        # Create runner with Haiku model explicitly
        runner = MultiAgentRunner(
            vector_store_path=str(vector_store_path),
            model="claude-haiku-4-5-20251001"  # Explicit Haiku model
        )

        print("Starting multi-agent execution...")
        print()

        # Execute query
        result_generator = runner.run(test_query)

        # Collect all events
        final_result = None
        async for event in result_generator:
            if event.get("type") == "final":
                final_result = event
                break

        if not final_result:
            print("❌ ERROR: No final result returned")
            return False

        print(f"✅ Query completed: {final_result.get('success')}")
        print(f"   Agent sequence: {final_result.get('agent_sequence', [])}")
        print(f"   Answer length: {len(final_result.get('final_answer', ''))} chars")

    except Exception as e:
        print(f"❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original method
        AnthropicProvider.create_message = original_create_message

    # Step 4: Analyze captured calls
    print("\n📊 STEP 4: Analyze captured LLM calls")
    spy.print_summary()

    # Step 5: Compare with CostTracker
    print("🔬 STEP 5: Verify CostTracker accuracy")
    print("-" * 80)

    tracker_cost = tracker.get_total_cost()
    manual_cost = spy.get_total_cost()

    print(f"CostTracker reported: ${tracker_cost:.6f}")
    print(f"Manual calculation:   ${manual_cost:.6f}")

    # Calculate difference
    if manual_cost > 0:
        diff_percent = abs(tracker_cost - manual_cost) / manual_cost * 100
        diff_absolute = abs(tracker_cost - manual_cost)

        print(f"Difference:           ${diff_absolute:.6f} ({diff_percent:.2f}%)")

        # Step 6: Verify pricing correctness
        print("\n✅ STEP 6: Verify pricing correctness")
        print("-" * 80)

        # Check official pricing
        print("Official Haiku 4.5 pricing (November 2025):")
        print("  Input:  $1.00 per 1M tokens")
        print("  Output: $5.00 per 1M tokens")
        print("  Cache read: $0.10 per 1M tokens (90% discount)")
        print("  Cache write: $1.25 per 1M tokens (25% markup)")
        print()

        # Breakdown
        tokens = spy.get_total_tokens()
        input_cost = tokens["input_tokens"] * (1.00 / 1_000_000)
        output_cost = tokens["output_tokens"] * (5.00 / 1_000_000)
        cache_read_cost = tokens["cache_read_tokens"] * (0.10 / 1_000_000)
        cache_write_cost = tokens["cache_creation_tokens"] * (1.25 / 1_000_000)

        print("Cost breakdown:")
        print(f"  Input tokens:    {tokens['input_tokens']:,} × $1.00/1M = ${input_cost:.6f}")
        print(f"  Output tokens:   {tokens['output_tokens']:,} × $5.00/1M = ${output_cost:.6f}")
        print(f"  Cache read:      {tokens['cache_read_tokens']:,} × $0.10/1M = ${cache_read_cost:.6f}")
        print(f"  Cache creation:  {tokens['cache_creation_tokens']:,} × $1.25/1M = ${cache_write_cost:.6f}")
        print(f"  Total:           ${manual_cost:.6f}")
        print()

        # Step 7: Final verdict
        print("\n🎯 STEP 7: Final verdict")
        print("=" * 80)

        # Tolerance: 0.1% (very strict)
        TOLERANCE_PERCENT = 0.1

        if diff_percent <= TOLERANCE_PERCENT:
            print(f"✅ PASS: Cost tracking accurate within {TOLERANCE_PERCENT}%")
            print(f"   Difference: ${diff_absolute:.6f} ({diff_percent:.4f}%)")
            print()
            print("Summary:")
            print(f"  - Captured {spy.call_count} LLM API calls")
            print(f"  - Total tokens: {sum(tokens.values()):,}")
            print(f"  - Total cost: ${tracker_cost:.6f}")
            print(f"  - Haiku pricing correctly applied ✅")
            print(f"  - Input/output distinction correct ✅")
            print(f"  - Cache discounts applied ✅")
            print()
            return True
        else:
            print(f"❌ FAIL: Cost tracking inaccurate (>{TOLERANCE_PERCENT}%)")
            print(f"   Expected: ${manual_cost:.6f}")
            print(f"   Got:      ${tracker_cost:.6f}")
            print(f"   Difference: ${diff_absolute:.6f} ({diff_percent:.2f}%)")
            print()
            print("Possible issues:")
            print("  - Some LLM calls not tracked")
            print("  - Wrong pricing model used")
            print("  - Input/output tokens swapped")
            print("  - Cache discounts not applied")
            print()
            return False
    else:
        print("❌ FAIL: No LLM calls captured (spy not working?)")
        return False


async def test_cost_tracker_pricing_tables():
    """
    Test that CostTracker has correct pricing for all Haiku models.

    This verifies the PRICING dict in cost_tracker.py matches
    official Anthropic pricing.
    """
    from src.cost_tracker import PRICING

    print("\n🔍 Testing CostTracker pricing tables")
    print("=" * 80)

    # Official Haiku pricing (November 2025)
    OFFICIAL_HAIKU_PRICING = {
        "input": 1.00,
        "output": 5.00,
    }

    # Check all Haiku aliases
    haiku_aliases = [
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5",
        "haiku",
    ]

    all_correct = True

    for alias in haiku_aliases:
        pricing = PRICING.get("anthropic", {}).get(alias)

        if not pricing:
            print(f"❌ FAIL: No pricing for {alias}")
            all_correct = False
            continue

        input_price = pricing.get("input")
        output_price = pricing.get("output")

        if input_price == OFFICIAL_HAIKU_PRICING["input"] and output_price == OFFICIAL_HAIKU_PRICING["output"]:
            print(f"✅ {alias:30s} → ${input_price:.2f}/${output_price:.2f} per 1M tokens")
        else:
            print(f"❌ {alias:30s} → ${input_price:.2f}/${output_price:.2f} per 1M tokens (WRONG!)")
            print(f"   Expected: ${OFFICIAL_HAIKU_PRICING['input']:.2f}/${OFFICIAL_HAIKU_PRICING['output']:.2f}")
            all_correct = False

    print("=" * 80)

    if all_correct:
        print("✅ All Haiku pricing correct")
        return True
    else:
        print("❌ Some pricing incorrect - update PRICING dict in cost_tracker.py")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE HAIKU COST TRACKING TEST SUITE")
    print("=" * 80)

    # Test 1: Pricing tables
    print("\n📋 TEST 1: Verify CostTracker pricing tables")
    test1_pass = await test_cost_tracker_pricing_tables()

    # Test 2: End-to-end tracking
    print("\n\n📋 TEST 2: End-to-end cost tracking")
    test2_pass = await test_haiku_cost_tracking_end_to_end()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"TEST 1 (Pricing tables): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"TEST 2 (E2E tracking):   {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print()

    if test1_pass and test2_pass:
        print("🎉 ALL TESTS PASSED - Cost tracking is accurate!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - See details above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
