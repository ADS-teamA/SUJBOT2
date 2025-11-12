"""
UNIT TEST: Cost Calculation Logic

This test verifies cost calculation without requiring API calls.
Tests the core pricing logic for Haiku models.

Author: Claude Code (2025-11-12)
"""

import pytest
from src.cost_tracker import CostTracker, PRICING


def test_haiku_pricing_constants():
    """Verify Haiku pricing constants match official pricing."""
    # Official Haiku 4.5 pricing (November 2025)
    EXPECTED_INPUT = 1.00
    EXPECTED_OUTPUT = 5.00

    haiku_aliases = [
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5",
        "haiku",
    ]

    for alias in haiku_aliases:
        pricing = PRICING["anthropic"].get(alias)
        assert pricing is not None, f"No pricing for {alias}"
        assert pricing["input"] == EXPECTED_INPUT, f"{alias}: input price mismatch"
        assert pricing["output"] == EXPECTED_OUTPUT, f"{alias}: output price mismatch"

    print("✅ All Haiku pricing constants correct")


def test_haiku_cost_calculation_simple():
    """Test basic cost calculation for Haiku model."""
    tracker = CostTracker()

    # Example: 1000 input tokens, 500 output tokens
    # Expected: $0.001 + $0.0025 = $0.0035
    cost = tracker.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=1000,
        output_tokens=500,
        operation="test"
    )

    expected_cost = (1000 * 1.00 / 1_000_000) + (500 * 5.00 / 1_000_000)
    assert abs(cost - expected_cost) < 1e-9, f"Cost mismatch: {cost} != {expected_cost}"

    print(f"✅ Simple cost calculation: ${cost:.6f} (expected ${expected_cost:.6f})")


def test_haiku_vs_sonnet_cost_difference():
    """Verify Haiku is cheaper than Sonnet for same token count."""
    haiku_tracker = CostTracker()
    sonnet_tracker = CostTracker()

    tokens_in = 10000
    tokens_out = 5000

    # Haiku: $1/$5 per 1M tokens
    haiku_cost = haiku_tracker.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        operation="test"
    )

    # Sonnet: $3/$15 per 1M tokens
    sonnet_cost = sonnet_tracker.track_llm(
        provider="anthropic",
        model="sonnet",
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        operation="test"
    )

    # Haiku should be 3x cheaper
    assert sonnet_cost > haiku_cost, "Sonnet should be more expensive than Haiku"
    ratio = sonnet_cost / haiku_cost
    assert abs(ratio - 3.0) < 0.01, f"Cost ratio should be ~3.0, got {ratio}"

    print(f"✅ Haiku: ${haiku_cost:.6f}, Sonnet: ${sonnet_cost:.6f}, Ratio: {ratio:.2f}x")


def test_output_vs_input_cost_difference():
    """Verify output tokens are 5x more expensive than input for Haiku."""
    tracker = CostTracker()

    # Case 1: 1000 input tokens only
    input_only_cost = tracker.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=1000,
        output_tokens=0,
        operation="test"
    )

    # Case 2: 1000 output tokens only
    tracker_output = CostTracker()
    output_only_cost = tracker_output.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=0,
        output_tokens=1000,
        operation="test"
    )

    # Output should be 5x more expensive
    ratio = output_only_cost / input_only_cost
    assert abs(ratio - 5.0) < 0.01, f"Output/input ratio should be ~5.0, got {ratio}"

    print(f"✅ Input only: ${input_only_cost:.6f}, Output only: ${output_only_cost:.6f}, Ratio: {ratio:.2f}x")


def test_cache_discount():
    """Test prompt caching discount (90% off input price)."""
    tracker = CostTracker()

    # Example: 10000 input tokens, 5000 cache read tokens
    # Cache reads should be 10% of input price
    cost = tracker.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=10000,
        output_tokens=5000,
        cache_read_tokens=5000,  # Additional cache hits
        operation="test"
    )

    # Manual calculation
    input_cost = 10000 * 1.00 / 1_000_000
    output_cost = 5000 * 5.00 / 1_000_000
    cache_cost = 5000 * 0.10 / 1_000_000  # 90% discount
    expected_total = input_cost + output_cost + cache_cost

    assert abs(cost - expected_total) < 1e-9, f"Cache cost mismatch: {cost} != {expected_total}"

    # Cache should save money
    no_cache_cost = (10000 + 5000) * 1.00 / 1_000_000 + output_cost
    savings = no_cache_cost - cost
    savings_percent = (savings / no_cache_cost) * 100

    print(f"✅ Cache discount test:")
    print(f"   With cache: ${cost:.6f}")
    print(f"   Without cache: ${no_cache_cost:.6f}")
    print(f"   Savings: ${savings:.6f} ({savings_percent:.1f}%)")


def test_large_conversation_cost():
    """Test cumulative cost tracking across multiple LLM calls."""
    tracker = CostTracker()

    # Simulate a conversation with 5 back-and-forth exchanges
    exchanges = [
        (1000, 500),   # User question, short answer
        (2000, 1000),  # Follow-up, longer answer
        (1500, 800),   # Clarification
        (3000, 2000),  # Complex query, detailed answer
        (1000, 500),   # Final question
    ]

    total_expected = 0.0

    for i, (input_tokens, output_tokens) in enumerate(exchanges, 1):
        cost = tracker.track_llm(
            provider="anthropic",
            model="haiku",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=f"exchange_{i}"
        )

        expected = (input_tokens * 1.00 / 1_000_000) + (output_tokens * 5.00 / 1_000_000)
        total_expected += expected

    final_cost = tracker.get_total_cost()

    assert abs(final_cost - total_expected) < 1e-9, f"Cumulative cost mismatch: {final_cost} != {total_expected}"

    print(f"✅ Conversation cost (5 exchanges):")
    print(f"   Total: ${final_cost:.6f}")
    print(f"   Per exchange: ${final_cost / 5:.6f} avg")


def test_token_accuracy_thousands():
    """Test cost calculation for various token counts."""
    test_cases = [
        # (input, output, expected_cost)
        (1000, 1000, 0.001 + 0.005),      # $0.006
        (5000, 2000, 0.005 + 0.010),      # $0.015
        (10000, 5000, 0.010 + 0.025),     # $0.035
        (50000, 10000, 0.050 + 0.050),    # $0.100
        (100000, 50000, 0.100 + 0.250),   # $0.350
    ]

    for input_tokens, output_tokens, expected in test_cases:
        tracker = CostTracker()
        cost = tracker.track_llm(
            provider="anthropic",
            model="haiku",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation="test"
        )

        assert abs(cost - expected) < 1e-9, f"Cost mismatch for {input_tokens}/{output_tokens}: {cost} != {expected}"

    print(f"✅ All {len(test_cases)} token count test cases passed")


def test_zero_tokens():
    """Test edge case: zero tokens."""
    tracker = CostTracker()

    cost = tracker.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=0,
        output_tokens=0,
        operation="test"
    )

    assert cost == 0.0, f"Zero tokens should cost $0, got ${cost}"

    print("✅ Zero tokens edge case handled correctly")


def test_input_only_vs_output_only():
    """Compare pure input vs pure output costs."""
    # Same number of tokens, different types
    TOKENS = 10000

    tracker_input = CostTracker()
    input_cost = tracker_input.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=TOKENS,
        output_tokens=0,
        operation="input_only"
    )

    tracker_output = CostTracker()
    output_cost = tracker_output.track_llm(
        provider="anthropic",
        model="haiku",
        input_tokens=0,
        output_tokens=TOKENS,
        operation="output_only"
    )

    # Expected
    expected_input = TOKENS * 1.00 / 1_000_000
    expected_output = TOKENS * 5.00 / 1_000_000

    assert abs(input_cost - expected_input) < 1e-9, "Input cost mismatch"
    assert abs(output_cost - expected_output) < 1e-9, "Output cost mismatch"
    assert output_cost == 5 * input_cost, "Output should be exactly 5x input cost"

    print(f"✅ Pure comparison ({TOKENS:,} tokens each):")
    print(f"   Input only:  ${input_cost:.6f}")
    print(f"   Output only: ${output_cost:.6f}")
    print(f"   Ratio: {output_cost / input_cost:.1f}x")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HAIKU COST CALCULATION UNIT TESTS")
    print("=" * 80)

    tests = [
        test_haiku_pricing_constants,
        test_haiku_cost_calculation_simple,
        test_haiku_vs_sonnet_cost_difference,
        test_output_vs_input_cost_difference,
        test_cache_discount,
        test_large_conversation_cost,
        test_token_accuracy_thousands,
        test_zero_tokens,
        test_input_only_vs_output_only,
    ]

    passed = 0
    failed = 0

    for i, test_func in enumerate(tests, 1):
        print(f"\n📋 TEST {i}/{len(tests)}: {test_func.__name__}")
        print("-" * 80)
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL UNIT TESTS PASSED!")
        exit(0)
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        exit(1)
