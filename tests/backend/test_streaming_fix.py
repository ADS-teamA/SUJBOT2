#!/usr/bin/env python3
"""
Test script to verify tool call streaming fix.

This script simulates the agent_adapter behavior to verify that:
1. Tool call notifications are detected correctly
2. tool_call events are emitted immediately
3. Text content is still streamed properly
"""

import re

# Simulate chunks from AgentCore.process_message()
test_chunks = [
    "I'll search for information about water in the reactor.\n",
    "\n",
    "\033[1;34m[Using search...]\033[0m\n",  # Tool call with ANSI codes
    "\n",
    "Based on the search results, I found that...\n",
    "\033[1;34m[Using exact_match_search...]\033[0m\n",  # Another tool call
    "\n",
    "The water in the reactor is demineralized H₂O.\n"
]

def test_streaming_detection():
    """Test that tool calls are detected and handled correctly."""
    print("Testing streaming detection...\n")

    tool_calls_detected = []
    text_chunks_collected = []

    for chunk in test_chunks:
        # Detect tool call notification: [Using TOOL_NAME...]
        tool_call_match = re.search(r'\[Using\s+([a-z_]+)\.{3}\]', chunk)

        if tool_call_match:
            # Extract tool name
            tool_name = tool_call_match.group(1)

            # Simulate tool_call event emission
            tool_calls_detected.append(tool_name)
            print(f"✅ tool_call event: {tool_name}")
        else:
            # Regular text content - strip ANSI color codes
            clean_chunk = re.sub(r'\033\[[0-9;]+m', '', chunk)

            if clean_chunk:  # Only send non-empty chunks
                text_chunks_collected.append(clean_chunk)
                print(f"📝 text_delta: {repr(clean_chunk)}")

    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Tool calls detected: {tool_calls_detected}")
    print(f"Expected: ['search', 'exact_match_search']")
    print(f"✅ PASS" if tool_calls_detected == ['search', 'exact_match_search'] else "❌ FAIL")
    print(f"\nText chunks collected: {len(text_chunks_collected)}")
    print(f"Expected: 6 chunks (excluding tool call lines)")
    print(f"✅ PASS" if len(text_chunks_collected) == 6 else "❌ FAIL")
    print("="*60)

    return tool_calls_detected == ['search', 'exact_match_search'] and len(text_chunks_collected) == 6

if __name__ == "__main__":
    success = test_streaming_detection()
    exit(0 if success else 1)
