#!/usr/bin/env python3
"""
Test script for question decomposition and synthesis
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from claude_sdk_wrapper import ClaudeSDKClient
from question_decomposer import QuestionDecomposer
from answer_synthesizer import AnswerSynthesizer, SubAnswer

async def main():
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("ERROR: CLAUDE_API_KEY not set")
        return

    # Initialize client
    client = ClaudeSDKClient(api_key=api_key)

    # Test question
    question = "Jaké jsou výhody a nevýhody umělé inteligence v medicíně a jak ovlivňuje pacienty?"

    print(f"{'='*80}")
    print(f"TESTING DECOMPOSITION")
    print(f"{'='*80}")
    print(f"\nOriginal question: {question}\n")

    # 1. Decompose
    decomposer = QuestionDecomposer(client, model="claude-3-5-haiku-20241022")
    decomposition = await decomposer.decompose_question(question)

    print(f"Strategy: {decomposition.decomposition_strategy}")
    print(f"Total sub-questions: {decomposition.total_sub_questions}\n")

    for i, sq in enumerate(decomposition.sub_questions, 1):
        print(f"[{i}] {sq.text}")
        print(f"    Type: {sq.type} | Priority: {sq.priority}")

    # 2. Simulate sub-answers
    print(f"\n{'='*80}")
    print(f"SIMULATING SUB-ANSWERS")
    print(f"{'='*80}\n")

    sub_answers = [
        SubAnswer(
            sub_question_id=decomposition.sub_questions[0].id,
            sub_question_text=decomposition.sub_questions[0].text,
            answer="Výhody AI v medicíně zahrnují rychlejší diagnózu, přesnější analýzu snímků a personalizovanou léčbu.",
            confidence=0.85,
            sources=[{"reference": "[1]", "chunk_id": "c1", "score": 0.9}],
            retrieval_score=0.9
        )
    ]

    if len(decomposition.sub_questions) > 1:
        sub_answers.append(SubAnswer(
            sub_question_id=decomposition.sub_questions[1].id,
            sub_question_text=decomposition.sub_questions[1].text,
            answer="Nevýhody zahrnují vysoké náklady, potřebu velkých dat a etické otázky ohledně soukromí.",
            confidence=0.80,
            sources=[{"reference": "[2]", "chunk_id": "c2", "score": 0.85}],
            retrieval_score=0.85
        ))

    for i, sa in enumerate(sub_answers, 1):
        print(f"Sub-answer {i}:")
        print(f"  Q: {sa.sub_question_text}")
        print(f"  A: {sa.answer}")
        print(f"  Confidence: {sa.confidence:.2%}\n")

    # 3. Synthesize
    print(f"{'='*80}")
    print(f"SYNTHESIZING FINAL ANSWER")
    print(f"{'='*80}\n")

    synthesizer = AnswerSynthesizer(client, model="claude-3-5-sonnet-20241022")
    result = await synthesizer.synthesize_answers(question, sub_answers)

    print(f"Final Answer:\n{result.final_answer}\n")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Strategy: {result.synthesis_strategy}")
    print(f"Sources: {len(result.sources)}")

    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())
