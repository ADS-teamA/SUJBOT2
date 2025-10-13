"""
Question Decomposer - Rozklad komplexních otázek na menší podotázky pomocí Claude Haiku
"""

import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import re

from claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions

logger = logging.getLogger(__name__)


@dataclass
class SubQuestion:
    """Reprezentace podotázky"""
    id: str
    text: str
    type: str  # factual, analytical, comparison, temporal
    parent_question_id: str
    priority: int  # 1 = highest priority
    keywords: List[str]


@dataclass
class QuestionDecompositionResult:
    """Výsledek dekompozice otázky"""
    original_question: str
    original_question_id: str
    sub_questions: List[SubQuestion]
    decomposition_strategy: str  # simple, hierarchical, aspect-based
    total_sub_questions: int


class QuestionDecomposer:
    """Rozklad komplexních otázek pomocí Claude Haiku"""

    def __init__(self, claude_client: ClaudeSDKClient, model: str = "claude-3-5-haiku-20241022"):
        """
        Inicializace question decomposer

        Args:
            claude_client: Instance Claude SDK klienta
            model: Model pro dekompozici (default: Haiku)
        """
        self.claude_client = claude_client
        self.model = model
        logger.info(f"QuestionDecomposer initialized with model: {model}")

    async def decompose_question(self, question: str, question_id: str = None, max_sub_questions: int = 5) -> QuestionDecompositionResult:
        """
        Rozloží komplexní otázku na menší podotázky

        Args:
            question: Původní otázka
            question_id: ID otázky (pokud není zadáno, vygeneruje se)
            max_sub_questions: Maximum podotázek (default: 5)

        Returns:
            QuestionDecompositionResult s podotázkami
        """
        if not question_id:
            question_id = f"q_{hash(question) % 10000}"

        logger.info(f"Decomposing question [{question_id}]: {question}")

        # Detekce zda je otázka dostatečně komplexní pro dekompozici
        if not self._is_complex_question(question):
            logger.info("Question is simple, skipping decomposition")
            # Pro jednoduchou otázku vrátíme ji samotnou jako jediný sub-query
            return QuestionDecompositionResult(
                original_question=question,
                original_question_id=question_id,
                sub_questions=[SubQuestion(
                    id=f"{question_id}_s1",
                    text=question,
                    type="factual",
                    parent_question_id=question_id,
                    priority=1,
                    keywords=self._extract_keywords(question)
                )],
                decomposition_strategy="simple",
                total_sub_questions=1
            )

        # Příprava promptu pro Haiku
        prompt = self._prepare_decomposition_prompt(question, max_sub_questions)

        try:
            # Volání Haiku modelu
            options = ClaudeCodeOptions(
                model=self.model,
                max_tokens=1500,
                temperature=0.3
            )

            response = await self.claude_client.query(prompt, options)

            # Parsování odpovědi
            sub_questions = self._parse_decomposition_response(response, question_id)

            if not sub_questions:
                logger.warning("No sub-questions generated, falling back to original question")
                sub_questions = [SubQuestion(
                    id=f"{question_id}_s1",
                    text=question,
                    type="factual",
                    parent_question_id=question_id,
                    priority=1,
                    keywords=self._extract_keywords(question)
                )]

            # Limitování počtu podotázek
            if len(sub_questions) > max_sub_questions:
                logger.info(f"Limiting sub-questions from {len(sub_questions)} to {max_sub_questions}")
                sub_questions = sub_questions[:max_sub_questions]

            decomposition_strategy = self._determine_strategy(sub_questions)

            result = QuestionDecompositionResult(
                original_question=question,
                original_question_id=question_id,
                sub_questions=sub_questions,
                decomposition_strategy=decomposition_strategy,
                total_sub_questions=len(sub_questions)
            )

            logger.info(f"Decomposed into {len(sub_questions)} sub-questions using {decomposition_strategy} strategy")
            return result

        except Exception as e:
            logger.error(f"Error during decomposition: {e}")
            # Fallback: vrátíme původní otázku
            return QuestionDecompositionResult(
                original_question=question,
                original_question_id=question_id,
                sub_questions=[SubQuestion(
                    id=f"{question_id}_s1",
                    text=question,
                    type="factual",
                    parent_question_id=question_id,
                    priority=1,
                    keywords=self._extract_keywords(question)
                )],
                decomposition_strategy="fallback",
                total_sub_questions=1
            )

    def _is_complex_question(self, question: str) -> bool:
        """Detekuje zda je otázka dostatečně komplexní pro dekompozici"""
        # Heuristiky pro komplexnost
        word_count = len(question.split())

        # Komplexní indikátory
        complex_indicators = [
            'compare', 'analyze', 'evaluate', 'describe', 'explain',
            'how does', 'why does', 'what are the differences',
            'impact', 'effect', 'relationship', 'connection',
            'porovnej', 'analyzuj', 'vyhodnoť', 'popiš', 'vysvětli',
            'jaký je rozdíl', 'jaké jsou rozdíly', 'dopad', 'vliv', 'vztah'
        ]

        has_complex_indicator = any(indicator in question.lower() for indicator in complex_indicators)

        # Komplexní otázka pokud:
        # - Má více než 10 slov NEBO
        # - Obsahuje komplexní indikátor
        return word_count > 10 or has_complex_indicator

    def _prepare_decomposition_prompt(self, question: str, max_sub_questions: int) -> str:
        """Připraví prompt pro dekompozici otázky"""
        return f"""Máš za úkol rozložit komplexní otázku na menší, konkrétní podotázky, které pomohou lépe vyhledat informace v dokumentu.

PŮVODNÍ OTÁZKA:
{question}

INSTRUKCE:
1. Rozlož tuto otázku na {max_sub_questions} menších, specifických podotázek
2. Každá podotázka by měla být:
   - Konkrétní a zaměřená na jeden aspekt
   - Formulovaná jako samostatná otázka
   - Vhodná pro vyhledávání v dokumentu pomocí embeddings
3. Podotázky by měly pokrýt různé aspekty původní otázky
4. Prioritizuj podotázky od nejdůležitějších

FORMÁT ODPOVĚDI (JSON):
{{{{
  "sub_questions": [
    {{{{
      "text": "první podotázka",
      "type": "factual|analytical|comparison|temporal",
      "priority": 1,
      "keywords": ["klíčové", "slovo"]
    }}}},
    ...
  ],
  "strategy": "hierarchical|aspect-based|sequential"
}}}}

Vrať pouze JSON, bez dalšího textu."""

    def _parse_decomposition_response(self, response: str, parent_question_id: str) -> List[SubQuestion]:
        """Parsuje odpověď od Haiku a vytvoří SubQuestion objekty"""
        try:
            # Extrakce JSON z odpovědi
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in decomposition response")
                return []

            data = json.loads(json_match.group())

            sub_questions = []
            for i, sq_data in enumerate(data.get('sub_questions', []), 1):
                sub_question = SubQuestion(
                    id=f"{parent_question_id}_s{i}",
                    text=sq_data.get('text', ''),
                    type=sq_data.get('type', 'factual'),
                    parent_question_id=parent_question_id,
                    priority=sq_data.get('priority', i),
                    keywords=sq_data.get('keywords', [])
                )
                sub_questions.append(sub_question)

            return sub_questions

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Response was: {response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing decomposition response: {e}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrahuje klíčová slova z textu"""
        # Jednoduchá extrakce - odstraní stop words
        stop_words = {'co', 'je', 'jsou', 'jak', 'kdy', 'kde', 'proč', 'kdo',
                     'what', 'is', 'are', 'how', 'when', 'where', 'why', 'who',
                     'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return list(set(keywords))[:10]  # Max 10 keywords

    def _determine_strategy(self, sub_questions: List[SubQuestion]) -> str:
        """Určí jakou strategii dekompozice byla použita"""
        if len(sub_questions) == 1:
            return "simple"

        types = [sq.type for sq in sub_questions]

        # Hierarchical: pokud obsahuje factual + analytical
        if 'factual' in types and 'analytical' in types:
            return "hierarchical"

        # Aspect-based: pokud různé typy
        if len(set(types)) > 2:
            return "aspect-based"

        return "sequential"

    async def decompose_multiple_questions(self, questions: List[str]) -> List[QuestionDecompositionResult]:
        """
        Rozloží více otázek najednou (paralelně)

        Args:
            questions: Seznam otázek k rozložení

        Returns:
            Seznam QuestionDecompositionResult
        """
        tasks = [
            self.decompose_question(q, f"q_{i}")
            for i, q in enumerate(questions, 1)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrování chyb
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error decomposing question {i}: {result}")
            else:
                valid_results.append(result)

        return valid_results


# CLI pro testování
async def test_decomposer():
    """Test funkce decomposer"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")

    if not api_key:
        print("ERROR: CLAUDE_API_KEY not set")
        return

    client = ClaudeSDKClient(api_key=api_key)
    decomposer = QuestionDecomposer(client)

    # Test otázky
    test_questions = [
        "Co je to umělá inteligence?",  # Jednoduchá - neměla by se rozkládat
        "Porovnej výhody a nevýhody používání umělé inteligence v medicíně a analyzuj její dopad na pacienty.",  # Komplexní
    ]

    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"ORIGINAL QUESTION: {question}")
        print('='*80)

        result = await decomposer.decompose_question(question)

        print(f"\nStrategy: {result.decomposition_strategy}")
        print(f"Total sub-questions: {result.total_sub_questions}\n")

        for sq in result.sub_questions:
            print(f"[{sq.priority}] {sq.text}")
            print(f"    Type: {sq.type} | Keywords: {', '.join(sq.keywords[:5])}")
            print()


if __name__ == "__main__":
    asyncio.run(test_decomposer())
