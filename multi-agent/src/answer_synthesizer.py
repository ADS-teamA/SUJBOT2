"""
Answer Synthesizer - Syntéza dílčích odpovědí do komplexní finální odpovědi
"""

import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import json

from claude_sdk_wrapper import ClaudeSDKClient, ClaudeCodeOptions

logger = logging.getLogger(__name__)


@dataclass
class SubAnswer:
    """Reprezentace dílčí odpovědi"""
    sub_question_id: str
    sub_question_text: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    retrieval_score: float


@dataclass
class SynthesizedAnswer:
    """Výsledek syntézy odpovědí"""
    original_question: str
    final_answer: str
    confidence: float
    sources: List[Dict[str, Any]]  # Deduplikované zdroje
    sub_answers_count: int
    synthesis_strategy: str  # merge, prioritize, consensus
    metadata: Dict[str, Any]


class AnswerSynthesizer:
    """Syntéza dílčích odpovědí pomocí Claude Sonnet/Haiku"""

    def __init__(self, claude_client: ClaudeSDKClient, model: str = "claude-3-5-sonnet-20241022"):
        """
        Inicializace answer synthesizer

        Args:
            claude_client: Instance Claude SDK klienta
            model: Model pro syntézu (default: Sonnet pro lepší kvalitu)
        """
        self.claude_client = claude_client
        self.model = model
        logger.info(f"AnswerSynthesizer initialized with model: {model}")

    async def synthesize_answers(
        self,
        original_question: str,
        sub_answers: List[SubAnswer],
        synthesis_strategy: str = "auto"
    ) -> SynthesizedAnswer:
        """
        Syntetizuje dílčí odpovědi do komplexní finální odpovědi

        Args:
            original_question: Původní uživatelská otázka
            sub_answers: Seznam dílčích odpovědí
            synthesis_strategy: Strategie syntézy (auto, merge, prioritize, consensus)

        Returns:
            SynthesizedAnswer s finální odpovědí
        """
        logger.info(f"Synthesizing {len(sub_answers)} sub-answers for: {original_question}")

        if not sub_answers:
            logger.warning("No sub-answers to synthesize")
            return self._create_empty_answer(original_question)

        # Pokud je jen jedna podotázka, vrátíme její odpověď přímo
        if len(sub_answers) == 1:
            logger.info("Single sub-answer, returning directly")
            return self._create_single_answer(original_question, sub_answers[0])

        # Automatická detekce strategie
        if synthesis_strategy == "auto":
            synthesis_strategy = self._determine_synthesis_strategy(sub_answers)
            logger.info(f"Auto-detected synthesis strategy: {synthesis_strategy}")

        # Příprava promptu pro syntézu
        prompt = self._prepare_synthesis_prompt(
            original_question,
            sub_answers,
            synthesis_strategy
        )

        try:
            # Volání Claude modelu
            options = ClaudeCodeOptions(
                model=self.model,
                max_tokens=3000,
                temperature=0.3
            )

            response = await self.claude_client.query(prompt, options)

            # Parsování odpovědi
            final_answer = self._parse_synthesis_response(response)

            # Deduplikace a agregace zdrojů
            all_sources = self._aggregate_sources(sub_answers)

            # Výpočet finální confidence
            final_confidence = self._calculate_final_confidence(sub_answers)

            result = SynthesizedAnswer(
                original_question=original_question,
                final_answer=final_answer,
                confidence=final_confidence,
                sources=all_sources,
                sub_answers_count=len(sub_answers),
                synthesis_strategy=synthesis_strategy,
                metadata={
                    "model_used": self.model,
                    "sub_answer_confidences": [sa.confidence for sa in sub_answers],
                    "avg_retrieval_score": sum(sa.retrieval_score for sa in sub_answers) / len(sub_answers)
                }
            )

            logger.info(f"Synthesis complete. Final confidence: {final_confidence:.3f}")
            return result

        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            # Fallback: sloučíme odpovědi jednoduchým způsobem
            return self._fallback_synthesis(original_question, sub_answers)

    def _determine_synthesis_strategy(self, sub_answers: List[SubAnswer]) -> str:
        """Určí optimální strategii syntézy na základě sub-answers"""
        # Pokud mají podotázky velmi rozdílné confidence, prioritize
        confidences = [sa.confidence for sa in sub_answers]
        max_conf = max(confidences)
        min_conf = min(confidences)

        if max_conf - min_conf > 0.4:
            return "prioritize"

        # Pokud jsou odpovědi krátké a faktické, merge
        avg_length = sum(len(sa.answer) for sa in sub_answers) / len(sub_answers)
        if avg_length < 200:
            return "merge"

        # Jinak consensus
        return "consensus"

    def _prepare_synthesis_prompt(
        self,
        original_question: str,
        sub_answers: List[SubAnswer],
        strategy: str
    ) -> str:
        """Připraví prompt pro syntézu odpovědí"""

        # Formátování dílčích odpovědí
        sub_answers_text = ""
        for i, sa in enumerate(sub_answers, 1):
            sub_answers_text += f"\n--- Dílčí odpověď {i} ---\n"
            sub_answers_text += f"Podotázka: {sa.sub_question_text}\n"
            sub_answers_text += f"Odpověď: {sa.answer}\n"
            sub_answers_text += f"Confidence: {sa.confidence:.2%}\n"
            if sa.sources:
                sources_preview = ", ".join([s['reference'] for s in sa.sources[:3]])
                sub_answers_text += f"Zdroje: {sources_preview}\n"

        strategy_instructions = {
            "merge": "Slouč všechny dílčí odpovědi do jedné soudržné odpovědi, která pokrývá všechny aspekty.",
            "prioritize": "Upřednostni odpovědi s vyšší confidence a vytvoř odpověď zaměřenou na nejspolehlivější informace.",
            "consensus": "Najdi společné informace napříč odpověďmi a vytvoř vyvážený konsensus."
        }

        instruction = strategy_instructions.get(strategy, strategy_instructions["consensus"])

        return f"""Tvým úkolem je syntetizovat několik dílčích odpovědí do jedné komplexní, přesné a soudržné finální odpovědi.

PŮVODNÍ OTÁZKA:
{original_question}

DÍLČÍ ODPOVĚDI:
{sub_answers_text}

STRATEGIE SYNTÉZY: {strategy}
{instruction}

INSTRUKCE:
1. Vytvoř jednu komplexní odpověď na původní otázku
2. Integruj informace ze všech relevantních dílčích odpovědí
3. Zachovej citace ze zdrojů [1], [2], atd.
4. Odstraň redundance a duplicity
5. Odpověď by měla být:
   - Přesná a komplexní
   - Přímo odpovídající na původní otázku
   - Podložená zdroji z dokumentu
   - Srozumitelná a strukturovaná

Vrať pouze finální odpověď, bez dodatečného úvodu nebo komentářů."""

    def _parse_synthesis_response(self, response: str) -> str:
        """Parsuje odpověď od Claude a extrahuje finální odpověď"""
        # Claude vrací přímo odpověď
        return response.strip()

    def _aggregate_sources(self, sub_answers: List[SubAnswer]) -> List[Dict[str, Any]]:
        """Agreguje a deduplikuje zdroje ze všech dílčích odpovědí"""
        seen_sources = {}

        for sa in sub_answers:
            for source in sa.sources:
                source_key = source.get('chunk_id', source.get('reference', ''))

                if source_key not in seen_sources:
                    seen_sources[source_key] = source
                else:
                    # Pokud už zdroj existuje, zvýš jeho skóre (indikátor relevance)
                    existing = seen_sources[source_key]
                    if 'score' in existing and 'score' in source:
                        existing['score'] = max(existing['score'], source['score'])

        # Seřazení podle skóre
        sources_list = list(seen_sources.values())
        sources_list.sort(key=lambda x: x.get('score', 0), reverse=True)

        return sources_list

    def _calculate_final_confidence(self, sub_answers: List[SubAnswer]) -> float:
        """Vypočítá finální confidence na základě dílčích odpovědí"""
        if not sub_answers:
            return 0.0

        # Vážený průměr podle retrieval scores
        total_weight = sum(sa.retrieval_score for sa in sub_answers)

        if total_weight == 0:
            # Pokud nejsou retrieval scores, prostý průměr
            return sum(sa.confidence for sa in sub_answers) / len(sub_answers)

        weighted_confidence = sum(
            sa.confidence * sa.retrieval_score
            for sa in sub_answers
        ) / total_weight

        # Penalizace pokud je velký rozptyl v confidences (indikátor nesouladu)
        confidences = [sa.confidence for sa in sub_answers]
        std_dev = (sum((c - weighted_confidence) ** 2 for c in confidences) / len(confidences)) ** 0.5

        penalty = min(0.2, std_dev)  # Max 20% penalty
        final_confidence = max(0.0, weighted_confidence - penalty)

        return round(final_confidence, 3)

    def _create_empty_answer(self, original_question: str) -> SynthesizedAnswer:
        """Vytvoří prázdnou odpověď když nejsou sub-answers"""
        return SynthesizedAnswer(
            original_question=original_question,
            final_answer="Nepodařilo se najít odpověď na tuto otázku v dokumentu.",
            confidence=0.0,
            sources=[],
            sub_answers_count=0,
            synthesis_strategy="empty",
            metadata={}
        )

    def _create_single_answer(self, original_question: str, sub_answer: SubAnswer) -> SynthesizedAnswer:
        """Vytvoří finální odpověď z jedné sub-answer"""
        return SynthesizedAnswer(
            original_question=original_question,
            final_answer=sub_answer.answer,
            confidence=sub_answer.confidence,
            sources=sub_answer.sources,
            sub_answers_count=1,
            synthesis_strategy="single",
            metadata={
                "sub_question": sub_answer.sub_question_text
            }
        )

    def _fallback_synthesis(self, original_question: str, sub_answers: List[SubAnswer]) -> SynthesizedAnswer:
        """Fallback syntéza pokud selže Claude"""
        logger.warning("Using fallback synthesis")

        # Prostě spojíme odpovědi s nejvyšší confidence
        sorted_answers = sorted(sub_answers, key=lambda x: x.confidence, reverse=True)

        combined_answer = "\n\n".join([
            f"{sa.answer}"
            for sa in sorted_answers[:3]  # Top 3 odpovědi
        ])

        return SynthesizedAnswer(
            original_question=original_question,
            final_answer=combined_answer,
            confidence=self._calculate_final_confidence(sub_answers),
            sources=self._aggregate_sources(sub_answers),
            sub_answers_count=len(sub_answers),
            synthesis_strategy="fallback",
            metadata={}
        )

    async def synthesize_multiple_questions(
        self,
        questions_and_answers: List[tuple[str, List[SubAnswer]]]
    ) -> List[SynthesizedAnswer]:
        """
        Syntetizuje odpovědi pro více otázek paralelně

        Args:
            questions_and_answers: Seznam tuples (otázka, sub_answers)

        Returns:
            Seznam SynthesizedAnswer
        """
        tasks = [
            self.synthesize_answers(question, sub_answers)
            for question, sub_answers in questions_and_answers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrování chyb
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error synthesizing question {i}: {result}")
                # Vytvoř fallback odpověď
                question = questions_and_answers[i][0]
                valid_results.append(self._create_empty_answer(question))
            else:
                valid_results.append(result)

        return valid_results


# CLI pro testování
async def test_synthesizer():
    """Test funkce synthesizer"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")

    if not api_key:
        print("ERROR: CLAUDE_API_KEY not set")
        return

    client = ClaudeSDKClient(api_key=api_key)
    synthesizer = AnswerSynthesizer(client)

    # Test data
    original_question = "Jaké jsou výhody a nevýhody umělé inteligence v medicíně?"

    sub_answers = [
        SubAnswer(
            sub_question_id="q1_s1",
            sub_question_text="Jaké jsou výhody umělé inteligence v medicíně?",
            answer="Umělá inteligence v medicíně přináší rychlejší diagnózu, přesnější analýzu lékařských snímků a personalizovanou léčbu. [1]",
            confidence=0.85,
            sources=[{"reference": "[1]", "chunk_id": "c1", "score": 0.9}],
            retrieval_score=0.9
        ),
        SubAnswer(
            sub_question_id="q1_s2",
            sub_question_text="Jaké jsou nevýhody umělé inteligence v medicíně?",
            answer="Mezi nevýhody patří vysoké náklady na implementaci, potřeba velkého množství dat a etické otázky ohledně soukromí pacientů. [2]",
            confidence=0.80,
            sources=[{"reference": "[2]", "chunk_id": "c2", "score": 0.85}],
            retrieval_score=0.85
        ),
        SubAnswer(
            sub_question_id="q1_s3",
            sub_question_text="Jak AI ovlivňuje pacienty?",
            answer="AI zlepšuje dostupnost zdravotní péče a může snížit chyby, ale pacienti mohou mít obavy z nedostatku lidského kontaktu. [3]",
            confidence=0.75,
            sources=[{"reference": "[3]", "chunk_id": "c3", "score": 0.80}],
            retrieval_score=0.80
        )
    ]

    print(f"\nORIGINAL QUESTION: {original_question}")
    print(f"\nSUB-ANSWERS: {len(sub_answers)}")

    result = await synthesizer.synthesize_answers(original_question, sub_answers)

    print(f"\n{'='*80}")
    print("SYNTHESIZED ANSWER:")
    print('='*80)
    print(result.final_answer)
    print(f"\nConfidence: {result.confidence:.2%}")
    print(f"Strategy: {result.synthesis_strategy}")
    print(f"Sources: {len(result.sources)}")


if __name__ == "__main__":
    asyncio.run(test_synthesizer())
