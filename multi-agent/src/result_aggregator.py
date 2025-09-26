"""
Result Aggregator - Modul pro agregaci a formátování výsledků
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Třída pro agregaci výsledků od subagentů"""

    def aggregate_results(self, results: List['AnalysisResult']) -> List[Dict[str, Any]]:
        """
        Agreguje výsledky od všech subagentů

        Args:
            results: Seznam výsledků od subagentů

        Returns:
            Agregované výsledky
        """
        aggregated = []

        # Seskupení výsledků podle otázek (pokud by bylo více agentů na otázku)
        results_by_question = defaultdict(list)

        for result in results:
            results_by_question[result.question_id].append(result)

        # Zpracování každé otázky
        for question_id, question_results in results_by_question.items():
            if len(question_results) == 1:
                # Jednoduchý případ - jeden agent na otázku
                single_result = question_results[0]
                aggregated.append(self._format_single_result(single_result))
            else:
                # Více agentů odpovědělo na stejnou otázku - potřeba konsolidace
                aggregated.append(self._consolidate_multiple_results(question_results))

        return aggregated

    def _format_single_result(self, result: 'AnalysisResult') -> Dict[str, Any]:
        """Formátuje jeden výsledek"""
        formatted_sources = []

        for source in result.sources:
            formatted_source = {
                'reference': self._format_source_reference(source),
                'quote': source.get('quote', ''),
                'page': source.get('page', 'nespecifikováno')
            }
            formatted_sources.append(formatted_source)

        return {
            'question_id': result.question_id,
            'question': result.question_text,
            'answer': result.answer,
            'sources': formatted_sources,
            'confidence': result.confidence,
            'metadata': {
                'agent_id': result.agent_id,
                'processing_time': result.processing_time,
                **result.metadata
            }
        }

    def _consolidate_multiple_results(self, results: List['AnalysisResult']) -> Dict[str, Any]:
        """
        Konsoliduje více odpovědí na stejnou otázku

        Strategie:
        1. Vybere odpověď s nejvyšší confidence
        2. Sloučí zdroje ze všech odpovědí
        3. Vypočítá průměrnou confidence
        """
        # Seřazení podle confidence
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
        best_result = sorted_results[0]

        # Sloučení zdrojů
        all_sources = []
        seen_sources = set()

        for result in results:
            for source in result.sources:
                source_key = self._get_source_key(source)
                if source_key not in seen_sources:
                    all_sources.append(source)
                    seen_sources.add(source_key)

        # Výpočet průměrné confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Vytvoření konsolidovaného výsledku
        return {
            'question_id': best_result.question_id,
            'question': best_result.question_text,
            'answer': best_result.answer,
            'sources': [self._format_source_reference(s) for s in all_sources],
            'confidence': avg_confidence,
            'metadata': {
                'consolidation': {
                    'agent_count': len(results),
                    'best_confidence': best_result.confidence,
                    'avg_confidence': avg_confidence,
                    'all_agents': [r.agent_id for r in results]
                },
                'total_processing_time': sum(r.processing_time for r in results)
            }
        }

    def _format_source_reference(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Formátuje referenci na zdroj"""
        section = source.get('section', 'Nespecifikovaná sekce')
        page = source.get('page', '')

        reference = f"{section}"
        if page:
            reference += f" (str. {page})"

        return {
            'reference': reference,
            'quote': source.get('quote', ''),
            'metadata': source.get('metadata', {})
        }

    def _get_source_key(self, source: Dict[str, Any]) -> str:
        """Vytvoří unikátní klíč pro zdroj"""
        return f"{source.get('section', '')}_{source.get('page', '')}_{source.get('quote', '')[:50]}"

    def generate_summary_report(self, aggregated_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generuje souhrnnou zprávu

        Args:
            aggregated_results: Agregované výsledky

        Returns:
            Souhrnná zpráva
        """
        total_questions = len(aggregated_results)
        answered_questions = sum(1 for r in aggregated_results if r['confidence'] > 0.3)
        high_confidence_answers = sum(1 for r in aggregated_results if r['confidence'] > 0.8)

        avg_confidence = sum(r['confidence'] for r in aggregated_results) / total_questions if total_questions > 0 else 0

        total_sources = sum(len(r['sources']) for r in aggregated_results)
        unique_sources = len(set(
            source['reference']
            for result in aggregated_results
            for source in result['sources']
        ))

        summary = {
            'statistics': {
                'total_questions': total_questions,
                'answered_questions': answered_questions,
                'high_confidence_answers': high_confidence_answers,
                'average_confidence': avg_confidence,
                'total_source_references': total_sources,
                'unique_sources': unique_sources
            },
            'confidence_distribution': self._calculate_confidence_distribution(aggregated_results),
            'category_breakdown': self._calculate_category_breakdown(aggregated_results)
        }

        return summary

    def _calculate_confidence_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Vypočítá distribuci confidence skóre"""
        distribution = {
            'velmi_vysoká (>90%)': 0,
            'vysoká (70-90%)': 0,
            'střední (50-70%)': 0,
            'nízká (30-50%)': 0,
            'velmi_nízká (<30%)': 0
        }

        for result in results:
            confidence = result['confidence']
            if confidence > 0.9:
                distribution['velmi_vysoká (>90%)'] += 1
            elif confidence > 0.7:
                distribution['vysoká (70-90%)'] += 1
            elif confidence > 0.5:
                distribution['střední (50-70%)'] += 1
            elif confidence > 0.3:
                distribution['nízká (30-50%)'] += 1
            else:
                distribution['velmi_nízká (<30%)'] += 1

        return distribution

    def _calculate_category_breakdown(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vypočítá rozdělení podle kategorií otázek"""
        categories = defaultdict(lambda: {'count': 0, 'avg_confidence': 0})

        for result in results:
            category = result.get('metadata', {}).get('category', 'general')
            categories[category]['count'] += 1
            categories[category]['avg_confidence'] += result['confidence']

        # Výpočet průměrů
        for category in categories:
            if categories[category]['count'] > 0:
                categories[category]['avg_confidence'] /= categories[category]['count']

        return dict(categories)

    def export_results(self, results: List[Dict[str, Any]], format: str = 'json') -> str:
        """
        Exportuje výsledky do specifikovaného formátu

        Args:
            results: Výsledky k exportu
            format: Formát exportu ('json', 'markdown', 'html')

        Returns:
            Exportovaná data jako string
        """
        if format == 'json':
            return json.dumps(results, ensure_ascii=False, indent=2)

        elif format == 'markdown':
            return self._export_to_markdown(results)

        elif format == 'html':
            return self._export_to_html(results)

        else:
            raise ValueError(f"Nepodporovaný formát exportu: {format}")

    def _export_to_markdown(self, results: List[Dict[str, Any]]) -> str:
        """Exportuje výsledky do Markdown formátu"""
        md_lines = ["# Výsledky analýzy dokumentu\n"]

        for i, result in enumerate(results, 1):
            md_lines.append(f"## Otázka {i}: {result['question']}\n")
            md_lines.append(f"**Odpověď:** {result['answer']}\n")

            if result['sources']:
                md_lines.append("**Zdroje:**")
                for source in result['sources']:
                    md_lines.append(f"- {source['reference']}")
                    if source.get('quote'):
                        md_lines.append(f"  > {source['quote']}")

            md_lines.append(f"\n*Jistota: {result['confidence']:.0%}*\n")
            md_lines.append("---\n")

        return '\n'.join(md_lines)

    def _export_to_html(self, results: List[Dict[str, Any]]) -> str:
        """Exportuje výsledky do HTML formátu"""
        html_template = """
<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <title>Výsledky analýzy dokumentu</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .question {{ background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .answer {{ margin: 10px 0; padding: 10px; background: white; }}
        .sources {{ margin-top: 10px; }}
        .source-item {{ margin: 5px 0; padding: 5px; background: #f9f9f9; }}
        .confidence {{ color: #666; font-style: italic; }}
        .high-confidence {{ color: green; }}
        .medium-confidence {{ color: orange; }}
        .low-confidence {{ color: red; }}
    </style>
</head>
<body>
    <h1>Výsledky analýzy dokumentu</h1>
    {content}
</body>
</html>
        """

        content_lines = []

        for i, result in enumerate(results, 1):
            confidence_class = 'high-confidence' if result['confidence'] > 0.7 else 'medium-confidence' if result['confidence'] > 0.4 else 'low-confidence'

            content_lines.append(f"""
            <div class="question">
                <h2>Otázka {i}: {result['question']}</h2>
                <div class="answer">
                    <strong>Odpověď:</strong> {result['answer']}
                </div>
                <div class="sources">
                    <strong>Zdroje:</strong>
                    <ul>
                        {''.join(f'<li class="source-item">{s["reference"]}</li>' for s in result['sources'])}
                    </ul>
                </div>
                <div class="confidence {confidence_class}">
                    Jistota: {result['confidence']:.0%}
                </div>
            </div>
            """)

        return html_template.format(content=' '.join(content_lines))