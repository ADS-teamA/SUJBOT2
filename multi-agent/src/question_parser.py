"""
Question Parser - Modul pro parsování a zpracování otázek
"""

import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any
import re
import hashlib
import logging

logger = logging.getLogger(__name__)


class QuestionParser:
    """Třída pro parsování otázek z různých vstupních formátů"""

    async def parse_questions(self, input_source: str) -> List[Dict[str, str]]:
        """
        Parsuje otázky ze vstupního zdroje

        Args:
            input_source: String s otázkami nebo cesta k souboru

        Returns:
            Seznam slovníků s otázkami
        """
        questions = []

        # Detekce, zda je vstup soubor nebo přímý text
        path = Path(input_source)

        if path.exists() and path.is_file():
            logger.info(f"Načítání otázek ze souboru: {input_source}")
            questions = await self._parse_from_file(path)
        else:
            logger.info("Parsování otázek z přímého textu")
            questions = await self._parse_from_text(input_source)

        # Přidání ID ke každé otázce
        for i, question in enumerate(questions):
            if 'id' not in question:
                # Vytvoření unikátního ID na základě obsahu otázky
                question_hash = hashlib.md5(question['text'].encode()).hexdigest()[:8]
                question['id'] = f"q_{i+1}_{question_hash}"

        logger.info(f"Naparsováno {len(questions)} otázek")
        return questions

    async def _parse_from_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parsuje otázky ze souboru"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        if file_path.suffix.lower() == '.md':
            return await self._parse_markdown_questions(content)
        else:
            return await self._parse_from_text(content)

    async def _parse_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Parsuje otázky z prostého textu

        Podporované formáty:
        - Jedna otázka na řádek
        - Číslované otázky (1. otázka, 2. otázka)
        - Otázky s odrážkami (- otázka, * otázka)
        - Otázky oddělené prázdnými řádky
        """
        questions = []
        lines = text.strip().split('\n')

        # Detekce formátu
        numbered_pattern = re.compile(r'^\d+[\.\)]\s+(.+)')
        bullet_pattern = re.compile(r'^[\-\*\+]\s+(.+)')
        question_mark_pattern = re.compile(r'^.+\?$')

        current_question = []

        for line in lines:
            line = line.strip()

            if not line:
                # Prázdný řádek - konec otázky
                if current_question:
                    question_text = ' '.join(current_question)
                    if question_text:
                        questions.append({
                            'text': question_text,
                            'type': 'general'
                        })
                    current_question = []
                continue

            # Kontrola číslovaných otázek
            numbered_match = numbered_pattern.match(line)
            if numbered_match:
                if current_question:
                    questions.append({
                        'text': ' '.join(current_question),
                        'type': 'general'
                    })
                    current_question = []
                current_question.append(numbered_match.group(1))
                continue

            # Kontrola odrážek
            bullet_match = bullet_pattern.match(line)
            if bullet_match:
                if current_question:
                    questions.append({
                        'text': ' '.join(current_question),
                        'type': 'general'
                    })
                    current_question = []
                current_question.append(bullet_match.group(1))
                continue

            # Kontrola otázek končících otazníkem
            if question_mark_pattern.match(line):
                if current_question:
                    current_question.append(line)
                    questions.append({
                        'text': ' '.join(current_question),
                        'type': 'general'
                    })
                    current_question = []
                else:
                    questions.append({
                        'text': line,
                        'type': 'general'
                    })
                continue

            # Přidání do aktuální otázky
            current_question.append(line)

        # Zpracování poslední otázky
        if current_question:
            questions.append({
                'text': ' '.join(current_question),
                'type': 'general'
            })

        return questions

    async def _parse_markdown_questions(self, content: str) -> List[Dict[str, str]]:
        """
        Parsuje otázky z Markdown formátu

        Podporuje:
        - Sekce s nadpisy (# Otázky, ## Kategorie)
        - Seznam otázek
        - Metadata v YAML front matter
        """
        questions = []
        lines = content.split('\n')

        current_category = 'general'
        current_priority = 'normal'
        in_questions_section = False

        # Parsování YAML front matter (pokud existuje)
        if lines[0].strip() == '---':
            yaml_end = lines[1:].index('---') + 1
            # Zde by se mohl parsovat YAML pro metadata
            lines = lines[yaml_end + 1:]

        for line in lines:
            line = line.strip()

            # Detekce nadpisů
            if line.startswith('#'):
                heading_level = len(line.split()[0])
                heading_text = line[heading_level:].strip()

                if 'otázk' in heading_text.lower() or 'question' in heading_text.lower():
                    in_questions_section = True
                    continue

                if in_questions_section and heading_level >= 2:
                    current_category = heading_text
                    continue

            # Parsování otázek
            if in_questions_section:
                if line.startswith('- ') or line.startswith('* '):
                    question_text = line[2:].strip()

                    # Detekce priority (např. [HIGH], [LOW])
                    priority_match = re.match(r'^\[(\w+)\]\s*(.+)', question_text)
                    if priority_match:
                        current_priority = priority_match.group(1).lower()
                        question_text = priority_match.group(2)

                    if question_text:
                        questions.append({
                            'text': question_text,
                            'type': 'detailed',
                            'category': current_category,
                            'priority': current_priority
                        })

                elif re.match(r'^\d+\.', line):
                    question_text = re.sub(r'^\d+\.\s*', '', line)
                    if question_text:
                        questions.append({
                            'text': question_text,
                            'type': 'detailed',
                            'category': current_category,
                            'priority': current_priority
                        })

        return questions

    def categorize_questions(self, questions: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Kategorizuje otázky podle typu nebo tématu

        Args:
            questions: Seznam otázek

        Returns:
            Slovník s kategorizovanými otázkami
        """
        categorized = {}

        for question in questions:
            category = question.get('category', 'general')

            if category not in categorized:
                categorized[category] = []

            categorized[category].append(question)

        return categorized

    def prioritize_questions(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Seřadí otázky podle priority

        Args:
            questions: Seznam otázek

        Returns:
            Seřazený seznam otázek
        """
        priority_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}

        def get_priority_value(question):
            priority = question.get('priority', 'normal').lower()
            return priority_order.get(priority, 2)

        return sorted(questions, key=get_priority_value)