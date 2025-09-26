"""
Prompt Manager - Modul pro správu a načítání prompt templates
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Třída pro správu prompt templates"""

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Inicializace prompt manageru

        Args:
            prompts_dir: Cesta k adresáři s prompty (relativní nebo absolutní)
        """
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            # Výchozí relativní cesta
            self.prompts_dir = Path(__file__).parent.parent / "prompts"

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Adresář s prompty neexistuje: {self.prompts_dir}")

        self.prompts_cache = {}
        self._load_all_prompts()

    def _load_all_prompts(self):
        """Načte všechny dostupné prompt templates"""
        for prompt_file in self.prompts_dir.glob("*.md"):
            prompt_name = prompt_file.stem
            try:
                self.prompts_cache[prompt_name] = self._load_prompt_file(prompt_file)
                logger.info(f"Načten prompt template: {prompt_name}")
            except Exception as e:
                logger.error(f"Chyba při načítání promptu {prompt_name}: {e}")

    def _load_prompt_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Načte jeden prompt soubor

        Args:
            file_path: Cesta k souboru s promptem

        Returns:
            Slovník s metadaty a obsahem promptu
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parsování YAML frontmatter
        metadata = {}
        prompt_content = content

        if content.startswith('---'):
            try:
                # Najdi konec frontmatter
                end_index = content.index('---', 3)
                yaml_content = content[3:end_index].strip()
                prompt_content = content[end_index + 3:].strip()

                # Parsuj YAML metadata
                metadata = yaml.safe_load(yaml_content) or {}
            except (ValueError, yaml.YAMLError) as e:
                logger.warning(f"Nepodařilo se parsovat YAML metadata v {file_path}: {e}")

        return {
            'name': metadata.get('name', file_path.stem),
            'description': metadata.get('description', ''),
            'metadata': metadata,
            'template': prompt_content,
            'file_path': str(file_path.relative_to(self.prompts_dir.parent))
        }

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Získá prompt a nahradí v něm proměnné

        Args:
            prompt_name: Název promptu
            **kwargs: Proměnné k nahrazení v promptu

        Returns:
            Vyplněný prompt
        """
        if prompt_name not in self.prompts_cache:
            raise ValueError(f"Prompt '{prompt_name}' nebyl nalezen")

        prompt_data = self.prompts_cache[prompt_name]
        template = prompt_data['template']

        # Nahrazení proměnných ve formátu {variable_name}
        return self._fill_template(template, kwargs)

    def _fill_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Nahradí proměnné v template

        Args:
            template: Template string
            variables: Slovník s proměnnými

        Returns:
            Vyplněný template
        """
        result = template

        # Najdi všechny proměnné ve formátu {variable_name} ale ignoruj {{ a }}
        # které jsou escape sekvence pro JSON
        pattern = r'(?<!\{)\{([^{}]+)\}(?!\})'

        for match in re.finditer(pattern, template):
            placeholder = match.group(1)
            if placeholder in variables:
                value = str(variables[placeholder])
                result = result.replace(f'{{{placeholder}}}', value, 1)
            elif not placeholder.startswith('{') and not placeholder.endswith('}'):
                # Log warning pouze pro skutečné proměnné, ne pro JSON syntax
                logger.warning(f"Proměnná '{placeholder}' nebyla poskytnuta")

        return result

    def list_prompts(self) -> Dict[str, str]:
        """
        Vrátí seznam dostupných promptů s jejich popisy

        Returns:
            Slovník {název: popis}
        """
        return {
            name: data['description']
            for name, data in self.prompts_cache.items()
        }

    def reload_prompts(self):
        """Znovu načte všechny prompty (užitečné pro vývoj)"""
        self.prompts_cache.clear()
        self._load_all_prompts()
        logger.info("Prompty byly znovu načteny")

    def get_prompt_metadata(self, prompt_name: str) -> Dict[str, Any]:
        """
        Získá metadata promptu

        Args:
            prompt_name: Název promptu

        Returns:
            Metadata promptu
        """
        if prompt_name not in self.prompts_cache:
            raise ValueError(f"Prompt '{prompt_name}' nebyl nalezen")

        return self.prompts_cache[prompt_name]['metadata']

    def validate_prompt_variables(self, prompt_name: str, variables: Dict[str, Any]) -> bool:
        """
        Validuje, že všechny požadované proměnné jsou poskytnuty

        Args:
            prompt_name: Název promptu
            variables: Poskytnuté proměnné

        Returns:
            True pokud jsou všechny proměnné poskytnuty
        """
        if prompt_name not in self.prompts_cache:
            return False

        template = self.prompts_cache[prompt_name]['template']
        pattern = r'\{([^}]+)\}'
        required_variables = set(re.findall(pattern, template))
        provided_variables = set(variables.keys())

        missing = required_variables - provided_variables
        if missing:
            logger.warning(f"Chybějící proměnné pro prompt '{prompt_name}': {missing}")
            return False

        return True