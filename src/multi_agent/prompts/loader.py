"""
Prompt Loader - Load prompts from prompts/agents/ folder

Implements:
- Caching for performance
- Prompt validation
- Dynamic formatting with context
- Hot-reloading in development
"""

from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Load agent prompts from folder.

    Prompts are cached in memory for performance.
    In development, prompts can be reloaded without restart.
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt loader.

        Args:
            prompts_dir: Directory containing agent prompts (default: prompts/agents)
        """
        if prompts_dir is None:
            # Default to prompts/agents from project root
            prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts" / "agents"

        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
        self._validate_prompts_dir()

        # Load all prompts on startup
        self._load_all_prompts()

    def _validate_prompts_dir(self) -> None:
        """Validate prompts directory exists."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {self.prompts_dir}\n"
                f"Expected structure: {self.prompts_dir}/<agent_name>.txt"
            )

        if not self.prompts_dir.is_dir():
            raise NotADirectoryError(
                f"Prompts path is not a directory: {self.prompts_dir}"
            )

    def _load_all_prompts(self) -> None:
        """Load all prompts from directory on startup."""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return

        prompt_files = list(self.prompts_dir.glob("*.txt"))
        if not prompt_files:
            logger.warning(f"No prompt files found in {self.prompts_dir}")
            return

        for prompt_file in prompt_files:
            agent_name = prompt_file.stem
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        self._cache[agent_name] = content
                        logger.debug(f"Loaded prompt: {agent_name} ({len(content)} chars)")
                    else:
                        logger.warning(f"Empty prompt file: {prompt_file}")
            except Exception as e:
                logger.error(f"Failed to load prompt {agent_name}: {e}")

        logger.info(f"Loaded {len(self._cache)} agent prompts from {self.prompts_dir}")

    def get_prompt(self, agent_name: str, reload: bool = False) -> str:
        """
        Get agent prompt.

        Args:
            agent_name: Name of agent (without .txt extension)
            reload: Force reload from disk (for development)

        Returns:
            Prompt text

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Check cache first (unless reload requested)
        if not reload and agent_name in self._cache:
            return self._cache[agent_name]

        # Load from disk
        prompt_file = self.prompts_dir / f"{agent_name}.txt"

        if not prompt_file.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            # Return fallback prompt
            return self._get_fallback_prompt(agent_name)

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Cache for future use
            self._cache[agent_name] = content

            logger.debug(f"{'Reloaded' if reload else 'Loaded'} prompt: {agent_name}")
            return content

        except Exception as e:
            logger.error(f"Failed to load prompt {agent_name}: {e}")
            return self._get_fallback_prompt(agent_name)

    def _get_fallback_prompt(self, agent_name: str) -> str:
        """
        Get fallback prompt if file not found.

        Args:
            agent_name: Agent name

        Returns:
            Generic fallback prompt
        """
        logger.warning(f"Using fallback prompt for {agent_name}")
        return (
            f"You are the {agent_name.upper()} agent. "
            f"Perform your assigned tasks according to your role and available tools."
        )

    def format_prompt(
        self,
        agent_name: str,
        context: Optional[Dict[str, str]] = None,
        reload: bool = False
    ) -> str:
        """
        Format prompt with context variables.

        Supports Python string formatting with {variable_name}.

        Args:
            agent_name: Agent name
            context: Dict of variables to substitute
            reload: Force reload from disk

        Returns:
            Formatted prompt text
        """
        prompt = self.get_prompt(agent_name, reload=reload)

        if context is None or not context:
            return prompt

        try:
            return prompt.format(**context)
        except KeyError as e:
            logger.warning(
                f"Prompt formatting failed for {agent_name}: missing variable {e}. "
                f"Returning unformatted prompt."
            )
            return prompt
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt {agent_name}: {e}")
            return prompt

    def list_available_prompts(self) -> list[str]:
        """
        List all available agent prompts.

        Returns:
            List of agent names with available prompts
        """
        return sorted(self._cache.keys())

    def reload_all(self) -> None:
        """Reload all prompts from disk (for development)."""
        self._cache.clear()
        self._load_all_prompts()
        logger.info("Reloaded all prompts from disk")

    def get_prompt_stats(self) -> Dict[str, any]:
        """
        Get prompt loading statistics.

        Returns:
            Dict with stats (count, sizes, etc.)
        """
        return {
            "total_prompts": len(self._cache),
            "prompts": {
                name: {
                    "length": len(content),
                    "lines": content.count("\n") + 1,
                    "words": len(content.split())
                }
                for name, content in self._cache.items()
            },
            "directory": str(self.prompts_dir)
        }


# Global prompt loader instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_dir: Optional[Path] = None) -> PromptLoader:
    """
    Get global prompt loader instance.

    Args:
        prompts_dir: Optional custom prompts directory

    Returns:
        PromptLoader instance
    """
    global _prompt_loader

    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_dir)

    return _prompt_loader


def load_prompt(agent_name: str, reload: bool = False) -> str:
    """
    Convenience function to load a prompt.

    Args:
        agent_name: Agent name
        reload: Force reload from disk

    Returns:
        Prompt text
    """
    loader = get_prompt_loader()
    return loader.get_prompt(agent_name, reload=reload)
