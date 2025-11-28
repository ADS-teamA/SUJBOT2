"""
Conversation Title Generator using Qwen 2.5 7B via DeepInfra.

Generates concise, descriptive titles from the first user message.
Uses async OpenAI-compatible API for non-blocking operation.
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    Generate conversation titles using Qwen 2.5 7B via DeepInfra.

    Uses the lighter 7B model for fast, cost-effective title generation.
    Thread-safe and designed for concurrent usage across multiple workers.
    """

    def __init__(self):
        """Initialize the title generator with DeepInfra client."""
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepinfra.com/v1/openai",
                timeout=15.0,
                max_retries=2
            )
            self.model = "Qwen/Qwen2.5-7B-Instruct"
            logger.info("TitleGenerator initialized with DeepInfra Qwen 2.5 7B")
        else:
            self.client = None
            self.model = None
            logger.warning("DEEPINFRA_API_KEY not set - title generation will use fallback")

    async def generate_title(self, first_message: str) -> Optional[str]:
        """
        Generate a conversation title from the first user message.

        Args:
            first_message: The user's first message in the conversation

        Returns:
            Generated title (max 60 chars), or fallback to truncated message
        """
        if not first_message or not first_message.strip():
            return None

        # Fallback if DeepInfra not configured
        if not self.client:
            return self._fallback_title(first_message)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a title generator. Your ONLY job is to create a short title "
                            "(2-5 words) that summarizes the user's message topic.\n\n"
                            "Rules:\n"
                            "- Output ONLY the title, nothing else\n"
                            "- No quotes, no punctuation at the end\n"
                            "- No explanations or preamble\n"
                            "- If the message is a greeting, output: Přátelský pozdrav\n"
                            "- If unclear, summarize the main topic in 2-5 words\n\n"
                            "Examples:\n"
                            "User: 'Ahoj, jak se máš?' → Přátelský pozdrav\n"
                            "User: 'Jak napsat rekurzi v Pythonu?' → Rekurze v Pythonu\n"
                            "User: 'Vysvětli mi kvantovou mechaniku' → Kvantová mechanika"
                        )
                    },
                    {"role": "user", "content": first_message[:500]}
                ],
                max_tokens=20,
                temperature=0.2
            )

            raw_title = response.choices[0].message.content
            logger.info(f"LLM raw title response: '{raw_title}' for message: '{first_message[:50]}...'")

            if raw_title:
                title = self._clean_title(raw_title)
                logger.info(f"Cleaned title: '{title}'")

                # If LLM returned the original message or something too long, use fallback
                if title and len(title) <= 60 and title.lower() != first_message.lower().strip():
                    logger.info(f"Using LLM title: '{title}'")
                    return title
                else:
                    logger.warning(f"Title rejected (too long or same as message), using fallback")

            fallback = self._fallback_title(first_message)
            logger.info(f"Using fallback title: '{fallback}'")
            return fallback

        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return self._fallback_title(first_message)

    def _clean_title(self, title: str) -> str:
        """Clean up LLM-generated title."""
        if not title:
            return ""
        # Remove common prefixes LLMs add
        prefixes_to_remove = [
            "Title:", "title:", "Název:", "název:",
            "Here is", "Here's", "The title is",
        ]
        cleaned = title.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        # Remove quotes and trailing punctuation
        cleaned = cleaned.strip('"\'').rstrip('.!?:')
        return cleaned

    def _fallback_title(self, message: str) -> str:
        """Generate fallback title by truncating the message."""
        # Clean up whitespace and truncate
        clean = " ".join(message.split())
        if len(clean) <= 50:
            return clean
        return clean[:47] + "..."


# Singleton instance for use across the application
title_generator = TitleGenerator()
