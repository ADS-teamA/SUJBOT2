"""
Wrapper pro Claude API používající standardní Anthropic SDK
Simuluje rozhraní claude_code_sdk pro kompatibilitu
"""

import os
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass
import asyncio
from anthropic import AsyncAnthropic
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCodeOptions:
    """Možnosti pro Claude API volání"""
    allowed_tools: List[str] = None
    permission_mode: str = 'acceptOnly'
    system_prompt: str = ''
    model: str = 'claude-3-5-haiku-20241022'
    max_tokens: int = 4000
    temperature: float = 0.7


class ClaudeSDKClient:
    """Wrapper pro Anthropic SDK simulující claude_code_sdk rozhraní"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("API klíč není nastaven. Nastavte CLAUDE_API_KEY v .env souboru.")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.current_message = None
        self.options = None

    async def query(self, prompt: str, options: Optional[ClaudeCodeOptions] = None) -> str:
        """Odešle dotaz na Claude API a vrátí odpověď"""
        self.options = options or ClaudeCodeOptions()

        messages = [{"role": "user", "content": prompt}]

        try:
            # Volání Claude API
            # Prepare system parameter - only include if not empty
            create_params = {
                'model': self.options.model,
                'messages': messages,
                'max_tokens': self.options.max_tokens,
                'temperature': self.options.temperature
            }

            if self.options.system_prompt and self.options.system_prompt.strip():
                create_params['system'] = self.options.system_prompt

            self.current_message = await self.client.messages.create(**create_params)
            logger.info(f"Dotaz odeslán na model {self.options.model}")

            # Extract and return response text
            response_parts = []
            for content in self.current_message.content:
                if content.type == 'text':
                    response_parts.append(content.text)

            return "".join(response_parts)

        except Exception as e:
            logger.error(f"Chyba při volání Claude API: {e}")
            raise

    async def receive_response(self) -> AsyncIterator[Dict[str, Any]]:
        """Přijme odpověď od Claude API"""
        if not self.current_message:
            yield {"type": "error", "text": "Žádná zpráva k přijetí"}
            return

        # Vrací obsah odpovědi jako stream
        for content in self.current_message.content:
            if content.type == 'text':
                yield {"type": "text", "text": content.text}

    async def __aenter__(self):
        """Vstup do context manageru"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Výstup z context manageru"""
        pass


# Pro zpětnou kompatibilitu - pokud někdo importuje přímo query
async def query(prompt: str, options: Optional[ClaudeCodeOptions] = None) -> AsyncIterator[str]:
    """Jednoduchý query interface pro rychlé použití"""
    async with ClaudeSDKClient() as client:
        await client.query(prompt, options)
        async for message in client.receive_response():
            if message.get("type") == "text":
                yield message.get("text", "")