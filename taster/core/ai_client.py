"""Abstract base class for AI provider clients."""
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class AIResponse:
    """Parsed response from an AI provider."""

    def __init__(
        self,
        text: str,
        raw_response: Any,
        blocked: bool = False,
        finish_reason: Optional[int] = None,
        provider: str = "unknown",
    ):
        self.text = text
        self.raw_response = raw_response
        self.blocked = blocked
        self.finish_reason = finish_reason
        self.provider = provider

    def parse_json(self, fallback: Optional[Dict] = None) -> Dict[str, Any]:
        """Parse JSON from response text, handling markdown code blocks."""
        # Remove markdown code blocks
        text_cleaned = re.sub(r'^```json\s*\n', '', self.text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = text_cleaned.strip()

        # Try to parse entire text
        try:
            return json.loads(text_cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*\}', text_cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try to find nested JSON
        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try to recover truncated JSON arrays (e.g. from max_tokens cutoff)
        if text_cleaned.lstrip().startswith('['):
            recovered = self._recover_truncated_array(text_cleaned)
            if recovered:
                return recovered

        if fallback is not None:
            return fallback

        raise ValueError(f"Could not parse JSON from response: {self.text[:200]}")

    @staticmethod
    def _recover_truncated_array(text: str) -> Optional[list]:
        """Recover complete objects from a truncated JSON array.

        When a response hits max_tokens, the JSON array may be cut mid-object.
        This extracts all fully-formed objects that were completed before truncation.
        """
        # Find all complete top-level objects in the array
        items = []
        depth = 0
        obj_start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        items.append(json.loads(text[obj_start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
        return items if items else None


class AIClient(ABC):
    """Abstract base class for AI provider clients."""

    provider_name: str = "unknown"

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Union[str, Image.Image, Path]]],
        generation_config: Optional[Dict[str, Any]] = None,
        handle_safety_errors: bool = True,
        rate_limit_delay: float = 0.5,
    ) -> AIResponse:
        """Generate content from a prompt."""
        ...

    @abstractmethod
    def generate_json(
        self,
        prompt: Union[str, List[Union[str, Image.Image, Path]]],
        fallback: Optional[Dict] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
        ...

    def supports_video(self) -> bool:
        """Whether this provider supports native video upload."""
        return False

    def supports_pdf(self) -> bool:
        """Whether this provider supports native PDF upload."""
        return False
