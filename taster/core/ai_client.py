"""Abstract base class for AI provider clients."""
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class FatalAPIError(Exception):
    """Unrecoverable provider error (billing, auth, missing model).

    Retrying cannot succeed, so the run must stop instead of routing every
    file to a fallback category.
    """
    pass


class ClassificationAborted(FatalAPIError):
    """Raised when the error-rate circuit breaker trips mid-run."""
    pass


# Substrings (lowercased) that mark an error as unrecoverable across providers.
_FATAL_ERROR_PATTERNS = (
    # Billing / credits
    "credits are depleted",          # Gemini prepay
    "prepayment",
    "billing",
    "insufficient_quota",            # OpenAI
    "credit balance is too low",     # Anthropic
    # Authentication / authorization
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "invalid x-api-key",
    "incorrect api key",
    "unauthenticated",
    "permission_denied",
    "permission denied",
    "authentication_error",
    # Model retired or misnamed
    "is not found for api version",
    "model_not_found",
    "not_found_error",
)


def is_fatal_api_error(error: Union[BaseException, str]) -> bool:
    """Return True if an API error cannot be fixed by retrying."""
    if isinstance(error, FatalAPIError):
        return True
    text = str(error).lower()
    return any(p in text for p in _FATAL_ERROR_PATTERNS)


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

        # Try to find JSON array in text (must come before single-object search
        # to avoid extracting one object from inside an array)
        array_match = re.search(r'\[.*\]', text_cleaned, re.DOTALL)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to recover truncated JSON arrays (e.g. from max_tokens cutoff)
        recovered = self._recover_truncated_array(text_cleaned)
        if recovered:
            return recovered

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

        if fallback is not None:
            return fallback

        raise ValueError(f"Could not parse JSON from response: {self.text[:200]}")

    @staticmethod
    def _recover_truncated_array(text: str) -> Optional[list]:
        """Recover complete objects from a truncated JSON array.

        When a response hits max_tokens, the JSON array may be cut mid-object.
        This extracts all fully-formed objects that were completed before truncation.
        Only triggers if the text contains '[' (potential array start).
        """
        bracket_pos = text.find('[')
        if bracket_pos == -1:
            return None

        # Find all complete top-level objects after the array opening bracket
        items = []
        depth = 0
        obj_start = None
        for i in range(bracket_pos, len(text)):
            ch = text[i]
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

    def preflight(self) -> None:
        """Make one tiny API call to verify key, billing and model before a run.

        Raises:
            FatalAPIError: If the provider rejects the call for a reason
                retrying cannot fix (billing, auth, unknown model).
        """
        try:
            self.generate(
                "Reply with the single word OK.",
                generation_config={"max_output_tokens": 256},
                rate_limit_delay=0,
            )
        except FatalAPIError:
            raise
        except Exception as e:
            if is_fatal_api_error(e):
                raise FatalAPIError(str(e)) from e
            # Transient problems are left to per-file retries and the circuit breaker
            print(f"Warning: preflight check failed ({e}); continuing")

    def supports_video(self) -> bool:
        """Whether this provider supports native video upload."""
        return False

    def supports_pdf(self) -> bool:
        """Whether this provider supports native PDF upload."""
        return False
