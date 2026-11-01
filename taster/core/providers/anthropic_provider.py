"""Anthropic AI provider (Claude)."""
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..ai_client import AIClient, AIResponse, FatalAPIError, is_fatal_api_error
from ..media_prep import ImageEncoder, VideoFrameExtractor


class AnthropicProvider(AIClient):
    """Anthropic provider using the anthropic SDK."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
        video_frame_count: int = 8,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set environment variable or pass to constructor."
            )
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.video_frame_count = video_frame_count

        from ...compat import require
        anthropic = require("anthropic", "anthropic")
        self._client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def supports_video(self) -> bool:
        return False

    def supports_pdf(self) -> bool:
        return True

    # ── prompt conversion ──────────────────────────────────────────────

    def _build_content(
        self, prompt: Union[str, List]
    ) -> List[Dict[str, Any]]:
        """Convert a unified prompt into Anthropic content blocks."""
        if isinstance(prompt, str):
            return [{"type": "text", "text": prompt}]

        blocks: List[Dict[str, Any]] = []
        for item in prompt:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                b64 = ImageEncoder.to_base64(item)
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
            elif isinstance(item, Path):
                blocks.extend(self._convert_path(item))
        return blocks

    def _convert_path(self, path: Path) -> List[Dict[str, Any]]:
        """Convert a Path to Anthropic content blocks."""
        from ..file_utils import FileTypeRegistry

        parts: List[Dict[str, Any]] = []

        if FileTypeRegistry.is_audio(path):
            parts.append({"type": "text", "text": f"[Audio: {path.name}]"})

        elif FileTypeRegistry.is_video(path):
            frames = VideoFrameExtractor.extract_frames(
                path, max_frames=self.video_frame_count
            )
            if frames:
                parts.append({
                    "type": "text",
                    "text": f"[Video: {path.name} — {len(frames)} sampled frames follow]",
                })
                for frame in frames:
                    b64 = ImageEncoder.to_base64(frame)
                    parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    })
            else:
                parts.append({"type": "text", "text": f"[Video: {path.name} — could not extract frames]"})

        elif path.suffix.lower() == ".pdf":
            # Anthropic supports native PDF via base64 document block
            try:
                pdf_bytes = path.read_bytes()
                b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
                parts.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                })
            except Exception as e:
                parts.append({"type": "text", "text": f"[PDF: {path.name} — error reading: {e}]"})

        elif FileTypeRegistry.is_image(path):
            from ..file_utils import ImageUtils

            img = ImageUtils.load_and_fix_orientation(path, max_size=1024)
            if img is not None:
                b64 = ImageEncoder.to_base64(img)
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
            else:
                parts.append({"type": "text", "text": f"[Image: {path.name} — could not load]"})
        else:
            parts.append({"type": "text", "text": f"[File: {path.name}]"})

        return parts

    # ── generation ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt: Union[str, List],
        generation_config: Optional[Dict[str, Any]] = None,
        handle_safety_errors: bool = True,
        rate_limit_delay: float = 0.5,
    ) -> AIResponse:
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        content = self._build_content(prompt)
        gen = generation_config or {}
        max_tokens = gen.get("max_output_tokens", 4096)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

                text = text.strip()
                blocked = response.stop_reason == "end_turn" and not text

                return AIResponse(
                    text=text,
                    raw_response=response,
                    blocked=blocked,
                    provider="anthropic",
                )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if is_fatal_api_error(e):
                    raise FatalAPIError(f"Anthropic API error: {e}") from e
                if any(kw in error_str for kw in ("rate", "overloaded", "timeout", "connection", "unavailable")):
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Warning: Anthropic error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue
                raise type(e)(f"Anthropic API error: {e}") from e

        raise RuntimeError(f"Anthropic: failed after {self.max_retries} retries: {last_error}")

    def generate_json(
        self,
        prompt: Union[str, List],
        fallback: Optional[Dict] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        response = self.generate(
            prompt=prompt,
            generation_config=generation_config,
            **kwargs,
        )
        try:
            return response.parse_json(fallback=fallback)
        except ValueError as e:
            if fallback is not None:
                print(f"Warning: JSON parsing failed, using fallback: {e}")
                return fallback
            raise
