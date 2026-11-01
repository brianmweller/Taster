"""OpenAI AI provider (GPT-5)."""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..ai_client import AIClient, AIResponse, FatalAPIError, is_fatal_api_error
from ..media_prep import ImageEncoder, VideoFrameExtractor, PDFPageRenderer


class OpenAIProvider(AIClient):
    """OpenAI provider using the openai SDK."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5.2",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
        video_frame_count: int = 8,
        pdf_render_dpi: int = 150,
        base_url: Optional[str] = None,
    ):
        self.base_url = base_url
        if base_url:
            # Local/custom endpoint — no API key required
            self.api_key = api_key or "not-needed"
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Set environment variable or pass to constructor."
                )
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.video_frame_count = video_frame_count
        self.pdf_render_dpi = pdf_render_dpi

        from ...compat import require
        openai = require("openai", "openai")
        client_kwargs = dict(api_key=self.api_key, timeout=self.timeout)
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = openai.OpenAI(**client_kwargs)

    def supports_video(self) -> bool:
        return False

    def supports_pdf(self) -> bool:
        return False

    # ── prompt conversion ──────────────────────────────────────────────

    def _build_messages(
        self, prompt: Union[str, List]
    ) -> List[Dict[str, Any]]:
        """Convert a unified prompt into OpenAI messages format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        content_parts: List[Dict[str, Any]] = []
        for item in prompt:
            if isinstance(item, str):
                content_parts.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                b64 = ImageEncoder.to_base64(item)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            elif isinstance(item, Path):
                content_parts.extend(self._convert_path(item))
            # Skip unknown types
        return [{"role": "user", "content": content_parts}]

    def _convert_path(self, path: Path) -> List[Dict[str, Any]]:
        """Convert a Path to one or more OpenAI content blocks."""
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
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            else:
                parts.append({"type": "text", "text": f"[Video: {path.name} — could not extract frames]"})
        elif path.suffix.lower() == ".pdf":
            pages = PDFPageRenderer.render_pages(path, dpi=self.pdf_render_dpi)
            if pages:
                parts.append({
                    "type": "text",
                    "text": f"[PDF: {path.name} — {len(pages)} rendered pages follow]",
                })
                for page_img in pages:
                    b64 = ImageEncoder.to_base64(page_img, max_size=1536)
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            else:
                parts.append({"type": "text", "text": f"[PDF: {path.name} — could not render pages]"})
        elif FileTypeRegistry.is_image(path):
            from ..file_utils import ImageUtils

            img = ImageUtils.load_and_fix_orientation(path, max_size=1024)
            if img is not None:
                b64 = ImageEncoder.to_base64(img)
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
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

        messages = self._build_messages(prompt)
        gen = generation_config or {}
        max_tokens = gen.get("max_output_tokens", 4096)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=gen.get("temperature", 0.2),
                )
                choice = response.choices[0]
                text = (choice.message.content or "").strip()

                blocked = choice.finish_reason == "content_filter"
                if blocked and handle_safety_errors:
                    print(f"Warning: OpenAI content filter triggered")
                    return AIResponse(
                        text='{"classification": "Review", "confidence": 0.3, "reasoning": "Blocked by content filter"}',
                        raw_response=response,
                        blocked=True,
                        provider="openai",
                    )

                return AIResponse(
                    text=text,
                    raw_response=response,
                    blocked=blocked,
                    provider="openai",
                )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if is_fatal_api_error(e):
                    raise FatalAPIError(f"OpenAI API error: {e}") from e
                if any(kw in error_str for kw in ("rate", "quota", "timeout", "connection", "unavailable")):
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Warning: OpenAI error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue
                raise type(e)(f"OpenAI API error: {e}") from e

        raise RuntimeError(f"OpenAI: failed after {self.max_retries} retries: {last_error}")

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
