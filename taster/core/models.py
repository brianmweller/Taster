"""Model management and Gemini API client with unified error handling."""
import os
import re
import json
import time
import warnings
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from PIL import Image

# Suppress proto warning for unrecognized FinishReason enum values (e.g. 19)
warnings.filterwarnings("ignore", message="Unrecognized FinishReason", category=UserWarning)

from .ai_client import AIClient, AIResponse
from ..compat import require


class GeminiError(Exception):
    """Base exception for Gemini API errors."""
    pass


class SafetyFilterError(GeminiError):
    """Exception raised when content is blocked by safety filters."""
    pass


class TokenLimitError(GeminiError):
    """Exception raised when token limit is exceeded."""
    pass


class RateLimitError(GeminiError):
    """Exception raised when rate limit is hit."""
    pass


# Backward compatibility alias
GeminiResponse = AIResponse


class GeminiClient(AIClient):
    """
    Unified Gemini API client with proper error handling and retries.

    Handles:
    - Safety filters
    - Rate limits
    - Token limits
    - JSON parsing
    - Retries with exponential backoff
    """

    provider_name = "gemini"

    def supports_video(self) -> bool:
        return True

    def supports_pdf(self) -> bool:
        return True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Model to use.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Initial retry delay in seconds (doubles each retry).
            timeout: Timeout for API calls in seconds (default: 120s).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Set environment variable or pass to constructor.")

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Configure API
        genai = require("google.generativeai", "gemini")
        self._genai = genai
        genai.configure(api_key=self.api_key)

    def _create_model(self, **model_kwargs):
        """Create model instance with optional kwargs."""
        return self._genai.GenerativeModel(self.model_name, **model_kwargs)

    def _validate_response(self, response: Any) -> GeminiResponse:
        """
        Validate and parse raw API response.

        Args:
            response: Raw response from Gemini API.

        Returns:
            GeminiResponse object.

        Raises:
            SafetyFilterError: If blocked by safety filters.
            GeminiError: If response is invalid.
        """
        # Check if response has candidates
        if not response.candidates:
            raise SafetyFilterError("Response blocked by safety filters")

        candidate = response.candidates[0]

        # Check finish reason
        # 1 = STOP (normal completion)
        # 2 = MAX_TOKENS
        # 3 = SAFETY
        # 4 = RECITATION
        # 5 = OTHER
        # 19 = MALFORMED_FUNCTION_CALL (benign, response still usable)
        if candidate.finish_reason == 3:
            raise SafetyFilterError("Content blocked by safety filters")
        elif candidate.finish_reason == 2:
            # Max tokens - not necessarily an error, but log it
            print(f"Warning: Response reached max tokens")
        elif candidate.finish_reason not in [1, 2, 19]:
            print(f"Warning: Unusual finish reason: {candidate.finish_reason}")

        # Extract text - handle truncated responses gracefully
        try:
            text = response.text.strip()
        except Exception as e:
            # If MAX_TOKENS and text extraction fails, raise retriable error
            if candidate.finish_reason == 2:
                raise GeminiError(f"Response truncated (MAX_TOKENS) and could not extract text: {e}")
            raise GeminiError(f"Could not extract text from response: {e}")

        return AIResponse(
            text=text,
            raw_response=response,
            blocked=False,
            finish_reason=candidate.finish_reason,
            provider="gemini",
        )

    def generate(
        self,
        prompt: Union[str, List[Union[str, Image.Image, Path]]],
        generation_config: Optional[Dict[str, Any]] = None,
        handle_safety_errors: bool = True,
        rate_limit_delay: float = 0.5,
    ) -> GeminiResponse:
        """
        Generate content with retries and error handling.

        Args:
            prompt: Text prompt or list of [text, images, videos].
            generation_config: Generation configuration dict.
            handle_safety_errors: If True, return fallback instead of raising.
            rate_limit_delay: Delay between requests to avoid rate limits.

        Returns:
            GeminiResponse object.

        Raises:
            GeminiError: If generation fails after all retries.
        """
        # Convert single string to list
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            # Load images from paths
            prompt = self._load_images_in_prompt(prompt)

        # Default generation config
        if generation_config is None:
            generation_config = {"max_output_tokens": 4096}

        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                model = self._create_model()

                # Note: Gemini API doesn't directly support timeout parameter in generate_content
                # We rely on the underlying HTTP client's timeout (typically set via environment)
                # For production use, consider wrapping in threading.Timer or asyncio.timeout
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": self.timeout}  # Passes to underlying HTTP client
                )
                return self._validate_response(response)

            except SafetyFilterError as e:
                if handle_safety_errors:
                    # Return fallback response
                    print(f"Safety filter triggered: {e}")
                    return AIResponse(
                        text='{"classification": "Review", "confidence": 0.3, "reasoning": "Blocked by safety filters"}',
                        raw_response=None,
                        blocked=True,
                        finish_reason=3,
                        provider="gemini",
                    )
                else:
                    raise

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if timeout error
                if "timeout" in error_str or "timed out" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Request timeout ({self.timeout}s), retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Check if rate limit error
                if "rate" in error_str or "quota" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Check if temporary error
                if "unavailable" in error_str or "connection" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Temporary error, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Unknown error - fail fast
                raise GeminiError(f"Gemini API error: {e}") from e

        # All retries failed
        raise GeminiError(f"Failed after {self.max_retries} retries: {last_error}")

    def generate_json(
        self,
        prompt: Union[str, List[Union[str, Image.Image, Path]]],
        fallback: Optional[Dict] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content and parse as JSON.

        Args:
            prompt: Text prompt or list of [text, images, videos].
            fallback: Fallback dict if JSON parsing fails.
            generation_config: Generation configuration dict.
            **kwargs: Additional arguments passed to generate().

        Returns:
            Parsed JSON dict or fallback.
        """
        response = self.generate(
            prompt=prompt,
            generation_config=generation_config,
            **kwargs
        )

        try:
            return response.parse_json(fallback=fallback)
        except ValueError as e:
            if fallback is not None:
                print(f"JSON parsing failed, using fallback: {e}")
                return fallback
            raise

    def _load_images_in_prompt(self, prompt: List) -> List:
        """
        Load images from Path objects in prompt, upload videos.

        Args:
            prompt: List potentially containing Path objects.

        Returns:
            List with Paths replaced by PIL Images or uploaded video files.
        """
        from .file_utils import ImageUtils, FileTypeRegistry

        result = []
        for item in prompt:
            if isinstance(item, Path):
                if FileTypeRegistry.is_video(item) or FileTypeRegistry.is_audio(item):
                    # Upload video/audio to Gemini and wait for processing
                    media_label = "Audio" if FileTypeRegistry.is_audio(item) else "Video"
                    uploaded = False
                    for upload_attempt in range(3):
                        try:
                            uploaded_file = self._genai.upload_file(str(item))
                            # Wait for file to be ready (ACTIVE state), with timeout
                            wait_start = time.time()
                            while uploaded_file.state.name == "PROCESSING":
                                if time.time() - wait_start > 300:
                                    print(f"Warning: {media_label} processing timed out for {item}")
                                    break
                                time.sleep(2)
                                uploaded_file = self._genai.get_file(uploaded_file.name)
                            if uploaded_file.state.name == "FAILED":
                                print(f"Warning: {media_label} processing failed for {item}")
                                break
                            if uploaded_file.state.name == "ACTIVE":
                                result.append(uploaded_file)
                                uploaded = True
                            break
                        except Exception as e:
                            if upload_attempt < 2:
                                delay = 2 ** (upload_attempt + 1)
                                print(f"Warning: {media_label} upload failed for {item.name}, retrying in {delay}s... ({e})")
                                time.sleep(delay)
                            else:
                                print(f"Warning: Error uploading {media_label.lower()} {item}: {e}")
                else:
                    # Load image
                    try:
                        img = ImageUtils.load_and_fix_orientation(item)
                        if img is not None:
                            result.append(img)
                        else:
                            print(f"Warning: Could not load image {item}")
                    except Exception as e:
                        print(f"Warning: Error loading image {item}: {e}")
            else:
                result.append(item)
        return result

    def count_tokens(self, prompt: Union[str, List]) -> int:
        """
        Estimate token count for prompt.

        Args:
            prompt: Text prompt or list of [text, images].

        Returns:
            Estimated token count.
        """
        try:
            model = self._create_model()
            if isinstance(prompt, str):
                prompt = [prompt]
            return model.count_tokens(prompt).total_tokens
        except Exception:
            # Fallback to rough estimate
            if isinstance(prompt, str):
                return len(prompt) // 4  # Rough estimate: 4 chars per token
            return 0


def initialize_gemini(api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview") -> GeminiClient:
    """
    Initialize Gemini client with API key.

    Args:
        api_key: API key. If None, reads from environment.
        model_name: Model name to use.

    Returns:
        GeminiClient instance.
    """
    return GeminiClient(api_key=api_key, model_name=model_name)


class GeminiImageClient:
    """
    Gemini client for image generation/improvement.

    Uses Gemini's native image generation capabilities to enhance photos
    while preserving their content.

    NOTE: Requires the NEW google-genai SDK (not google-generativeai).
    Install with: pip install google-genai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-image-preview",
        max_output_tokens: int = 8192,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        """
        Initialize Gemini image client.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Model for image generation (default: gemini-3-pro-image-preview).
            max_output_tokens: Maximum output tokens for image generation.
            max_retries: Maximum number of retries.
            retry_delay: Initial retry delay in seconds.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Set environment variable or pass to constructor.")

        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Import and configure the NEW google-genai SDK
        try:
            from google import genai as genai_new
            from google.genai import types
            self._genai = genai_new
            self._types = types
            self._client = genai_new.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Image generation requires the google-genai SDK.\n"
                "Install with: pip install google-genai"
            )

    def _get_closest_aspect_ratio(self, width: int, height: int) -> str:
        """
        Find the closest supported aspect ratio for an image.

        Supported ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9
        """
        SUPPORTED_RATIOS = {
            "1:1": 1.0,
            "2:3": 2/3,
            "3:2": 3/2,
            "3:4": 3/4,
            "4:3": 4/3,
            "9:16": 9/16,
            "16:9": 16/9,
            "21:9": 21/9,
        }

        actual_ratio = width / height
        closest = min(SUPPORTED_RATIOS.items(), key=lambda x: abs(x[1] - actual_ratio))
        return closest[0]

    def generate_improved_image(
        self,
        image_path: Path,
        prompt: str
    ) -> Optional[bytes]:
        """
        Generate an improved version of an image using Gemini.

        Args:
            image_path: Path to the source image.
            prompt: Improvement prompt describing what to enhance.

        Returns:
            Improved image bytes, or None if generation fails.
        """
        from .file_utils import ImageUtils

        # Load source image at high resolution (up to 2048px for quality)
        source_image = ImageUtils.load_and_fix_orientation(image_path, max_size=2048)
        if source_image is None:
            print(f"Could not load image: {image_path}")
            return None

        # Get the aspect ratio to preserve
        width, height = source_image.size
        aspect_ratio = self._get_closest_aspect_ratio(width, height)

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Pass PIL Image directly per Google's documentation
                # Generate content with image input using new SDK
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, source_image],
                    config=self._types.GenerateContentConfig(
                        response_modalities=["Text", "Image"],
                        image_config=self._types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size="2K"
                        )
                    )
                )

                # Extract image from response
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]

                    # Look for image part in response
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Check if it's an image
                            mime_type = part.inline_data.mime_type
                            if mime_type and mime_type.startswith('image/'):
                                return part.inline_data.data

                # No image found in response
                print(f"No image generated in response")
                return None

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if rate limit error
                if "rate" in error_str or "quota" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Check if temporary error
                if "unavailable" in error_str or "connection" in error_str:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Temporary error, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                # Check for model not available
                if "model" in error_str and ("not found" in error_str or "unavailable" in error_str):
                    print(f"Model {self.model_name} not available: {e}")
                    print("   Check https://ai.google.dev/gemini-api/docs/pricing for current models")
                    return None

                # Unknown error
                print(f"Image generation error: {e}")
                return None

        # All retries failed
        print(f"Image generation failed after {self.max_retries} retries: {last_error}")
        return None

    def save_image(self, image_bytes: bytes, output_path: Path) -> bool:
        """
        Save generated image bytes to file.

        Args:
            image_bytes: Image data bytes.
            output_path: Path to save the image.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
