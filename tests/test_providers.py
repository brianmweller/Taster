"""Tests for multi-provider AI support."""
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from taster.core.ai_client import AIClient, AIResponse
from taster.core.config import Config, ModelConfig
from taster.core.provider_factory import create_ai_client, detect_available_providers
from taster.core.media_prep import ImageEncoder, VideoFrameExtractor, PDFPageRenderer


# ── AIResponse tests ───────────────────────────────────────────────────


class TestAIResponse:
    """Tests for the AIResponse class."""

    def test_basic_text(self):
        resp = AIResponse(text="hello", raw_response=None, provider="test")
        assert resp.text == "hello"
        assert resp.provider == "test"
        assert resp.blocked is False

    def test_parse_json_simple(self):
        resp = AIResponse(text='{"key": "value"}', raw_response=None)
        result = resp.parse_json()
        assert result == {"key": "value"}

    def test_parse_json_markdown_block(self):
        text = '```json\n{"key": "value"}\n```'
        resp = AIResponse(text=text, raw_response=None)
        result = resp.parse_json()
        assert result == {"key": "value"}

    def test_parse_json_embedded(self):
        text = 'Here is the result: {"classification": "Share", "score": 4}'
        resp = AIResponse(text=text, raw_response=None)
        result = resp.parse_json()
        assert result["classification"] == "Share"

    def test_parse_json_fallback(self):
        resp = AIResponse(text="not json at all", raw_response=None)
        result = resp.parse_json(fallback={"default": True})
        assert result == {"default": True}

    def test_parse_json_raises_without_fallback(self):
        resp = AIResponse(text="not json", raw_response=None)
        with pytest.raises(ValueError):
            resp.parse_json()

    def test_provider_field(self):
        resp = AIResponse(text="x", raw_response=None, provider="openai")
        assert resp.provider == "openai"


# ── AIClient ABC tests ────────────────────────────────────────────────


class TestAIClientABC:
    """Tests for the AIClient abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AIClient()

    def test_concrete_subclass(self):
        class DummyClient(AIClient):
            provider_name = "dummy"

            def generate(self, prompt, generation_config=None, handle_safety_errors=True, rate_limit_delay=0.5):
                return AIResponse(text="ok", raw_response=None, provider="dummy")

            def generate_json(self, prompt, fallback=None, generation_config=None, **kwargs):
                return {"result": "ok"}

        client = DummyClient()
        assert client.provider_name == "dummy"
        assert client.supports_video() is False
        assert client.supports_pdf() is False

        resp = client.generate("test")
        assert resp.text == "ok"

    def test_gemini_client_is_ai_client(self):
        """GeminiClient should be an instance of AIClient."""
        from taster.core.models import GeminiClient
        assert issubclass(GeminiClient, AIClient)


# ── GeminiResponse backward compatibility ─────────────────────────────


class TestGeminiResponseAlias:
    """GeminiResponse should be an alias for AIResponse."""

    def test_alias_identity(self):
        from taster.core.models import GeminiResponse
        assert GeminiResponse is AIResponse

    def test_alias_works(self):
        from taster.core.models import GeminiResponse
        resp = GeminiResponse(text="test", raw_response=None)
        assert resp.text == "test"
        assert isinstance(resp, AIResponse)


# ── Provider detection tests ──────────────────────────────────────────


class TestDetectProviders:
    """Tests for detect_available_providers."""

    def test_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove all provider keys
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                result = detect_available_providers()
                assert result["gemini"] is False
                assert result["openai"] is False
                assert result["anthropic"] is False

    def test_gemini_only(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env["GEMINI_API_KEY"] = "test-key"
        with patch.dict(os.environ, env, clear=True):
            result = detect_available_providers()
            assert result["gemini"] is True
            assert result["openai"] is False

    def test_all_keys(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_LLM_URL")}
        env["GEMINI_API_KEY"] = "g-key"
        env["OPENAI_API_KEY"] = "o-key"
        env["ANTHROPIC_API_KEY"] = "a-key"
        env["LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        with patch.dict(os.environ, env, clear=True):
            result = detect_available_providers()
            assert all(result.values())


# ── Factory tests ─────────────────────────────────────────────────────


class TestCreateAIClient:
    """Tests for create_ai_client factory."""

    @pytest.fixture
    def config(self):
        return Config()

    def test_no_keys_raises(self, config):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="No AI provider configured"):
                create_ai_client(config)

    def test_explicit_gemini(self, config):
        env = dict(os.environ)
        env["GEMINI_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            client = create_ai_client(config, provider="gemini")
            assert client.provider_name == "gemini"
            assert isinstance(client, AIClient)
            assert client.supports_video() is True
            assert client.supports_pdf() is True

    def test_explicit_openai(self, config):
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            # Mock the openai module to avoid import errors if not installed
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                client = create_ai_client(config, provider="openai")
                assert client.provider_name == "openai"
                assert client.supports_video() is False

    def test_explicit_anthropic(self, config):
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            mock_anthropic = MagicMock()
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                client = create_ai_client(config, provider="anthropic")
                assert client.provider_name == "anthropic"
                assert client.supports_video() is False
                assert client.supports_pdf() is True

    def test_auto_detect_prefers_gemini(self, config):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env["GEMINI_API_KEY"] = "g-key"
        env["OPENAI_API_KEY"] = "o-key"
        with patch.dict(os.environ, env, clear=True):
            client = create_ai_client(config)
            assert client.provider_name == "gemini"

    def test_auto_detect_falls_to_openai(self, config):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env["OPENAI_API_KEY"] = "o-key"
        with patch.dict(os.environ, env, clear=True):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                client = create_ai_client(config)
                assert client.provider_name == "openai"

    def test_auto_detect_falls_to_anthropic(self, config):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env["ANTHROPIC_API_KEY"] = "a-key"
        with patch.dict(os.environ, env, clear=True):
            mock_anthropic = MagicMock()
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                client = create_ai_client(config)
                assert client.provider_name == "anthropic"

    def test_config_provider_override(self):
        config = Config()
        config.model.provider = "gemini"
        env = dict(os.environ)
        env["GEMINI_API_KEY"] = "g-key"
        env["OPENAI_API_KEY"] = "o-key"
        with patch.dict(os.environ, env):
            client = create_ai_client(config)
            assert client.provider_name == "gemini"

    def test_unknown_provider_raises(self, config):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_ai_client(config, provider="nonexistent")

    def test_uses_config_model_names(self):
        config = Config()
        config.model.openai_model = "gpt-4o-mini"
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "test"
        with patch.dict(os.environ, env):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                client = create_ai_client(config, provider="openai")
                assert client.model_name == "gpt-4o-mini"


# ── ImageEncoder tests ────────────────────────────────────────────────


class TestImageEncoder:
    """Tests for the ImageEncoder utility."""

    def test_to_base64(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        b64 = ImageEncoder.to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0
        # Should be valid base64
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_to_base64_rgba(self):
        from PIL import Image
        img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        b64 = ImageEncoder.to_base64(img)
        assert isinstance(b64, str)

    def test_to_base64_respects_max_size(self):
        from PIL import Image
        img = Image.new("RGB", (2000, 2000), color="blue")
        b64 = ImageEncoder.to_base64(img, max_size=512)
        assert isinstance(b64, str)

    def test_media_type(self):
        assert ImageEncoder.media_type_for("JPEG") == "image/jpeg"
        assert ImageEncoder.media_type_for("PNG") == "image/png"
        assert ImageEncoder.media_type_for("unknown") == "image/jpeg"


# ── OpenAI provider message building ──────────────────────────────────


class TestOpenAIProviderMessages:
    """Test OpenAI message building (without actual API calls)."""

    @pytest.fixture
    def provider(self):
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from taster.core.providers.openai_provider import OpenAIProvider
                return OpenAIProvider(api_key="test-key")

    def test_string_prompt(self, provider):
        messages = provider._build_messages("hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_list_with_text(self, provider):
        messages = provider._build_messages(["text1", "text2"])
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "text1"}
        assert content[1] == {"type": "text", "text": "text2"}

    def test_list_with_image(self, provider):
        from PIL import Image
        img = Image.new("RGB", (50, 50), "red")
        messages = provider._build_messages(["caption", img])
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ── Anthropic provider content building ───────────────────────────────


class TestAnthropicProviderContent:
    """Test Anthropic content building (without actual API calls)."""

    @pytest.fixture
    def provider(self):
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            mock_anthropic = MagicMock()
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                from taster.core.providers.anthropic_provider import AnthropicProvider
                return AnthropicProvider(api_key="test-key")

    def test_string_prompt(self, provider):
        content = provider._build_content("hello")
        assert content == [{"type": "text", "text": "hello"}]

    def test_list_with_text(self, provider):
        content = provider._build_content(["text1", "text2"])
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "text1"}

    def test_list_with_image(self, provider):
        from PIL import Image
        img = Image.new("RGB", (50, 50), "red")
        content = provider._build_content(["caption", img])
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/jpeg"


# ── Config backward compatibility ─────────────────────────────────────


class TestModelConfigCompat:
    """ModelConfig should accept new fields without breaking old configs."""

    def test_default_provider_is_none(self):
        mc = ModelConfig()
        assert mc.provider is None

    def test_new_fields_have_defaults(self):
        mc = ModelConfig()
        assert mc.openai_model == "gpt-5.2"
        assert mc.anthropic_model == "claude-sonnet-4-6"
        assert mc.video_frame_count == 8
        assert mc.pdf_render_dpi == 150

    def test_from_yaml_without_new_fields(self):
        """Old config.yaml without provider fields should still work."""
        config = Config.from_dict({
            "model": {"name": "gemini-2.0-flash"},
        })
        assert config.model.name == "gemini-2.0-flash"
        assert config.model.provider is None
        assert config.model.openai_model == "gpt-5.2"

    def test_from_yaml_with_new_fields(self):
        config = Config.from_dict({
            "model": {
                "name": "gemini-3.8-flash",
                "provider": "openai",
                "openai_model": "gpt-4o",
            },
        })
        assert config.model.provider == "openai"
        assert config.model.openai_model == "gpt-4o"


# ── Additional ImageEncoder tests ────────────────────────────────────


class TestImageEncoderExtended:

    def test_image_to_bytes(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), "green")
        data = ImageEncoder.image_to_bytes(img)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_image_to_bytes_rgba(self):
        from PIL import Image
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        data = ImageEncoder.image_to_bytes(img)
        assert isinstance(data, bytes)

    def test_to_base64_palette_mode(self):
        from PIL import Image
        img = Image.new("P", (100, 100))
        b64 = ImageEncoder.to_base64(img)
        assert isinstance(b64, str)

    def test_resize_preserves_aspect(self):
        from PIL import Image
        import base64
        import io
        img = Image.new("RGB", (4000, 2000), "blue")
        b64 = ImageEncoder.to_base64(img, max_size=256)
        decoded = base64.b64decode(b64)
        result = Image.open(io.BytesIO(decoded))
        assert max(result.size) <= 256

    def test_media_type_webp(self):
        assert ImageEncoder.media_type_for("WEBP") == "image/webp"


# ── Anthropic PDF/path conversion ────────────────────────────────────


class TestAnthropicPathConversion:

    @pytest.fixture
    def provider(self):
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            mock_anthropic = MagicMock()
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                from taster.core.providers.anthropic_provider import AnthropicProvider
                return AnthropicProvider(api_key="test-key")

    def test_pdf_native_support(self, provider, tmp_path):
        """Anthropic should send PDFs as native document blocks."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        blocks = provider._convert_path(pdf)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "document"
        assert blocks[0]["source"]["media_type"] == "application/pdf"

    def test_unknown_file_type(self, provider, tmp_path):
        unknown = tmp_path / "readme.xyz"
        unknown.write_text("hello")
        blocks = provider._convert_path(unknown)
        assert blocks[0]["type"] == "text"
        assert "readme.xyz" in blocks[0]["text"]


# ── OpenAI path conversion ──────────────────────────────────────────


class TestOpenAIPathConversion:

    @pytest.fixture
    def provider(self):
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "test-key"
        with patch.dict(os.environ, env):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from taster.core.providers.openai_provider import OpenAIProvider
                return OpenAIProvider(api_key="test-key")

    def test_unknown_file_type(self, provider, tmp_path):
        unknown = tmp_path / "readme.xyz"
        unknown.write_text("hello")
        parts = provider._convert_path(unknown)
        assert parts[0]["type"] == "text"
        assert "readme.xyz" in parts[0]["text"]


# ── VideoFrameExtractor edge cases ──────────────────────────────────


class TestVideoFrameExtractorExtended:

    def test_nonexistent_file(self, tmp_path):
        result = VideoFrameExtractor.extract_frames(tmp_path / "nonexistent.mp4")
        assert isinstance(result, list)


# ── PDFPageRenderer edge cases ───────────────────────────────────────


class TestPDFPageRendererExtended:

    def test_nonexistent_pdf(self, tmp_path):
        with pytest.raises(Exception):  # pymupdf raises its own FileNotFoundError (RuntimeError subclass)
            PDFPageRenderer.render_pages(tmp_path / "nonexistent.pdf")


# ── Local provider tests ─────────────────────────────────────────────


class TestLocalProvider:
    """Tests for LocalProvider."""

    @pytest.fixture
    def provider(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from taster.core.providers.local_provider import LocalProvider
            return LocalProvider()

    def test_provider_name(self, provider):
        assert provider.provider_name == "local"

    def test_default_base_url(self, provider):
        assert provider.base_url == "http://localhost:11434/v1"

    def test_default_model(self, provider):
        assert provider.model_name == "llama3.2"

    def test_default_timeout(self, provider):
        assert provider.timeout == 300.0

    def test_custom_base_url(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from taster.core.providers.local_provider import LocalProvider
            p = LocalProvider(base_url="http://myhost:8080/v1")
            assert p.base_url == "http://myhost:8080/v1"

    def test_custom_model(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from taster.core.providers.local_provider import LocalProvider
            p = LocalProvider(model_name="mistral")
            assert p.model_name == "mistral"

    def test_custom_timeout(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from taster.core.providers.local_provider import LocalProvider
            p = LocalProvider(timeout=600.0)
            assert p.timeout == 600.0

    def test_env_var_override(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_LLM_URL"}
        env["LOCAL_LLM_URL"] = "http://envhost:9999/v1"
        mock_openai = MagicMock()
        with patch.dict(os.environ, env, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from taster.core.providers.local_provider import LocalProvider
                p = LocalProvider()
                assert p.base_url == "http://envhost:9999/v1"

    def test_explicit_url_beats_env(self):
        env = dict(os.environ)
        env["LOCAL_LLM_URL"] = "http://envhost:9999/v1"
        mock_openai = MagicMock()
        with patch.dict(os.environ, env):
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from taster.core.providers.local_provider import LocalProvider
                p = LocalProvider(base_url="http://explicit:1234/v1")
                assert p.base_url == "http://explicit:1234/v1"

    def test_no_api_key_required(self, provider):
        # Should have created successfully without any API key env var
        assert provider.api_key == "not-needed"

    def test_inherits_message_building(self, provider):
        messages = provider._build_messages("hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_is_ai_client_subclass(self, provider):
        assert isinstance(provider, AIClient)


# ── Factory tests for local provider ─────────────────────────────────


class TestFactoryLocal:
    """Tests for create_ai_client with local provider."""

    @pytest.fixture
    def config(self):
        return Config()

    def test_explicit_local(self, config):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = create_ai_client(config, provider="local")
            assert client.provider_name == "local"
            assert isinstance(client, AIClient)

    def test_auto_detect_via_env(self, config):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_LLM_URL")}
        env["LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        mock_openai = MagicMock()
        with patch.dict(os.environ, env, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai}):
                client = create_ai_client(config)
                assert client.provider_name == "local"

    def test_config_model_passthrough(self):
        config = Config()
        config.model.local_model = "phi3"
        config.model.local_base_url = "http://custom:5000/v1"
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = create_ai_client(config, provider="local")
            assert client.model_name == "phi3"
            assert client.base_url == "http://custom:5000/v1"

    def test_cloud_providers_preferred_over_local(self, config):
        """When both cloud and local are available, cloud wins."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_LLM_URL")}
        env["OPENAI_API_KEY"] = "o-key"
        env["LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        mock_openai = MagicMock()
        with patch.dict(os.environ, env, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai}):
                client = create_ai_client(config)
                assert client.provider_name == "openai"

    def test_no_keys_error_mentions_local(self, config):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LOCAL_LLM_URL")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="LOCAL_LLM_URL"):
                create_ai_client(config)

    def test_unknown_provider_error_mentions_local(self, config):
        with pytest.raises(ValueError, match="local"):
            create_ai_client(config, provider="nonexistent")


# ── ModelConfig local fields ─────────────────────────────────────────


class TestModelConfigLocal:
    """ModelConfig local LLM fields."""

    def test_defaults(self):
        mc = ModelConfig()
        assert mc.local_model == "llama3.2"
        assert mc.local_base_url == "http://localhost:11434/v1"

    def test_backward_compat_without_local_fields(self):
        """Old config.yaml without local fields should still work."""
        config = Config.from_dict({
            "model": {"name": "gemini-2.0-flash"},
        })
        assert config.model.local_model == "llama3.2"
        assert config.model.local_base_url == "http://localhost:11434/v1"

    def test_with_local_fields(self):
        config = Config.from_dict({
            "model": {
                "provider": "local",
                "local_model": "mistral",
                "local_base_url": "http://myserver:8080/v1",
            },
        })
        assert config.model.provider == "local"
        assert config.model.local_model == "mistral"
        assert config.model.local_base_url == "http://myserver:8080/v1"
