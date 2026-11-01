"""Tests for fail-fast handling: fatal API errors, circuit breaker, preflight, error routing."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from taster.core.config import load_config
from taster.core.ai_client import FatalAPIError, ClassificationAborted, is_fatal_api_error
from taster.classification.classifier import MediaClassifier
from taster.classification.prompt_builder import PromptBuilder
from taster.classification.routing import Router
from taster.pipelines import MixedPipeline


CREDITS_DEPLETED = (
    "429 Your prepayment credits are depleted. Please go to AI Studio at "
    "https://ai.studio/projects to manage your project and billing."
)


@pytest.fixture
def config():
    return load_config(Path("config.yaml"))


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    return cache


@pytest.fixture
def classifier(config, mock_client, mock_cache):
    config.classification.retry_delay_seconds = 0
    return MediaClassifier(config, mock_client, PromptBuilder(config), cache_manager=mock_cache)


# ── Fatal error detection ───────────────────────────────────────────


class TestIsFatalApiError:

    @pytest.mark.parametrize("msg", [
        CREDITS_DEPLETED,
        "400 API key not valid. Please pass a valid API key. [reason: \"API_KEY_INVALID\"]",
        "404 models/gemini-9 is not found for API version v1beta",
        "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
        "Your credit balance is too low to access the Anthropic API.",
        "403 PERMISSION_DENIED",
    ])
    def test_fatal(self, msg):
        assert is_fatal_api_error(msg)

    @pytest.mark.parametrize("msg", [
        "429 Resource has been exhausted (e.g. check quota).",
        "503 The service is currently unavailable.",
        "Request timed out",
        "Could not parse JSON from response",
    ])
    def test_not_fatal(self, msg):
        assert not is_fatal_api_error(msg)

    def test_fatal_exception_instance(self):
        assert is_fatal_api_error(FatalAPIError("anything"))


# ── Classifier propagation and caching ──────────────────────────────


class TestClassifierFatalErrors:

    @patch("taster.classification.classifier.ImageUtils")
    def test_fatal_error_propagates_without_caching(self, mock_utils, classifier, mock_client, mock_cache):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.side_effect = FatalAPIError(CREDITS_DEPLETED)
        with pytest.raises(FatalAPIError):
            classifier.classify_singleton(Path("a.jpg"))
        assert mock_client.generate_json.call_count == 1  # no retries
        mock_cache.set.assert_not_called()

    @patch("taster.classification.classifier.ImageUtils")
    def test_fatal_error_propagates_from_burst(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.side_effect = FatalAPIError(CREDITS_DEPLETED)
        with pytest.raises(FatalAPIError):
            classifier.classify_burst([Path("a.jpg"), Path("b.jpg")])

    def test_fatal_error_propagates_from_video(self, classifier, mock_client):
        mock_client.generate_json.side_effect = FatalAPIError(CREDITS_DEPLETED)
        with pytest.raises(FatalAPIError):
            classifier.classify_video(Path("a.mp4"))

    @patch("taster.classification.classifier.ImageUtils")
    def test_transient_error_not_cached(self, mock_utils, classifier, mock_client, mock_cache):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.side_effect = RuntimeError("503 unavailable")
        result = classifier.classify_singleton(Path("a.jpg"))
        assert result["is_error_fallback"]
        mock_cache.set.assert_not_called()

    @patch("taster.classification.classifier.ImageUtils")
    def test_success_is_cached(self, mock_utils, classifier, mock_client, mock_cache):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 5, "reasoning": "great",
            "contains_children": True, "is_appropriate": True,
        }
        classifier.classify_singleton(Path("a.jpg"))
        mock_cache.set.assert_called_once()


# ── Circuit breaker ─────────────────────────────────────────────────


class TestCircuitBreaker:

    def _error(self, classifier):
        return classifier._create_fallback_response("Error: boom", error_type="api_error")

    def test_trips_when_window_mostly_errors(self, config, mock_client):
        config.classification.circuit_breaker_window = 4
        config.classification.circuit_breaker_error_rate = 0.5
        c = MediaClassifier(config, mock_client, PromptBuilder(config))
        c._record_outcome({"classification": "Share"})
        c._record_outcome({"classification": "Share"})
        c._record_outcome(self._error(c))
        with pytest.raises(ClassificationAborted, match="2 of the last 4"):
            c._record_outcome(self._error(c))

    def test_does_not_trip_before_window_full(self, config, mock_client):
        config.classification.circuit_breaker_window = 4
        c = MediaClassifier(config, mock_client, PromptBuilder(config))
        for _ in range(3):
            c._record_outcome(self._error(c))

    def test_load_errors_do_not_count(self, config, mock_client):
        config.classification.circuit_breaker_window = 2
        c = MediaClassifier(config, mock_client, PromptBuilder(config))
        for _ in range(5):
            c._record_outcome(c._create_fallback_response("bad file", error_type="load_error"))

    def test_disabled_with_zero_rate(self, config, mock_client):
        config.classification.circuit_breaker_window = 2
        config.classification.circuit_breaker_error_rate = 0
        c = MediaClassifier(config, mock_client, PromptBuilder(config))
        for _ in range(5):
            c._record_outcome(self._error(c))

    def test_aborted_is_fatal(self):
        assert issubclass(ClassificationAborted, FatalAPIError)


# ── Error routing ───────────────────────────────────────────────────


class TestErrorRouting:

    def _fallback(self, config, mock_client):
        c = MediaClassifier(config, mock_client, PromptBuilder(config))
        return c._create_fallback_response("Error: boom", error_type="api_error")

    def test_error_fallback_goes_to_errors(self, config, mock_client):
        router = Router(config)
        fb = self._fallback(config, mock_client)
        assert router.route_singleton(fb) == "Errors"
        assert router.route_video(fb) == "Errors"
        assert router.route_document(fb) == "Errors"

    def test_error_fallback_in_burst_goes_to_errors(self, config, mock_client):
        config.classification.enable_diversity_check = False
        router = Router(config)
        fb = self._fallback(config, mock_client)
        ok = {"classification": "Storage", "score": 2, "rank": 1,
              "contains_children": True, "is_appropriate": True}
        assert router.route_burst([Path("a.jpg"), Path("b.jpg")], [ok, fb]) == ["Storage", "Errors"]

    def test_error_destination_none_keeps_default(self, config, mock_client):
        config.classification.error_destination = None
        router = Router(config)
        assert router.route_singleton(self._fallback(config, mock_client)) == "Review"


# ── Preflight ───────────────────────────────────────────────────────


class TestPreflight:

    def test_pipeline_aborts_before_processing(self, config, tmp_path):
        client = MagicMock()
        client.preflight.side_effect = FatalAPIError(CREDITS_DEPLETED)
        (tmp_path / "a.jpg").write_bytes(b"not really a jpg")
        pipeline = MixedPipeline(config, None, None, client)
        with patch("taster.pipelines.mixed_pipeline.PhotoPipeline") as photo:
            with pytest.raises(FatalAPIError):
                pipeline.run(tmp_path, tmp_path / "out")
            photo.assert_not_called()
        assert not (tmp_path / "out").exists()

    def test_preflight_can_be_disabled(self, config, tmp_path):
        config.classification.preflight_check = False
        client = MagicMock()
        pipeline = MixedPipeline(config, None, None, client)
        pipeline.run(tmp_path, tmp_path / "out")
        client.preflight.assert_not_called()

    def test_client_preflight_converts_fatal_errors(self):
        from taster.core.ai_client import AIClient

        class Dummy(AIClient):
            def generate(self, *a, **k):
                raise RuntimeError(CREDITS_DEPLETED)

            def generate_json(self, *a, **k):
                return {}

        with pytest.raises(FatalAPIError):
            Dummy().preflight()

    def test_client_preflight_tolerates_transient_errors(self):
        from taster.core.ai_client import AIClient

        class Dummy(AIClient):
            def generate(self, *a, **k):
                raise RuntimeError("503 unavailable")

            def generate_json(self, *a, **k):
                return {}

        Dummy().preflight()  # warns, does not raise


class TestCachedTransientErrors:

    @patch("taster.classification.classifier.ImageUtils")
    def test_cached_transient_fallback_is_ignored(self, mock_utils, classifier, mock_client, mock_cache):
        """Fallbacks cached by older versions (e.g. credits depleted) are re-classified."""
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_cache.get.return_value = classifier._create_fallback_response(
            f"Error: Gemini API error: {CREDITS_DEPLETED}", error_type="api_error"
        )
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 5, "reasoning": "great",
            "contains_children": True, "is_appropriate": True,
        }
        result = classifier.classify_singleton(Path("a.jpg"))
        assert mock_client.generate_json.called
        assert not result.get("is_error_fallback")

    def test_cached_success_is_used(self, classifier, mock_client, mock_cache):
        mock_cache.get.return_value = {"classification": "Share", "score": 5}
        assert classifier.classify_video(Path("a.mp4"))["score"] == 5
        mock_client.generate_json.assert_not_called()
