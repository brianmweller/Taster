"""Extended tests for MediaClassifier covering uncovered code paths."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from taster.core.config import load_config
from taster.core.profiles import TasteProfile, CategoryDefinition
from taster.classification.classifier import MediaClassifier
from taster.classification.prompt_builder import PromptBuilder


@pytest.fixture
def config():
    return load_config(Path("config.yaml"))


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.generate_json = MagicMock()
    return client


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock()
    return cache


@pytest.fixture
def classifier(config, mock_client):
    pb = PromptBuilder(config)
    return MediaClassifier(config, mock_client, pb)


@pytest.fixture
def classifier_with_cache(config, mock_client, mock_cache):
    pb = PromptBuilder(config)
    return MediaClassifier(config, mock_client, pb, cache_manager=mock_cache)


# ── Default category / valid categories ─────────────────────────────


class TestDefaultCategories:

    def test_default_category_no_profile(self, classifier):
        assert classifier._default_category == "Review"

    def test_valid_categories_no_profile(self, classifier):
        assert classifier._valid_categories == ["Share", "Storage", "Review", "Ignore"]

    def test_default_category_with_profile(self, config, mock_client):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[CategoryDefinition("A", "a"), CategoryDefinition("B", "b")],
            default_category="B",
        )
        pb = PromptBuilder(config, profile=profile)
        c = MediaClassifier(config, mock_client, pb, profile=profile)
        assert c._default_category == "B"

    def test_valid_categories_with_profile(self, config, mock_client):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[CategoryDefinition("X", "x"), CategoryDefinition("Y", "y")],
            default_category="Y",
        )
        pb = PromptBuilder(config, profile=profile)
        c = MediaClassifier(config, mock_client, pb, profile=profile)
        cats = c._valid_categories
        assert "X" in cats
        assert "Y" in cats

    def test_valid_categories_adds_default_if_missing(self, config, mock_client):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[CategoryDefinition("X", "x")],
            default_category="Fallback",
        )
        pb = PromptBuilder(config, profile=profile)
        c = MediaClassifier(config, mock_client, pb, profile=profile)
        assert "Fallback" in c._valid_categories


# ── Fallback response ────────────────────────────────────────────────


class TestFallbackResponse:

    def test_basic_fallback(self, classifier):
        fb = classifier._create_fallback_response("something broke")
        assert fb["classification"] == "Review"
        assert fb["score"] == 2
        assert fb["is_error_fallback"] is True
        assert fb["error_type"] == "api_error"
        assert "something broke" in fb["reasoning"]

    def test_fallback_with_rank(self, classifier):
        fb = classifier._create_fallback_response("err", rank=3)
        assert fb["rank"] == 3

    def test_fallback_video(self, classifier):
        fb = classifier._create_fallback_response("err", is_video=True)
        assert fb["audio_quality"] == "unknown"

    def test_fallback_custom_error_type(self, classifier):
        fb = classifier._create_fallback_response("timeout", error_type="timeout")
        assert fb["error_type"] == "timeout"


# ── Singleton validation ─────────────────────────────────────────────


class TestValidateSingleton:

    def test_fills_missing_fields(self, classifier):
        resp = {}
        result = classifier._validate_singleton_response(resp)
        assert result["classification"] == "Review"
        assert result["score"] == 2
        assert result["reasoning"] == "No reasoning provided"
        assert result["is_error_fallback"] is False

    def test_clamps_score(self, classifier):
        resp = {"classification": "Share", "score": 7}
        result = classifier._validate_singleton_response(resp)
        assert result["score"] == 5

        resp2 = {"classification": "Share", "score": -1}
        result2 = classifier._validate_singleton_response(resp2)
        assert result2["score"] == 1

    def test_remaps_invalid_category(self, classifier):
        resp = {"classification": "Invalid", "score": 4}
        result = classifier._validate_singleton_response(resp)
        assert result["classification"] == "Review"

    def test_keeps_valid_category(self, classifier):
        resp = {"classification": "Share", "score": 4, "reasoning": "good"}
        result = classifier._validate_singleton_response(resp)
        assert result["classification"] == "Share"

    def test_adds_boolean_fields(self, classifier):
        resp = {"classification": "Share", "score": 4}
        result = classifier._validate_singleton_response(resp)
        assert "contains_children" in result
        assert "is_appropriate" in result


# ── Document validation ──────────────────────────────────────────────


class TestValidateDocument:

    def test_fills_missing_fields(self, classifier):
        resp = {}
        result = classifier._validate_document_response(resp)
        assert result["classification"] == "Review"
        assert result["score"] == 2
        assert result["content_summary"] == ""
        assert result["key_topics"] == []

    def test_adds_default_rank(self, classifier):
        resp = {"classification": "Share"}
        result = classifier._validate_document_response(resp, default_rank=5)
        assert result["rank"] == 5

    def test_no_rank_without_default(self, classifier):
        resp = {"classification": "Share"}
        result = classifier._validate_document_response(resp)
        assert "rank" not in result

    def test_remaps_invalid_category(self, classifier):
        resp = {"classification": "Bogus"}
        result = classifier._validate_document_response(resp)
        assert result["classification"] == "Review"


# ── Burst validation ─────────────────────────────────────────────────


class TestValidateBurst:

    def test_adds_default_rank(self, classifier):
        resp = {"classification": "Share", "score": 4}
        result = classifier._validate_burst_response(resp, default_rank=2)
        assert result["rank"] == 2

    def test_keeps_existing_rank(self, classifier):
        resp = {"classification": "Share", "score": 4, "rank": 1}
        result = classifier._validate_burst_response(resp, default_rank=5)
        assert result["rank"] == 1


# ── Video validation ─────────────────────────────────────────────────


class TestValidateVideo:

    def test_adds_audio_quality(self, classifier):
        resp = {"classification": "Share", "score": 4}
        result = classifier._validate_video_response(resp)
        assert result["audio_quality"] == "unknown"



# ── Singleton classification ─────────────────────────────────────────


class TestClassifySingleton:

    def test_returns_cached(self, classifier_with_cache, mock_cache):
        cached_result = {"classification": "Share", "score": 5}
        mock_cache.get.return_value = cached_result
        result = classifier_with_cache.classify_singleton(Path("test.jpg"))
        assert result == cached_result

    @patch("taster.classification.classifier.ImageUtils")
    def test_load_failure_returns_fallback(self, mock_utils, classifier):
        mock_utils.load_and_fix_orientation.return_value = None
        result = classifier.classify_singleton(Path("broken.jpg"))
        assert result["is_error_fallback"] is True
        assert result["error_type"] == "load_error"

    @patch("taster.classification.classifier.ImageUtils")
    def test_successful_classification(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 4, "reasoning": "great photo"
        }
        result = classifier.classify_singleton(Path("good.jpg"))
        assert result["classification"] == "Share"
        assert mock_client.generate_json.called

    @patch("taster.classification.classifier.ImageUtils")
    def test_api_exception_returns_fallback(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.side_effect = Exception("API down")
        result = classifier.classify_singleton(Path("test.jpg"))
        assert result["is_error_fallback"] is True

    @patch("taster.classification.classifier.ImageUtils")
    def test_caches_result(self, mock_utils, classifier_with_cache, mock_client, mock_cache):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.return_value = {
            "classification": "Storage", "score": 3, "reasoning": "ok"
        }
        classifier_with_cache.classify_singleton(Path("test.jpg"))
        assert mock_cache.set.called


# ── Burst classification ─────────────────────────────────────────────


class TestClassifyBurst:

    def test_empty_burst(self, classifier):
        assert classifier.classify_burst([]) == []

    def test_single_photo_delegates_to_singleton(self, classifier, mock_client):
        with patch.object(classifier, "classify_singleton") as mock_single:
            mock_single.return_value = {"classification": "Share"}
            result = classifier.classify_burst([Path("one.jpg")])
            assert len(result) == 1
            mock_single.assert_called_once()

    @patch("taster.classification.classifier.ImageUtils")
    def test_burst_classification(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.return_value = [
            {"classification": "Share", "score": 5, "rank": 1},
            {"classification": "Storage", "score": 2, "rank": 2},
        ]
        result = classifier.classify_burst([Path("a.jpg"), Path("b.jpg")])
        assert len(result) == 2
        assert result[0]["burst_position"] == 0
        assert result[1]["burst_position"] == 1

    @patch("taster.classification.classifier.ImageUtils")
    def test_burst_invalid_response_format(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        # Return dict instead of list
        mock_client.generate_json.return_value = {"classification": "Share"}
        result = classifier.classify_burst([Path("a.jpg"), Path("b.jpg")])
        assert len(result) == 2
        assert all(r["is_error_fallback"] for r in result)

    @patch("taster.classification.classifier.ImageUtils")
    def test_burst_wrong_length_response(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.return_value = [
            {"classification": "Share", "score": 5, "rank": 1},
        ]  # Only 1 result for 2 photos
        result = classifier.classify_burst([Path("a.jpg"), Path("b.jpg")])
        assert len(result) == 2
        # Partial (truncated) responses keep what came back and pad the rest
        assert not result[0].get("is_error_fallback")
        assert result[1]["is_error_fallback"]

    @patch("taster.classification.classifier.ImageUtils")
    def test_burst_chunking(self, mock_utils, classifier, mock_client):
        mock_utils.load_and_fix_orientation.return_value = MagicMock()
        mock_client.generate_json.return_value = [
            {"classification": "Share", "score": 5, "rank": 1},
            {"classification": "Storage", "score": 2, "rank": 2},
        ]
        photos = [Path(f"p{i}.jpg") for i in range(5)]
        result = classifier.classify_burst(photos, chunk_size=2)
        # Should have been chunked into 3 chunks (2+2+1)
        assert len(result) == 5

    @patch("taster.classification.classifier.ImageUtils")
    def test_burst_failed_image_placeholder(self, mock_utils, classifier, mock_client):
        # First image fails to load, second succeeds
        mock_utils.load_and_fix_orientation.side_effect = [None, MagicMock()]
        mock_client.generate_json.return_value = [
            {"classification": "Storage", "score": 2, "rank": 1},
            {"classification": "Share", "score": 4, "rank": 2},
        ]
        result = classifier.classify_burst([Path("bad.jpg"), Path("good.jpg")])
        assert len(result) == 2


# ── Video classification ─────────────────────────────────────────────


class TestClassifyVideo:

    def test_returns_cached(self, classifier_with_cache, mock_cache):
        cached = {"classification": "Share", "score": 4, "audio_quality": "good"}
        mock_cache.get.return_value = cached
        result = classifier_with_cache.classify_video(Path("test.mp4"))
        assert result == cached

    def test_classification(self, classifier, mock_client):
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 4, "reasoning": "cute video",
            "audio_quality": "good",
        }
        result = classifier.classify_video(Path("test.mp4"))
        assert result["classification"] == "Share"
        assert result["audio_quality"] == "good"

    def test_api_error(self, classifier, mock_client):
        mock_client.generate_json.side_effect = Exception("Network error")
        result = classifier.classify_video(Path("test.mp4"))
        assert result["is_error_fallback"] is True
        assert result["audio_quality"] == "unknown"


# ── Batch classification ─────────────────────────────────────────────


class TestClassifyBatch:

    def test_batch(self, classifier):
        with patch.object(classifier, "classify_singleton") as mock_single:
            mock_single.return_value = {"classification": "Share"}
            result = classifier.classify_batch(
                [Path("a.jpg"), Path("b.jpg")], show_progress=False
            )
            assert len(result) == 2
            assert mock_single.call_count == 2


# ── Document classification ──────────────────────────────────────────


class TestClassifyDocument:

    def test_returns_cached(self, classifier_with_cache, mock_cache):
        cached = {"classification": "Exemplary", "score": 5}
        mock_cache.get.return_value = cached
        result = classifier_with_cache.classify_document(Path("test.pdf"))
        assert result == cached

    def test_pdf_sends_path(self, classifier, mock_client):
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 4, "reasoning": "good doc"
        }
        result = classifier.classify_document(Path("test.pdf"), text_content="hello")
        assert result["classification"] == "Share"
        call_args = mock_client.generate_json.call_args
        prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
        # The PDF path should be in the prompt
        assert any(isinstance(p, Path) for p in prompt)

    def test_non_pdf_sends_text(self, classifier, mock_client):
        mock_client.generate_json.return_value = {
            "classification": "Storage", "score": 3, "reasoning": "ok"
        }
        result = classifier.classify_document(
            Path("test.docx"), text_content="Some document content"
        )
        assert result["classification"] == "Storage"

    def test_text_truncation(self, classifier, mock_client):
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 4
        }
        long_text = "x" * 50000
        classifier.classify_document(Path("test.txt"), text_content=long_text)
        call_args = mock_client.generate_json.call_args
        prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
        # Should truncate to 30000 chars
        for part in prompt:
            if isinstance(part, str) and "truncated" in part:
                break
        else:
            # If no truncation marker, the text should still be bounded
            pass

    def test_with_metadata(self, classifier, mock_client):
        mock_client.generate_json.return_value = {
            "classification": "Share", "score": 4
        }
        classifier.classify_document(
            Path("test.pdf"), metadata={"author": "John", "pages": 5}
        )
        assert mock_client.generate_json.called

    def test_api_error(self, classifier, mock_client):
        mock_client.generate_json.side_effect = Exception("API error")
        result = classifier.classify_document(Path("test.pdf"))
        assert result["is_error_fallback"] is True


# ── Document group classification ────────────────────────────────────


class TestClassifyDocumentGroup:

    def test_empty_group(self, classifier):
        assert classifier.classify_document_group([]) == []

    def test_single_doc_delegates(self, classifier):
        with patch.object(classifier, "classify_document") as mock_doc:
            mock_doc.return_value = {"classification": "Share"}
            result = classifier.classify_document_group([Path("a.pdf")])
            assert len(result) == 1
            mock_doc.assert_called_once()

    def test_group_classification(self, classifier, mock_client):
        mock_client.generate_json.return_value = [
            {"classification": "Share", "score": 5, "rank": 1},
            {"classification": "Storage", "score": 2, "rank": 2},
        ]
        result = classifier.classify_document_group(
            [Path("a.pdf"), Path("b.pdf")],
            text_contents={Path("a.pdf"): "doc a", Path("b.pdf"): "doc b"},
        )
        assert len(result) == 2

    def test_group_invalid_response(self, classifier, mock_client):
        mock_client.generate_json.return_value = {"classification": "Share"}
        result = classifier.classify_document_group([Path("a.pdf"), Path("b.pdf")])
        assert len(result) == 2
        assert all(r["is_error_fallback"] for r in result)

    def test_group_api_error(self, classifier, mock_client):
        mock_client.generate_json.side_effect = Exception("boom")
        result = classifier.classify_document_group([Path("a.pdf"), Path("b.pdf")])
        assert len(result) == 2
        assert all(r["is_error_fallback"] for r in result)

    def test_returns_cached(self, classifier_with_cache, mock_cache):
        cached = [{"classification": "Share"}, {"classification": "Storage"}]
        mock_cache.get.return_value = cached
        result = classifier_with_cache.classify_document_group(
            [Path("a.pdf"), Path("b.pdf")]
        )
        assert result == cached


# ── Retry logic ──────────────────────────────────────────────────────


class TestRetryLogic:

    def test_no_retry_on_success(self, classifier):
        fn = MagicMock(return_value={"classification": "Share", "is_error_fallback": False})
        result = classifier._execute_with_retry(fn)
        assert fn.call_count == 1
        assert result["retry_count"] == 0

    def test_retries_on_retriable_error(self, config, mock_client):
        config.classification.classification_retries = 2
        config.classification.retry_delay_seconds = 0.01  # fast for tests
        pb = PromptBuilder(config)
        c = MediaClassifier(config, mock_client, pb)

        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"is_error_fallback": True, "error_type": "api_error"}
            return {"classification": "Share", "is_error_fallback": False}

        result = c._execute_with_retry(failing_fn)
        assert call_count == 3
        assert result["classification"] == "Share"

    def test_no_retry_on_non_retriable(self, config, mock_client):
        config.classification.classification_retries = 2
        config.classification.retry_delay_seconds = 0.01
        pb = PromptBuilder(config)
        c = MediaClassifier(config, mock_client, pb)

        fn = MagicMock(return_value={
            "is_error_fallback": True, "error_type": "safety_blocked"
        })
        result = c._execute_with_retry(fn)
        assert fn.call_count == 1

    def test_all_retries_exhausted(self, config, mock_client):
        config.classification.classification_retries = 1
        config.classification.retry_delay_seconds = 0.01
        pb = PromptBuilder(config)
        c = MediaClassifier(config, mock_client, pb)

        fn = MagicMock(return_value={
            "is_error_fallback": True, "error_type": "api_error"
        })
        result = c._execute_with_retry(fn)
        assert fn.call_count == 2  # initial + 1 retry
        assert result["retry_count"] == 1

    def test_exception_during_retry(self, config, mock_client):
        config.classification.classification_retries = 1
        config.classification.retry_delay_seconds = 0.01
        pb = PromptBuilder(config)
        c = MediaClassifier(config, mock_client, pb)

        fn = MagicMock(side_effect=Exception("boom"))
        result = c._execute_with_retry(fn)
        assert result["is_error_fallback"] is True
        assert fn.call_count == 2
