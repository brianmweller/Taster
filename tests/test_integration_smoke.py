"""Smoke tests to verify refactored infrastructure works together."""
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from taster.core import load_config, CacheManager
from taster.features import QualityScorer, BurstDetector


class TestInfrastructureIntegration:
    """Test that all refactored components work together."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample test images."""
        images = []
        for i in range(5):
            img_path = tmp_path / f"test{i}.jpg"
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img.save(img_path)
            images.append(img_path)
        return images

    def test_config_loading(self):
        """Test configuration loads successfully."""
        config = load_config(Path("config.yaml"))

        assert config.model.name == "gemini-3.8-flash"
        assert config.classification.classify_videos is True
        assert config.classification.parallel_video_workers == 2

    def test_cache_manager_initialization(self, tmp_path):
        """Test cache manager initializes correctly."""
        cache = CacheManager(tmp_path / "cache", enabled=True)

        assert cache.enabled
        assert "embeddings" in cache.caches
        assert "quality" in cache.caches

    def test_quality_scorer_with_real_images(self, sample_images):
        """Test quality scorer on real images."""
        config = load_config(Path("config.yaml"))
        cache = CacheManager(Path(". taste_cache_test"), enabled=False)

        scorer = QualityScorer(config.quality, cache)

        # Score single image
        score = scorer.compute_score(sample_images[0], use_cache=False)
        assert 0.0 <= score <= 1.0

        # Score batch
        scores = scorer.compute_scores_batch(
            sample_images,
            use_multiprocessing=False,
            use_cache=False
        )
        assert len(scores) == len(sample_images)
        for path, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_burst_detector_with_embeddings(self, sample_images):
        """Test burst detector with mock embeddings."""
        config = load_config(Path("config.yaml"))
        detector = BurstDetector(config.burst_detection)

        # Create mock embeddings (normalized random vectors)
        embeddings = np.random.rand(len(sample_images), 512)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Detect bursts
        bursts = detector.detect_bursts(sample_images, embeddings)

        # Should return all photos (either as bursts or singletons)
        total_photos = sum(len(burst) for burst in bursts)
        assert total_photos == len(sample_images)

    def test_end_to_end_pipeline(self, sample_images):
        """Test complete pipeline: config → cache → quality → bursts."""
        # Load config
        config = load_config(Path("config.yaml"))

        # Initialize cache
        cache = CacheManager(Path(".taste_cache_test"), enabled=False)

        # Initialize components
        scorer = QualityScorer(config.quality, cache)
        detector = BurstDetector(config.burst_detection)

        # 1. Compute quality scores
        quality_scores = scorer.compute_scores_batch(
            sample_images,
            use_multiprocessing=False,
            use_cache=False
        )
        assert len(quality_scores) == len(sample_images)

        # 2. Create mock embeddings
        embeddings = np.random.rand(len(sample_images), 512)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 3. Detect bursts
        bursts = detector.detect_bursts(sample_images, embeddings)

        # Verify pipeline completed successfully
        assert len(bursts) > 0
        total_photos = sum(len(burst) for burst in bursts)
        assert total_photos == len(sample_images)


def test_imports():
    """Test that all new modules can be imported."""
    # Core
    from taster.core import Config, load_config, CacheManager, CacheKey
    from taster.core import FileTypeRegistry, ImageUtils
    from taster.core import GeminiClient, initialize_gemini

    # Features
    from taster.features import QualityScorer, FaceDetector
    from taster.features import BurstDetector, EmbeddingExtractor

    # All imports successful
    assert True
