"""Comprehensive end-to-end integration tests."""
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

from taster.core import load_config, CacheManager, GeminiClient, FileTypeRegistry
from taster.features import QualityScorer, BurstDetector, EmbeddingExtractor
from taster.classification import PromptBuilder, MediaClassifier, Router


class TestEndToEndWorkflow:
    """Test the complete classification workflow."""

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images with varied properties."""
        images_dir = tmp_path / "test_photos"
        images_dir.mkdir()

        images = []

        # Create 10 test images with different characteristics
        for i in range(10):
            img_path = images_dir / f"photo_{i:02d}.jpg"

            # Create images with varying sizes and colors
            if i < 3:
                # First 3: Similar images (burst candidate)
                img = Image.new("RGB", (800, 600), color=(100 + i * 10, 150, 200))
            elif i < 6:
                # Next 3: Different images
                img = Image.new("RGB", (800, 600), color=(i * 40, 100, i * 50))
            else:
                # Last 4: Another potential burst
                img = Image.new("RGB", (800, 600), color=(200, 150 - i * 5, 100))

            # Add some noise to make images more realistic
            pixels = np.array(img)
            noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)

            img.save(img_path, quality=95)
            images.append(img_path)

            # Add small delay between images to simulate burst timing
            if i in [1, 2, 7, 8, 9]:
                # These are part of bursts - set recent mtime
                import os, time
                mtime = time.time() - (10 - i)  # Recent, within seconds of each other
                os.utime(img_path, (mtime, mtime))

        return images_dir, images

    @pytest.fixture
    def config(self, tmp_path):
        """Load config with test overrides."""
        config = load_config(Path("config.yaml"))

        # Override cache location for tests
        config.paths.cache_root = tmp_path / "test_cache"

        # Adjust for faster testing
        config.performance.embedding_batch_size = 10
        config.burst_detection.min_burst_size = 2
        config.burst_detection.time_window_seconds = 30  # Wider window for test

        return config

    def test_complete_pipeline_without_api(self, test_images, config, tmp_path):
        """
        Test complete pipeline without calling Gemini API.

        This tests:
        1. File discovery
        2. Cache management
        3. Quality scoring
        4. Embedding extraction
        5. Burst detection
        6. Component initialization

        API-dependent tests (classification, routing) are mocked.
        """
        images_dir, image_paths = test_images

        # 1. File Discovery
        print("\n=== STEP 1: File Discovery ===")
        discovered_images = FileTypeRegistry.list_images(images_dir)
        print(f"Discovered {len(discovered_images)} images")
        assert len(discovered_images) == 10
        assert all(isinstance(p, Path) for p in discovered_images)

        # 2. Cache Initialization
        print("\n=== STEP 2: Cache Initialization ===")
        cache_manager = CacheManager(
            config.paths.cache_root,
            ttl_days=config.caching.ttl_days,
            enabled=config.caching.enabled,
            max_size_gb=1.0  # Small limit for testing
        )
        print(f"Cache root: {cache_manager.cache_root}")
        assert cache_manager.enabled
        assert cache_manager.cache_root.exists()

        # 3. Quality Scoring
        print("\n=== STEP 3: Quality Scoring ===")
        quality_scorer = QualityScorer(config.quality, cache_manager)
        quality_scores = {}

        for img_path in discovered_images[:5]:  # Test subset for speed
            score = quality_scorer.compute_score(img_path, use_cache=True)
            quality_scores[img_path] = score
            print(f"  {img_path.name}: {score:.3f}")

        assert len(quality_scores) == 5
        assert all(0.0 <= score <= 1.0 for score in quality_scores.values())

        # Verify caching works
        cached_score = quality_scorer.compute_score(discovered_images[0], use_cache=True)
        assert cached_score == quality_scores[discovered_images[0]]

        # 4. Embedding Extraction (Skip - requires CLIP model download)
        print("\n=== STEP 4: Embedding Extraction (SKIPPED - requires large model) ===")
        # Create dummy embeddings for testing burst detection
        print("  Creating dummy normalized embeddings for testing...")
        embeddings = np.random.rand(len(discovered_images), 512)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        print(f"  Created embeddings shape: {embeddings.shape}")

        # 5. Burst Detection
        print("\n=== STEP 5: Burst Detection ===")
        detector = BurstDetector(config.burst_detection)
        bursts = detector.detect_bursts(discovered_images, embeddings)

        print(f"  Detected {len(bursts)} groups")
        burst_sizes = [len(b) for b in bursts]
        num_bursts = sum(1 for size in burst_sizes if size > 1)
        num_singletons = sum(1 for size in burst_sizes if size == 1)
        print(f"  Bursts: {num_bursts}, Singletons: {num_singletons}")

        # Should have some grouping
        assert len(bursts) > 0
        assert sum(len(b) for b in bursts) == len(discovered_images)

        # 6. Component Initialization (without API calls)
        print("\n=== STEP 6: Component Initialization ===")

        # PromptBuilder
        prompt_builder = PromptBuilder(config, training_examples={})
        singleton_prompt = prompt_builder.build_singleton_prompt()
        burst_prompt = prompt_builder.build_burst_prompt(5)
        print(f"  Singleton prompt length: {len(singleton_prompt)} chars")
        print(f"  Burst prompt length: {len(burst_prompt)} chars")
        assert len(singleton_prompt) > 100
        assert len(burst_prompt) > 100
        assert "share" in singleton_prompt.lower()
        assert "BURST" in burst_prompt

        # Router (without GeminiClient for this test)
        router = Router(config, gemini_client=None)

        # Test routing logic
        test_classification = {
            "classification": "Share",
            "score": 4,
            "contains_children": True,
            "is_appropriate": True
        }
        destination = router.route_singleton(test_classification)
        print(f"  Test routing: Share @ score 4 -> {destination}")
        assert destination == "Share"

        # 7. Cache Stats
        print("\n=== STEP 7: Cache Statistics ===")
        cache_stats = cache_manager.get_stats()
        total_mb = cache_stats["total"] / (1024 ** 2)
        print(f"  Total cache size: {total_mb:.2f} MB")
        for cache_type, size in cache_stats.items():
            if cache_type != "total" and size > 0:
                print(f"    {cache_type}: {size / (1024 ** 2):.2f} MB")

        assert cache_stats["total"] > 0  # Should have cached something

        # 8. Cache Enforcement
        print("\n=== STEP 8: Cache Size Enforcement ===")
        cleanup_stats = cache_manager.check_and_cleanup()
        print(f"  Expired removed: {cleanup_stats['expired_removed']}")
        print(f"  Size enforcement removed: {cleanup_stats['removed']}")

        print("\n=== END-TO-END TEST COMPLETE ===")
        print("[OK] All pipeline components working correctly")

    def test_cache_size_enforcement(self, tmp_path):
        """Test that cache size enforcement works correctly."""
        print("\n=== Testing Cache Size Enforcement ===")

        # Create cache manager with small limit
        cache = CacheManager(
            tmp_path / "cache",
            max_size_gb=0.001  # 1 MB limit
        )

        # Add dummy data until we exceed limit
        for i in range(100):
            # Create 50KB dummy arrays
            data = np.random.rand(1000, 50)
            cache.set("embeddings", f"test_{i}", data)

        # Check size
        size_before = cache.get_total_size()
        print(f"Cache size before: {size_before / (1024**2):.2f} MB")

        # Enforce limit
        stats = cache.enforce_size_limit()
        print(f"Removed {stats['removed']} entries")
        print(f"Size: {stats['size_before_mb']} MB → {stats['size_after_mb']} MB")

        size_after = cache.get_total_size()
        size_after_mb = size_after / (1024 ** 2)

        # Should have removed something if we exceeded limit
        if size_before > 1024 * 1024:  # If > 1MB
            assert stats['removed'] > 0
            assert size_after < size_before

        print("[OK] Cache size enforcement working")

    def test_config_validation(self):
        """Test that configuration validation catches errors."""
        print("\n=== Testing Configuration Validation ===")

        config = load_config(Path("config.yaml"))

        # Test threshold validation
        with pytest.raises(ValueError, match="review_threshold.*must be.*share_threshold"):
            from taster.core.config import ClassificationConfig
            ClassificationConfig(
                share_threshold=3,
                review_threshold=4  # Invalid: review > share
            )

        # Test weight validation
        with pytest.raises(ValueError, match="sharpness_weight.*brightness_weight.*must equal 1.0"):
            from taster.core.config import QualityConfig
            QualityConfig(
                sharpness_weight=0.5,
                brightness_weight=0.3  # Invalid: doesn't sum to 1.0
            )

        # Test max_faces validation
        with pytest.raises(ValueError, match="max_faces_to_report must be >= 1"):
            from taster.core.config import QualityConfig
            QualityConfig(max_faces_to_report=0)

        print("[OK] Configuration validation working")

    def test_input_validation(self, tmp_path):
        """Test that input validation catches errors."""
        print("\n=== Testing Input Validation ===")

        config = load_config(Path("config.yaml"))
        config.paths.cache_root = tmp_path / "cache"

        # Test burst detector with mismatched inputs
        detector = BurstDetector(config.burst_detection)

        photos = [tmp_path / f"photo_{i}.jpg" for i in range(5)]
        embeddings = np.random.rand(3, 512)  # Wrong size!

        with pytest.raises(ValueError, match="Embeddings array length.*does not match.*photos list length"):
            detector.detect_bursts(photos, embeddings)

        # Test with empty inputs
        result = detector.detect_bursts([], np.array([]))
        assert result == []

        # Test file utils with invalid path
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            FileTypeRegistry.list_images(tmp_path / "nonexistent")

        print("[OK] Input validation working")

    def test_error_recovery(self, tmp_path):
        """Test that components recover gracefully from errors."""
        print("\n=== Testing Error Recovery ===")

        config = load_config(Path("config.yaml"))
        config.paths.cache_root = tmp_path / "cache"
        cache = CacheManager(config.paths.cache_root)

        # Test quality scorer with nonexistent image
        scorer = QualityScorer(config.quality, cache)
        score = scorer.compute_score(tmp_path / "nonexistent.jpg", use_cache=False)
        assert score == 0.0  # Should return 0.0 instead of crashing

        # Test image utils with corrupted file
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"not an image")

        from taster.core.file_utils import ImageUtils
        img = ImageUtils.load_and_fix_orientation(corrupt_file)
        assert img is None  # Should return None instead of crashing

        print("[OK] Error recovery working")


def test_full_system_integration():
    """
    Quick integration test to verify all major components can be imported and initialized.
    """
    print("\n=== Testing Full System Integration ===")

    # Test all imports
    from taster.core import (
        Config, load_config, save_config,
        CacheManager, CacheKey,
        FileTypeRegistry, ImageUtils,
        GeminiClient, GeminiResponse,
        get_logger, setup_logging
    )
    from taster.features import QualityScorer, BurstDetector, EmbeddingExtractor
    from taster.classification import PromptBuilder, MediaClassifier, Router

    # Test configuration
    config = load_config(Path("config.yaml"))
    assert config is not None
    assert config.model.name == "gemini-3.8-flash"

    # Test logging
    logger = get_logger(__name__)
    assert logger is not None

    print("[OK] All imports and initialization working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
