"""Tests for configuration management."""
import pytest
import tempfile
from pathlib import Path
import yaml
from taster.core.config import (
    Config,
    ModelConfig,
    ClassificationConfig,
    BurstDetectionConfig,
    QualityConfig,
    load_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ModelConfig()
        assert config.name == "gemini-3.8-flash"
        assert config.clip_model == "ViT-B-32"

    def test_custom_values(self):
        """Test custom values."""
        config = ModelConfig(name="gemini-1.5-pro")
        assert config.name == "gemini-1.5-pro"


class TestClassificationConfig:
    """Tests for ClassificationConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ClassificationConfig()
        assert config.share_threshold == 4
        assert config.review_threshold == 3
        assert config.classify_videos is True
        assert config.parallel_video_workers == 10

    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid thresholds
        config = ClassificationConfig(share_threshold=5, review_threshold=3)
        assert config.share_threshold == 5

        # Invalid: share < review
        with pytest.raises(ValueError, match="review_threshold.*must be"):
            ClassificationConfig(share_threshold=2, review_threshold=4)

        # Invalid: out of range
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            ClassificationConfig(share_threshold=6)

    def test_parallel_workers_validation(self):
        """Test parallel workers validation."""
        with pytest.raises(ValueError, match="parallel_video_workers must be >= 1"):
            ClassificationConfig(parallel_video_workers=0)

    def test_parallel_photo_workers_validation(self):
        """Test parallel photo workers validation."""
        config = ClassificationConfig()
        assert config.parallel_photo_workers == 10
        with pytest.raises(ValueError, match="parallel_photo_workers must be >= 1"):
            ClassificationConfig(parallel_photo_workers=0)


class TestBurstDetectionConfig:
    """Tests for BurstDetectionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = BurstDetectionConfig()
        assert config.time_window_seconds == 10
        assert config.embedding_similarity_threshold == 0.92
        assert config.min_burst_size == 2

    def test_validation(self):
        """Test parameter validation."""
        # Invalid similarity threshold
        with pytest.raises(ValueError, match="embedding_similarity_threshold"):
            BurstDetectionConfig(embedding_similarity_threshold=1.5)

        # Invalid min burst size
        with pytest.raises(ValueError, match="min_burst_size must be >= 2"):
            BurstDetectionConfig(min_burst_size=1)

        # Invalid max < min
        with pytest.raises(ValueError, match="max_burst_size must be >= min_burst_size"):
            BurstDetectionConfig(min_burst_size=10, max_burst_size=5)


class TestQualityConfig:
    """Tests for QualityConfig."""

    def test_defaults(self):
        """Test default values."""
        config = QualityConfig()
        assert config.sharpness_weight == 0.8
        assert config.brightness_weight == 0.2
        assert config.face_detection_method == "mediapipe"

    def test_weight_validation(self):
        """Test that weights sum to 1.0."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            QualityConfig(sharpness_weight=0.5, brightness_weight=0.4)

    def test_face_detection_validation(self):
        """Test face detection method validation."""
        with pytest.raises(ValueError, match="face_detection_method"):
            QualityConfig(face_detection_method="invalid")


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_config(self, tmp_path):
        """Test loading config from YAML."""
        # Create test config
        config_data = {
            "model": {
                "name": "gemini-3.8-flash",
                "clip_model": "ViT-B-32",
            },
            "classification": {
                "share_threshold": 5,
                "review_threshold": 3,
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(config_file)

        assert config.model.name == "gemini-3.8-flash"
        assert config.classification.share_threshold == 5
        assert config.classification.review_threshold == 3

    def test_load_nonexistent_config(self):
        """Test loading non-existent config."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_partial_config(self, tmp_path):
        """Test loading partial config with defaults."""
        # Only specify model config
        config_data = {
            "model": {
                "name": "gemini-1.5-pro",
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        # Custom value
        assert config.model.name == "gemini-1.5-pro"

        # Defaults
        assert config.classification.share_threshold == 4
        assert config.burst_detection.time_window_seconds == 10


@pytest.fixture
def sample_config():
    """Create a sample config for testing."""
    return Config()


class TestConfig:
    """Tests for master Config class."""

    def test_defaults(self, sample_config):
        """Test default config creation."""
        assert sample_config.model.name == "gemini-3.8-flash"
        assert sample_config.classification.share_threshold == 4
        assert sample_config.burst_detection.enabled is True

    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            "model": {"name": "test-model"},
            "classification": {"share_threshold": 5},
        }

        config = Config.from_dict(data)
        assert config.model.name == "test-model"
        assert config.classification.share_threshold == 5
