"""Configuration management with YAML loading and validation."""
import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gemini-3-flash-preview"
    max_output_tokens: int = 4096
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    # Multi-provider support
    provider: Optional[str] = None  # "gemini"/"openai"/"anthropic"/"local"/None=auto
    openai_model: str = "gpt-5.2"
    anthropic_model: str = "claude-sonnet-4-6"
    local_model: str = "llama3.2"
    local_base_url: str = "http://localhost:11434/v1"
    video_frame_count: int = 8
    pdf_render_dpi: int = 150


@dataclass
class PathConfig:
    """Path configuration."""
    root: str = ""
    family_photos: str = ""
    private_photos: str = ""
    holding_cell: str = ""
    camera_roll: str = ""
    cache_root: str = ".taste_cache"
    taste_preferences: str = "taste_preferences.json"
    taste_preferences_generated: str = "taste_preferences_generated.json"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.cache_root = Path(self.cache_root)
        self.taste_preferences = Path(self.taste_preferences)
        self.taste_preferences_generated = Path(self.taste_preferences_generated)


@dataclass
class FileTypeConfig:
    """File type configuration."""
    image_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"
    ])
    video_extensions: List[str] = field(default_factory=lambda: [
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv", ".flv", ".webm"
    ])
    audio_extensions: List[str] = field(default_factory=lambda: [
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus", ".aiff"
    ])

    def __post_init__(self):
        """Convert to sets for fast lookup."""
        self.image_extensions = set(ext.lower() for ext in self.image_extensions)
        self.video_extensions = set(ext.lower() for ext in self.video_extensions)
        self.audio_extensions = set(ext.lower() for ext in self.audio_extensions)


@dataclass
class ClassificationConfig:
    """Classification configuration."""
    classify_videos: bool = True
    classify_audio: bool = True
    parallel_video_workers: int = 10
    parallel_photo_workers: int = 10
    parallel_audio_workers: int = 10
    share_threshold: int = 4
    review_threshold: int = 3
    burst_rank_consider: int = 2
    burst_rank_review: int = 4
    target_share_rate: float = 0.275
    enable_diversity_check: bool = True
    diversity_similarity_threshold: float = 0.95
    handle_blocked_safely: bool = True
    # Retry settings for failed classifications
    classification_retries: int = 2
    retry_delay_seconds: float = 2.0
    retry_on_errors: List[str] = field(default_factory=lambda: [
        "api_error", "invalid_response", "timeout", "rate_limit"
    ])

    def __post_init__(self):
        """Validate thresholds."""
        if not 1 <= self.share_threshold <= 5:
            raise ValueError(f"share_threshold must be between 1 and 5, got {self.share_threshold}")
        if not 1 <= self.review_threshold <= 5:
            raise ValueError(f"review_threshold must be between 1 and 5, got {self.review_threshold}")
        if self.review_threshold >= self.share_threshold:
            raise ValueError(f"review_threshold ({self.review_threshold}) must be < share_threshold ({self.share_threshold})")
        if self.parallel_video_workers < 1:
            raise ValueError(f"parallel_video_workers must be >= 1, got {self.parallel_video_workers}")
        if self.parallel_photo_workers < 1:
            raise ValueError(f"parallel_photo_workers must be >= 1, got {self.parallel_photo_workers}")
        if self.classification_retries < 0:
            raise ValueError(f"classification_retries must be >= 0, got {self.classification_retries}")


@dataclass
class BurstDetectionConfig:
    """Burst detection configuration."""
    enabled: bool = True
    time_window_seconds: int = 10
    embedding_similarity_threshold: float = 0.92
    min_burst_size: int = 2
    max_burst_size: int = 50
    enable_chunking: bool = True
    chunk_size: int = 10

    def __post_init__(self):
        """Validate parameters."""
        if not 0.0 <= self.embedding_similarity_threshold <= 1.0:
            raise ValueError(f"embedding_similarity_threshold must be between 0 and 1, got {self.embedding_similarity_threshold}")
        if self.min_burst_size < 2:
            raise ValueError(f"min_burst_size must be >= 2, got {self.min_burst_size}")
        if self.max_burst_size < self.min_burst_size:
            raise ValueError(f"max_burst_size must be >= min_burst_size")


@dataclass
class DedupConfig:
    """Perceptual hash (phash) duplicate detection configuration."""
    enabled: bool = True
    max_hamming_distance: int = 4
    hash_size: int = 16

    def __post_init__(self):
        """Validate parameters."""
        if self.max_hamming_distance < 0:
            raise ValueError(f"max_hamming_distance must be >= 0, got {self.max_hamming_distance}")
        if self.hash_size < 8:
            raise ValueError(f"hash_size must be >= 8, got {self.hash_size}")


@dataclass
class QualityConfig:
    """Quality scoring configuration."""
    sharpness_weight: float = 0.8
    brightness_weight: float = 0.2
    sharpness_threshold: float = 200.0
    quality_filter_threshold: float = 0.3
    enable_face_detection: bool = True
    face_detection_method: str = "mediapipe"
    max_faces_to_report: int = 10

    def __post_init__(self):
        """Validate parameters."""
        if abs(self.sharpness_weight + self.brightness_weight - 1.0) > 0.01:
            raise ValueError(f"sharpness_weight + brightness_weight must equal 1.0, got {self.sharpness_weight + self.brightness_weight}")
        if self.face_detection_method not in ["mediapipe", "face_recognition"]:
            raise ValueError(f"face_detection_method must be 'mediapipe' or 'face_recognition', got {self.face_detection_method}")
        if self.max_faces_to_report < 1:
            raise ValueError(f"max_faces_to_report must be >= 1, got {self.max_faces_to_report}")


@dataclass
class CachingConfig:
    """Caching configuration."""
    enabled: bool = True
    cache_embeddings: bool = True
    cache_quality_scores: bool = True
    cache_features: bool = True
    cache_faces: bool = True
    cache_gemini_responses: bool = True
    force_recompute: bool = False
    ttl_days: int = 365
    max_cache_size_gb: float = 10.0


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    embedding_batch_size: int = 128
    quality_batch_size: int = 64
    classification_batch_size: int = 32
    use_multiprocessing: bool = True
    quality_workers: int = 4
    dedup_workers: int = 4
    device: str = "cuda"

    def __post_init__(self):
        """Validate parameters."""
        if self.embedding_batch_size < 1:
            raise ValueError(f"embedding_batch_size must be >= 1, got {self.embedding_batch_size}")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")


@dataclass
class TrainingConfig:
    """Training configuration."""
    validation_split: float = 0.2
    random_seed: int = 1337
    max_negative_per_positive: int = 3
    balance_method: str = "undersample"
    within_cluster_weight: float = 2.0
    between_cluster_weight: float = 1.0
    use_camera_roll_negatives: bool = True
    date_range_days: int = 30
    # Pairwise training settings
    pairwise_within_ratio: float = 0.5
    pairwise_between_ratio: float = 0.3
    pairwise_gallery_ratio: float = 0.2
    gallery_weight: float = 1.5
    large_burst_threshold: int = 5
    min_labels_before_synthesis: int = 15


@dataclass
class CostConfig:
    """Cost estimation configuration."""
    per_1m_input_tokens: float = 0.50
    per_1m_output_tokens: float = 3.00
    per_photo: float = 0.0013
    per_burst_photo: float = 0.00043
    tokens_per_video_second: int = 300
    prompt_tokens_video: int = 1500
    output_tokens_video: int = 300
    tokens_per_audio_second: int = 32
    prompt_tokens_audio: int = 1500
    output_tokens_audio: int = 300


@dataclass
class SystemConfig:
    """System configuration."""
    dry_run: bool = False
    verbose: bool = True
    show_progress: bool = True
    checkpoint_every_n: int = 50
    continue_on_error: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


@dataclass
class DocumentConfig:
    """Document processing configuration."""
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".xlsx", ".pptx", ".html", ".htm", ".txt", ".md", ".csv", ".rtf"
    ])
    max_file_size_mb: float = 50.0
    max_pages_per_document: int = 20
    text_extraction_max_chars: int = 50000
    enable_text_embeddings: bool = True
    text_embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85

    def __post_init__(self):
        """Convert to set for fast lookup."""
        self.supported_extensions = set(ext.lower() for ext in self.supported_extensions)


@dataclass
class ProfileConfig:
    """Profile management configuration."""
    profiles_dir: str = "profiles"
    active_profile: str = "default-photos"
    auto_detect_media_type: bool = True


@dataclass
class Config:
    """Master configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    file_types: FileTypeConfig = field(default_factory=FileTypeConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    burst_detection: BurstDetectionConfig = field(default_factory=BurstDetectionConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    profiles: ProfileConfig = field(default_factory=ProfileConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            paths=PathConfig(**data.get("paths", {})),
            file_types=FileTypeConfig(**data.get("file_types", {})),
            classification=ClassificationConfig(**data.get("classification", {})),
            burst_detection=BurstDetectionConfig(**data.get("burst_detection", {})),
            dedup=DedupConfig(**data.get("dedup", {})),
            quality=QualityConfig(**data.get("quality", {})),
            caching=CachingConfig(**data.get("caching", {})),
            performance=PerformanceConfig(**data.get("performance", {})),
            training=TrainingConfig(**data.get("training", {})),
            cost=CostConfig(**data.get("cost", {})),
            system=SystemConfig(**data.get("system", {})),
            document=DocumentConfig(**data.get("document", {})),
            profiles=ProfileConfig(**data.get("profiles", {})),
        )


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.

    Resolution order:
        1. Explicit *config_path* (if provided and exists)
        2. ``./config.yaml`` (project-local, for development)
        3. User config dir ``config.yaml`` (platform-aware)
        4. Built-in defaults (no file needed)

    Args:
        config_path: Explicit path to config.yaml file.

    Returns:
        Config object with validated settings.
    """
    from taster.dirs import find_config

    resolved = find_config(config_path)

    if resolved is None:
        return Config()

    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def save_config(config: Config, config_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save.
        config_path: Path to save config.yaml file.
    """
    # Convert config to dictionary using asdict
    def _convert_paths(obj):
        """Convert Path objects to strings for YAML serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: _convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_paths(item) for item in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))  # Convert sets to sorted lists
        return obj

    # Convert dataclasses to dict
    data = {
        "model": asdict(config.model),
        "paths": asdict(config.paths),
        "file_types": asdict(config.file_types),
        "classification": asdict(config.classification),
        "burst_detection": asdict(config.burst_detection),
        "dedup": asdict(config.dedup),
        "quality": asdict(config.quality),
        "caching": asdict(config.caching),
        "performance": asdict(config.performance),
        "training": asdict(config.training),
        "cost": asdict(config.cost),
        "system": asdict(config.system),
        "document": asdict(config.document),
        "profiles": asdict(config.profiles),
    }

    # Convert Path objects and sets to YAML-serializable formats
    data = _convert_paths(data)

    # Write to file
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
