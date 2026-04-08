"""Photo/video classification pipeline.

Extracts the core logic from taste_classify.py into a reusable pipeline class.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from ..core.config import Config
from ..core.cache import CacheManager
from ..core.ai_client import AIClient
from ..core.file_utils import FileTypeRegistry
from ..core.profiles import TasteProfile
from ..features.embeddings import EmbeddingExtractor
from ..features.burst_detector import BurstDetector
from ..features.dedup_detector import DuplicateDetector
from ..classification.prompt_builder import PromptBuilder
from ..classification.classifier import MediaClassifier
from ..classification.routing import Router
from .base import ClassificationPipeline, ClassificationResult


class PhotoPipeline(ClassificationPipeline):
    """Pipeline for classifying photos and videos."""

    def __init__(
        self,
        config: Config,
        profile: Optional[TasteProfile] = None,
        cache_manager: Optional[CacheManager] = None,
        gemini_client: Optional[AIClient] = None,
    ):
        super().__init__(config, profile)
        self.cache_manager = cache_manager
        self.gemini_client = gemini_client

    def collect_files(self, folder: Path) -> List[Path]:
        """Collect images from the folder."""
        return FileTypeRegistry.list_images(folder, recursive=False)

    def collect_videos(self, folder: Path) -> List[Path]:
        """Collect videos from the folder."""
        return FileTypeRegistry.list_videos(folder, recursive=False)

    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Extract CLIP embeddings for burst detection."""
        if not files:
            return {}

        print(f"\nStep 1: Extracting CLIP embeddings for {len(files)} images...")
        extractor = EmbeddingExtractor(
            self.config.model, self.config.performance, self.cache_manager
        )
        embeddings = extractor.extract_embeddings_batch(
            files, use_cache=True, show_progress=True
        )
        return {"embeddings": embeddings, "images": files}

    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Detect bursts using temporal + visual similarity."""
        if not files:
            return []

        print("\nStep 2: Detecting photo bursts...")
        embeddings = features.get("embeddings")
        detector = BurstDetector(self.config.burst_detection)
        return detector.detect_bursts(files, embeddings)

    def _classify_group(self, group, classifier):
        """Classify a single group (singleton or burst). Thread-safe."""
        if len(group) == 1:
            photo = group[0]
            classification = classifier.classify_singleton(photo, use_cache=True)
            return [{"path": photo, "burst_size": 1, "burst_index": -1, "classification": classification}]
        else:
            classifications = classifier.classify_burst(group, use_cache=True)
            return [
                {"path": photo, "burst_size": len(group), "burst_index": i, "classification": cls}
                for i, (photo, cls) in enumerate(zip(group, classifications))
            ]

    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Classify all photos using Gemini."""
        if not groups:
            return []

        prompt_builder = PromptBuilder(
            self.config, training_examples={}, profile=self.profile
        )
        classifier = MediaClassifier(
            self.config, self.gemini_client, prompt_builder,
            self.cache_manager, profile=self.profile
        )

        print(f"\nStep 3: Classifying photos with Gemini...")
        workers = self.config.classification.parallel_photo_workers
        results = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._classify_group, group, classifier): idx
                for idx, group in enumerate(groups)
            }
            with tqdm(total=len(groups), desc="Processing bursts/singletons") as pbar:
                for future in as_completed(futures):
                    results.extend(future.result())
                    pbar.update(1)

        return results

    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route classified photos to destinations."""
        router = Router(self.config, self.gemini_client, profile=self.profile)

        for result in results:
            classification = result["classification"]
            if result.get("burst_size", 1) > 1:
                # For burst photos, use singleton routing per-photo
                # (burst routing is done at the group level in classify())
                result["destination"] = router.route_singleton(classification)
            else:
                result["destination"] = router.route_singleton(classification)

        return results

    def run(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool = False,
        classify_videos: bool = True,
    ) -> ClassificationResult:
        """Execute the full photo/video pipeline."""
        # Process images
        images = self.collect_files(input_folder)
        image_result = ClassificationResult()

        if images:
            print(f"\nProcessing {len(images)} images...")
            features = self.extract_features(images)

            # Dedup: remove near-duplicates before classification (saves Gemini calls)
            images, features, dupe_results = self._dedup(images, features, output_folder, dry_run)

            groups = self.group_files(images, features)
            results = self.classify(groups, features)
            routed = self.route(results)
            if not dry_run:
                self.move_files(routed, output_folder)
            image_result = ClassificationResult(
                results=routed + dupe_results,
                stats=self.compute_stats(routed + dupe_results),
            )

        # Process videos
        videos = self.collect_videos(input_folder)
        video_result = ClassificationResult()

        if videos:
            print(f"\nProcessing {len(videos)} videos...")
            if classify_videos:
                video_result = self._classify_videos(videos, output_folder, dry_run)
            else:
                video_result = self._copy_videos(videos, output_folder, dry_run)

        return image_result.merge(video_result)

    def _dedup(self, images, features, output_folder, dry_run):
        """Remove near-duplicate images, routing them to Storage."""
        detector = DuplicateDetector(self.config.dedup)
        embeddings = features.get("embeddings")
        unique_files, unique_embeddings, dupes = detector.detect(images, embeddings)

        dupe_results = []
        if dupes:
            # Route duplicates straight to Storage
            dupe_routed = [
                {
                    "path": p,
                    "burst_size": 1,
                    "burst_index": -1,
                    "classification": {
                        "classification": "Storage",
                        "score": 0,
                        "reasoning": "Near-duplicate (phash)",
                    },
                    "destination": "Storage",
                }
                for p in dupes
            ]
            if not dry_run:
                self.move_files(dupe_routed, output_folder)
            dupe_results = dupe_routed

        features = {"embeddings": unique_embeddings, "images": unique_files}
        return unique_files, features, dupe_results

    def _classify_single_video(self, video, classifier, router):
        """Classify a single video. Thread-safe."""
        classification = classifier.classify_video(video, use_cache=True)
        destination = router.route_video(classification)
        return {"path": video, "classification": classification, "destination": destination}

    def _classify_videos(
        self, videos: List[Path], output_folder: Path, dry_run: bool
    ) -> ClassificationResult:
        """Classify videos with Gemini."""
        prompt_builder = PromptBuilder(
            self.config, training_examples={}, profile=self.profile
        )
        classifier = MediaClassifier(
            self.config, self.gemini_client, prompt_builder,
            self.cache_manager, profile=self.profile
        )
        router = Router(self.config, self.gemini_client, profile=self.profile)

        workers = self.config.classification.parallel_video_workers
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._classify_single_video, v, classifier, router): v for v in videos}
            with tqdm(total=len(videos), desc="Classifying videos") as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

        if not dry_run:
            self.move_files(results, output_folder)

        return ClassificationResult(
            results=results,
            stats=self.compute_stats(results),
        )

    def _copy_videos(
        self, videos: List[Path], output_folder: Path, dry_run: bool
    ) -> ClassificationResult:
        """Copy videos without classification."""
        results = []
        for video in videos:
            results.append({
                "path": video,
                "classification": {"classification": "Videos", "confidence": None},
                "destination": "Videos",
            })

        if not dry_run:
            self.move_files(results, output_folder)

        return ClassificationResult(
            results=results,
            stats=self.compute_stats(results),
        )
