"""Mixed-media classification pipeline.

Auto-detects media types in a folder and dispatches to the appropriate sub-pipeline.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.config import Config
from ..core.cache import CacheManager
from ..core.ai_client import AIClient
from ..core.file_utils import FileTypeRegistry
from ..core.profiles import TasteProfile
from .base import ClassificationPipeline, ClassificationResult
from .photo_pipeline import PhotoPipeline
from .document_pipeline import DocumentPipeline


class MixedPipeline(ClassificationPipeline):
    """Pipeline that auto-detects media types and dispatches to sub-pipelines."""

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
        """Collect all media files (delegated to sub-pipelines)."""
        media = FileTypeRegistry.list_all_media(folder)
        return media["images"] + media["videos"] + media["documents"]

    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Not used directly - delegated to sub-pipelines."""
        return {}

    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Not used directly - delegated to sub-pipelines."""
        return [[f] for f in files]

    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Not used directly - delegated to sub-pipelines."""
        return []

    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Not used directly - delegated to sub-pipelines."""
        return results

    def run(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool = False,
        classify_videos: bool = True,
        classify_bundles: Optional[bool] = None,
    ) -> ClassificationResult:
        """Execute the mixed pipeline by dispatching to sub-pipelines.

        Args:
            input_folder: Folder to classify.
            output_folder: Where to copy sorted files.
            dry_run: If True, classify but don't move files.
            classify_videos: Whether to classify video files.
            classify_bundles: True = force bundles, False = disable,
                None = auto-detect (classify subfolders as bundles if present).

        Raises:
            FatalAPIError: On billing/auth/model errors or a tripped circuit breaker.
                Files already classified are cached, so a re-run resumes cheaply.
        """
        combined = ClassificationResult()

        # Fail fast on billing/auth/model problems before touching any files
        if self.config.classification.preflight_check and self.gemini_client is not None:
            print("\nPreflight: checking API key, billing and model...")
            self.gemini_client.preflight()

        # Auto-detect bundles: if not explicitly set, check for subfolders
        if classify_bundles is None:
            bundles = FileTypeRegistry.list_bundles(input_folder)
            if bundles:
                print(f"\nAuto-detected {len(bundles)} bundle(s) (subfolders with files). "
                      f"Use --no-bundles to classify files individually.")
                classify_bundles = True
            else:
                classify_bundles = False

        # Bundle mode: classify each subfolder as a single package
        if classify_bundles:
            bundle_result = self._classify_bundles(input_folder, output_folder, dry_run)
            combined = combined.merge(bundle_result)
            # Also classify any loose files at the top level
            media = FileTypeRegistry.list_all_media(input_folder)
            has_loose = any(len(v) > 0 for v in media.values())
            if not has_loose:
                return combined
            # Fall through to classify loose top-level files

        media = FileTypeRegistry.list_all_media(input_folder)

        has_images = len(media["images"]) > 0 or len(media["videos"]) > 0
        has_documents = len(media["documents"]) > 0

        if has_images:
            print(f"\nFound {len(media['images'])} images, {len(media['videos'])} videos")
            photo_pipeline = PhotoPipeline(
                self.config, self.profile, self.cache_manager, self.gemini_client
            )
            photo_result = photo_pipeline.run(
                input_folder, output_folder, dry_run, classify_videos
            )
            combined = combined.merge(photo_result)

        if has_documents:
            print(f"\nFound {len(media['documents'])} documents")
            doc_pipeline = DocumentPipeline(
                self.config, self.profile, self.cache_manager, self.gemini_client
            )
            doc_result = doc_pipeline.run(input_folder, output_folder, dry_run)
            combined = combined.merge(doc_result)

        if not has_images and not has_documents and not classify_bundles:
            print("\nNo media files found to process.")

        return combined

    def _classify_bundles(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool,
    ) -> ClassificationResult:
        """Classify subfolder bundles."""
        from ..classification.prompt_builder import PromptBuilder
        from ..classification.classifier import MediaClassifier
        from ..classification.routing import Router

        bundles = FileTypeRegistry.list_bundles(input_folder)
        if not bundles:
            print("\nNo subfolder bundles found.")
            return ClassificationResult()

        print(f"\nFound {len(bundles)} bundle(s)")

        prompt_builder = PromptBuilder(self.config, profile=self.profile)
        classifier = MediaClassifier(
            self.config, self.gemini_client, prompt_builder,
            self.cache_manager, profile=self.profile,
        )
        router = Router(self.config, self.gemini_client, profile=self.profile)

        # Optionally extract text for non-image files
        text_contents: Dict[Path, str] = {}
        try:
            from ..features.document_features import DocumentFeatureExtractor
            extractor = DocumentFeatureExtractor(self.config)
            for bundle in bundles:
                all_files = (
                    bundle["files"].get("documents", [])
                    + bundle["files"].get("audio", [])
                )
                for f in all_files:
                    try:
                        text_contents[f] = extractor.extract_text(f)
                    except Exception:
                        pass
        except ImportError:
            pass  # document extras not installed

        results = []
        stats: Dict[str, int] = {}
        for bundle in bundles:
            all_files = (
                bundle["files"].get("images", [])
                + bundle["files"].get("videos", [])
                + bundle["files"].get("audio", [])
                + bundle["files"].get("documents", [])
            )
            n = len(all_files)
            print(f"  {bundle['name']} ({n} file{'s' if n != 1 else ''})")

            classification = classifier.classify_bundle(
                bundle["name"], all_files, text_contents=text_contents,
            )
            destination = router.route_document(classification)

            result = {
                "path": bundle["path"],
                "bundle": True,
                "bundle_name": bundle["name"],
                "files": all_files,
                "classification": classification,
                "destination": destination,
            }
            results.append(result)
            stats[destination] = stats.get(destination, 0) + 1

        if not dry_run:
            self.move_files(results, output_folder)

        return ClassificationResult(results=results, stats=stats)
