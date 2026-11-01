#!/usr/bin/env python3
"""
LLM Taste Cloner - Main Classification Script

Automatically classify and sort family photos/videos/documents using Google Gemini AI.
Supports multiple taste profiles and mixed media types.
"""

import os
import sys
import argparse
import shutil
import csv
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Import refactored infrastructure
from taster.core import (
    load_config,
    CacheManager,
    FileTypeRegistry,
    ProfileManager,
    create_ai_client,
)
from taster.features import (
    QualityScorer,
    BurstDetector,
    EmbeddingExtractor,
)
from taster.classification import (
    PromptBuilder,
    MediaClassifier,
    Router,
)
from taster.pipelines import MixedPipeline, PhotoPipeline, DocumentPipeline
from taster.core.ai_client import FatalAPIError


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify family photos/videos/documents using Google Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C:\\Photos\\MyFolder"
  %(prog)s "C:\\Photos\\MyFolder" --profile default-photos
  %(prog)s "C:\\Photos\\MyFolder" --classify-videos
  %(prog)s "C:\\Photos\\MyFolder" --dry-run
  %(prog)s "C:\\Photos\\MyFolder" --config custom_config.yaml
        """
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing photos/videos/documents to classify"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: auto-detect)"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Taste profile name to use (default: from config or auto-detect)"
    )

    parser.add_argument(
        "--classify-videos",
        action="store_true",
        help="Enable video classification (ON by default in config.yaml, use --no-classify-videos to disable)"
    )

    parser.add_argument(
        "--no-classify-videos",
        action="store_true",
        help="Disable video classification"
    )

    parser.add_argument(
        "--parallel-videos",
        type=int,
        help="Number of parallel workers for video classification (default: from config)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't move files, just show what would happen"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: <folder>_sorted)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory (default: .taste_cache)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai", "anthropic", "local"],
        default=None,
        help="AI provider (default: auto-detect from API keys)"
    )

    parser.add_argument(
        "--bundles",
        action="store_true",
        help="Treat each subfolder as a bundle (one classification per subfolder)"
    )

    return parser.parse_args()


def setup_output_directories(output_base: Path, classify_videos: bool = True, profile=None) -> dict:
    """Create output directories for sorted media."""
    if profile:
        # Use profile-defined categories
        directories = {}
        for cat in profile.categories:
            directories[cat.name] = output_base / cat.name
        directories["Reports"] = output_base / "Reports"
    else:
        # Default directories
        directories = {
            "Share": output_base / "Share",
            "Storage": output_base / "Storage",
            "Review": output_base / "Review",
            "Ignore": output_base / "Ignore",
            "Reports": output_base / "Reports",
        }

    # Only create Videos folder when video classification is disabled
    if not classify_videos:
        directories["Videos"] = output_base / "Videos"

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def generate_report(results: list, report_dir: Path):
    """Generate classification report CSV."""
    rows = []

    for result in results:
        classification = result.get("classification", {})
        row = {
            "source": str(result["path"]),
            "destination": result.get("destination", "Unknown"),
            "classification": classification.get("classification", "Unknown"),
            "score": classification.get("score", 0),
            "burst_size": result.get("burst_size", result.get("group_size", 1)),
            "burst_index": result.get("burst_index", result.get("group_index", -1)),
            "rank": classification.get("rank", None),
            "reasoning": classification.get("reasoning", ""),
            "contains_children": classification.get("contains_children", None),
            "is_appropriate": classification.get("is_appropriate", None),
            "content_summary": classification.get("content_summary", ""),
            "key_topics": ",".join(classification.get("key_topics", [])),
            # Error tracking fields
            "is_error_fallback": classification.get("is_error_fallback", False),
            "error_type": classification.get("error_type", ""),
            "error_message": classification.get("error_message", ""),
            "retry_count": classification.get("retry_count", 0),
            "burst_id": classification.get("burst_id", ""),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    report_path = report_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_path, index=False)
    print(f"\nReport saved: {report_path}")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Validate input folder
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config) if args.config else None
    print(f"Loading configuration...")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Override config with CLI arguments
    if args.no_classify_videos:
        config.classification.classify_videos = False
    elif args.classify_videos:
        config.classification.classify_videos = True

    if args.parallel_videos:
        config.classification.parallel_video_workers = args.parallel_videos

    if args.dry_run:
        config.system.dry_run = True

    if args.cache_dir:
        config.paths.cache_root = Path(args.cache_dir)

    # Load taste profile
    profile = None
    profile_name = args.profile or config.profiles.active_profile
    profile_manager = ProfileManager(config.profiles.profiles_dir)

    if profile_manager.profile_exists(profile_name):
        profile = profile_manager.load_profile(profile_name)
        print(f"Using taste profile: {profile.name} ({profile.description})")
    else:
        # Auto-detect media type and use appropriate default
        if config.profiles.auto_detect_media_type:
            media_type = FileTypeRegistry.detect_media_type(folder)
            print(f"Auto-detected media type: {media_type}")
            profile = profile_manager.get_default_profile(media_type)
            print(f"Using default profile: {profile.name}")
        else:
            print(f"Profile '{profile_name}' not found, using legacy taste preferences")

    # Determine output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = folder.parent / f"{folder.name}_sorted"

    print(f"Output directory: {output_base}")

    # Setup output directories
    output_dirs = setup_output_directories(
        output_base,
        config.classification.classify_videos,
        profile=profile
    )

    # Initialize cache manager
    cache_manager = CacheManager(
        config.paths.cache_root,
        ttl_days=config.caching.ttl_days,
        enabled=config.caching.enabled
    )

    # Initialize AI client
    try:
        gemini_client = create_ai_client(config, provider=args.provider)
        print(f"AI client initialized: {gemini_client.provider_name}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Scan folder contents
    media = FileTypeRegistry.list_all_media(folder)
    total = len(media["images"]) + len(media["videos"]) + len(media["documents"])
    print(f"\nScanning folder: {folder}")
    print(f"   Found {len(media['images'])} images, {len(media['videos'])} videos, {len(media['documents'])} documents")

    if total == 0:
        print("\nNo media files found to process.")
        sys.exit(0)

    # Run the appropriate pipeline
    pipeline = MixedPipeline(
        config, profile, cache_manager, gemini_client
    )
    start_time = time.time()
    try:
        result = pipeline.run(
            folder, output_base, config.system.dry_run,
            classify_videos=config.classification.classify_videos,
            classify_bundles=getattr(args, 'bundles', False)
        )
    except FatalAPIError as e:
        from taster.cli import _print_fatal_api_error
        _print_fatal_api_error(e)
        sys.exit(1)
    result.elapsed_seconds = time.time() - start_time

    # Generate reports
    if result.results:
        report_dir = output_dirs.get("Reports", output_base / "Reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        generate_report(result.results, report_dir)
        result.generate_summary_report(report_dir, provider_name=gemini_client.provider_name)

    # Print summary
    result.print_summary("media files")

    if config.system.dry_run:
        print("\nThis was a dry run. No files were moved.")
        print("   Remove --dry-run to actually move files.")

    print("\nClassification complete!")


if __name__ == "__main__":
    main()
