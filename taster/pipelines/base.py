"""Abstract base class for classification pipelines."""
import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.config import Config
from ..core.profiles import TasteProfile

# Gemini Flash token rates
_TOKENS_PER_IMAGE = 258
_VIDEO_TOKENS_PER_SEC = 290  # 258 visual (1fps) + 32 audio
_VIDEO_BITRATE_BPS = 12_000_000  # 12 Mbps typical phone video
_PROMPT_TOKENS = 1000  # average prompt size
_OUTPUT_TOKENS = 300  # average output size

# Gemini 3.8 Flash pricing per 1M tokens (standard paid tier, Oct 2026)
_INPUT_PRICE_PER_M = 0.75
_OUTPUT_PRICE_PER_M = 3.75


@dataclass
class ClassificationResult:
    """Result from running a classification pipeline."""
    results: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def print_summary(self, media_label: str = "items"):
        """Print a human-readable summary."""
        total = sum(self.stats.values()) if self.stats else 0
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION SUMMARY ({media_label})")
        print(f"{'='*60}")
        if self.stats:
            for destination, count in sorted(self.stats.items()):
                pct = (count / total * 100) if total else 0
                print(f"   {destination:25s}: {count:4d}  ({pct:.1f}%)")
            print(f"   {'':25s}  {'----':>4s}")
            print(f"   {'TOTAL':25s}: {total:4d}")
        else:
            print("   No items processed.")
        if self.elapsed_seconds > 0:
            minutes = int(self.elapsed_seconds // 60)
            seconds = int(self.elapsed_seconds % 60)
            print(f"\n   Time: {minutes}m {seconds}s")
        cost = self._estimate_cost()
        if cost > 0:
            print(f"   Est. cost: ~${cost:.2f} (Gemini Flash)")
        top = self._top_confident(5)
        if top:
            print(f"\n   Top {len(top)} highest-confidence:")
            for i, item in enumerate(top, 1):
                name = Path(item["path"]).name if isinstance(item["path"], (str, Path)) else str(item["path"])
                cls = item.get("classification", {})
                score = cls.get("score", 0)
                reason = cls.get("reasoning", "")
                if len(reason) > 60:
                    reason = reason[:57] + "..."
                print(f"   {i}. {name} ({score}/5) - {reason}")
        print(f"{'='*60}")

    def _estimate_cost(self) -> float:
        """Estimate API cost from token counts (images, video duration, prompts)."""
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tif", ".tiff", ".bmp"}
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

        total_input_tokens = 0
        api_calls = 0

        for r in self.results:
            path = Path(r["path"]) if isinstance(r["path"], (str, Path)) else r["path"]
            ext = path.suffix.lower()

            if ext in image_exts:
                total_input_tokens += _TOKENS_PER_IMAGE
            elif ext in video_exts:
                # Estimate duration from file size
                try:
                    file_bytes = path.stat().st_size
                    duration_sec = max(file_bytes * 8 / _VIDEO_BITRATE_BPS, 1)
                except OSError:
                    duration_sec = 60  # fallback: assume 1 minute
                total_input_tokens += int(duration_sec * _VIDEO_TOKENS_PER_SEC)
            else:
                total_input_tokens += 2000  # documents: rough estimate

            # Count API calls (burst photos share a single call)
            burst_size = r.get("burst_size", 1)
            if burst_size <= 1:
                api_calls += 1
            else:
                api_calls += 1 / burst_size  # fractional: shared call

        # Add prompt + output tokens per API call
        api_calls = max(int(api_calls), 1)
        total_input_tokens += api_calls * _PROMPT_TOKENS
        total_output_tokens = api_calls * _OUTPUT_TOKENS

        input_cost = total_input_tokens / 1_000_000 * _INPUT_PRICE_PER_M
        output_cost = total_output_tokens / 1_000_000 * _OUTPUT_PRICE_PER_M
        return input_cost + output_cost

    def _top_confident(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the top N highest-scoring results."""
        scored = []
        for r in self.results:
            cls = r.get("classification", {})
            score = cls.get("score", 0)
            if score and not cls.get("is_error_fallback", False):
                scored.append(r)
        scored.sort(key=lambda x: x.get("classification", {}).get("score", 0), reverse=True)
        return scored[:n]

    def generate_summary_report(self, report_dir: Path, provider_name: str = "gemini"):
        """Save a human-readable summary report to Reports/summary.txt."""
        total = sum(self.stats.values()) if self.stats else 0
        cost = self._estimate_cost()
        top = self._top_confident(5)

        lines = []
        lines.append(f"Classification Complete")
        lines.append("=" * 48)
        if self.stats:
            for dest, count in sorted(self.stats.items()):
                pct = (count / total * 100) if total else 0
                lines.append(f"  {dest:25s}: {count:4d}  ({pct:.1f}%)")
            lines.append(f"  {'':25s}  {'----':>4s}")
            lines.append(f"  {'Total':25s}: {total:4d}")
        lines.append("=" * 48)
        if self.elapsed_seconds > 0:
            minutes = int(self.elapsed_seconds // 60)
            seconds = int(self.elapsed_seconds % 60)
            lines.append(f"  Time: {minutes}m {seconds}s")
        if cost > 0:
            lines.append(f"  Est. cost: ~${cost:.2f} ({provider_name.title()})")
        lines.append("")

        # Error summary
        errors = [r for r in self.results if r.get("classification", {}).get("is_error_fallback")]
        if errors:
            lines.append(f"  Errors: {len(errors)} files fell back to default category")
            lines.append("")

        if top:
            lines.append(f"  Top {len(top)} highest-scoring:")
            for i, item in enumerate(top, 1):
                name = Path(item["path"]).name if isinstance(item["path"], (str, Path)) else str(item["path"])
                cls = item.get("classification", {})
                score = cls.get("score", 0)
                dest = item.get("destination", "")
                reason = cls.get("reasoning", "")
                if len(reason) > 70:
                    reason = reason[:67] + "..."
                lines.append(f"  {i}. {name}")
                lines.append(f"     {dest} ({score}/5) - {reason}")
            lines.append("")

        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Summary saved: {summary_path}")
        return "\n".join(lines)

    def merge(self, other: "ClassificationResult") -> "ClassificationResult":
        """Merge another result into this one."""
        combined_stats = defaultdict(int)
        for k, v in self.stats.items():
            combined_stats[k] += v
        for k, v in other.stats.items():
            combined_stats[k] += v
        return ClassificationResult(
            results=self.results + other.results,
            stats=dict(combined_stats),
        )


class ClassificationPipeline(ABC):
    """Abstract base for all classification pipelines."""

    def __init__(self, config: Config, profile: Optional[TasteProfile] = None):
        self.config = config
        self.profile = profile

    @abstractmethod
    def collect_files(self, folder: Path) -> List[Path]:
        """Collect files to process."""

    @abstractmethod
    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Extract features from files."""

    @abstractmethod
    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Group related files (bursts, similar docs)."""

    @abstractmethod
    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Classify all files."""

    @abstractmethod
    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route results to output categories."""

    def run(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool = False
    ) -> ClassificationResult:
        """Execute the full pipeline."""
        files = self.collect_files(input_folder)
        if not files:
            return ClassificationResult()

        features = self.extract_features(files)
        groups = self.group_files(files, features)
        results = self.classify(groups, features)
        routed = self.route(results)

        if not dry_run:
            self.move_files(routed, output_folder)

        stats = self.compute_stats(routed)
        return ClassificationResult(results=routed, stats=stats)

    def move_files(self, results: List[Dict[str, Any]], output_folder: Path):
        """Copy files to their destination subfolders."""
        for result in results:
            destination = result["destination"]
            if result.get("bundle"):
                # Bundle: copy all files into destination/bundle_name/
                bundle_name = result["bundle_name"]
                dst_dir = output_folder / destination / bundle_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                for source in result["files"]:
                    try:
                        shutil.copy2(source, dst_dir / source.name)
                    except Exception as e:
                        print(f"Warning: Error copying {source.name}: {e}")
            else:
                source = result["path"]
                dst_dir = output_folder / destination
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / source.name
                try:
                    shutil.copy2(source, dst_path)
                except Exception as e:
                    print(f"Warning: Error copying {source.name}: {e}")

    def compute_stats(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute statistics from routed results."""
        stats = defaultdict(int)
        for result in results:
            stats[result["destination"]] += 1
        return dict(stats)
