"""Perceptual hash (phash) duplicate detection.

Catches true duplicates — same image recompressed, resized, or reformatted
(WhatsApp re-sends, motion_photo extracts, etc.) — without false-positiving
on similar-but-different shots the way embedding similarity does.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ..core.config import DedupConfig
from ..core.file_utils import ImageUtils


class DuplicateDetector:
    """Detect duplicate images using perceptual hashing."""

    def __init__(self, config: DedupConfig):
        self.config = config

    def detect(
        self,
        files: List[Path],
        embeddings: "np.ndarray",
    ) -> Tuple[List[Path], "np.ndarray", List[Path]]:
        """Find and separate duplicates from unique images.

        Uses perceptual hashing (phash) to find true duplicates.
        For each group, keeps the file whose name sorts first and marks
        the rest as dupes.

        Args:
            files: Ordered list of image paths.
            embeddings: Corresponding CLIP embeddings (passed through unchanged).

        Returns:
            ``(unique_files, unique_embeddings, duplicate_files)``
        """
        import numpy as np
        import imagehash

        if not self.config.enabled or len(files) < 2:
            return files, embeddings, []

        max_dist = self.config.max_hamming_distance
        hash_size = self.config.hash_size

        # Compute perceptual hashes
        hashes = {}
        for f in files:
            try:
                img = ImageUtils.load_and_fix_orientation(f, max_size=512)
                if img is None:
                    continue
                img = ImageUtils.ensure_rgb(img)
                hashes[f] = imagehash.phash(img, hash_size=hash_size)
            except Exception:
                continue

        # Greedy grouping by hamming distance
        file_list = [f for f in files if f in hashes]
        seen = set()
        duplicate_set = set()

        for i, fi in enumerate(file_list):
            if fi in seen:
                continue
            seen.add(fi)
            for j in range(i + 1, len(file_list)):
                fj = file_list[j]
                if fj in seen:
                    continue
                if hashes[fi] - hashes[fj] <= max_dist:
                    seen.add(fj)
                    duplicate_set.add(fj)

        if not duplicate_set:
            return files, embeddings, []

        # Build outputs preserving original order
        unique_mask = np.array([f not in duplicate_set for f in files])
        unique_files = [f for f in files if f not in duplicate_set]
        unique_embeddings = embeddings[unique_mask]
        duplicate_files = [f for f in files if f in duplicate_set]

        print(f"   Dedup: {len(duplicate_files)} duplicates detected "
              f"(phash, max distance {max_dist}), "
              f"{len(unique_files)} unique images remain")

        return unique_files, unique_embeddings, duplicate_files
