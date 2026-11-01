"""Unified routing logic with burst-awareness and profile support."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from ..core.config import Config, ClassificationConfig
from ..core.ai_client import AIClient
from ..core.file_utils import ImageUtils
from ..core.profiles import TasteProfile


class Router:
    """
    Unified router for classification results.

    Handles:
    - Singleton routing (threshold-based)
    - Burst-aware routing (rank + threshold hybrid)
    - Diversity checking (prevent similar photos in Share)
    - Document routing (profile-defined categories)
    """

    def __init__(
        self,
        config: Config,
        gemini_client: Optional[AIClient] = None,
        profile: Optional[TasteProfile] = None
    ):
        """
        Initialize router.

        Args:
            config: Configuration object.
            gemini_client: Optional AI client for diversity checking.
            profile: Optional TasteProfile for profile-defined category routing.
        """
        self.config = config
        self.client = gemini_client
        self.profile = profile

    def _error_destination(self, classification: Dict[str, Any]) -> Optional[str]:
        """Return the error folder for failed classifications, or None to route normally."""
        if classification.get("is_error_fallback"):
            return self.config.classification.error_destination
        return None

    def route_singleton(self, classification: Dict[str, Any]) -> str:
        """
        Route a singleton photo based on classification.

        Args:
            classification: Classification result dict.

        Returns:
            Destination folder: "Share", "Storage", "Review", or "Ignore".
        """
        error_dest = self._error_destination(classification)
        if error_dest:
            return error_dest

        # Extract fields
        category = classification.get("classification", "Review")
        score = classification.get("score", 2)
        contains_children = classification.get("contains_children", None)
        is_appropriate = classification.get("is_appropriate", None)

        # If explicitly classified as Ignore
        if category == "Ignore":
            return "Ignore"

        # If no children detected
        if contains_children == False:
            return "Ignore"

        # If inappropriate content
        if is_appropriate == False:
            return "Review"  # Route to Review for manual inspection

        # If AI explicitly classified as Review, respect it
        if category == "Review":
            return "Review"

        # Apply thresholds (score is the calibrated signal)
        if category == "Share" and score >= self.config.classification.share_threshold:
            return "Share"
        elif score >= self.config.classification.review_threshold:
            return "Review"  # Borderline — score 3 = genuinely torn
        else:
            return "Storage"

    def route_burst(
        self,
        burst_photos: List[Path],
        classifications: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Route a burst of photos with burst-awareness.

        Applies hybrid routing:
        1. Filter by rank (only top N considered for Share)
        2. Apply absolute share-worthiness threshold
        3. Check diversity between Share candidates

        Args:
            burst_photos: List of photo paths in burst.
            classifications: List of classification results (one per photo).
            embeddings: Optional CLIP embeddings for diversity checking.

        Returns:
            List of destinations (one per photo, in order).
        """
        if len(burst_photos) != len(classifications):
            raise ValueError(f"Mismatch: {len(burst_photos)} photos, {len(classifications)} classifications")

        destinations = []

        # Extract ranks and sort by rank
        ranked_indices = sorted(
            range(len(classifications)),
            key=lambda i: classifications[i].get("rank", i + 1)
        )

        # Track Share candidates for diversity checking
        share_candidates = []

        for i in range(len(burst_photos)):
            photo_path = burst_photos[i]
            classification = classifications[i]
            rank = classification.get("rank", i + 1)

            error_dest = self._error_destination(classification)
            if error_dest:
                destinations.append(error_dest)
                continue

            # Extract fields
            category = classification.get("classification", "Review")
            score = classification.get("score", 2)
            contains_children = classification.get("contains_children", None)
            is_appropriate = classification.get("is_appropriate", None)

            # If explicitly classified as Ignore
            if category == "Ignore" or contains_children == False:
                destinations.append("Ignore")
                continue

            # If inappropriate
            if is_appropriate == False:
                destinations.append("Review")
                continue

            # If AI explicitly classified as Review, respect it
            if category == "Review":
                destinations.append("Review")
                continue

            # Burst-aware routing (score is the calibrated signal)
            if rank <= self.config.classification.burst_rank_consider:
                # Top-ranked photos: consider for Share based on absolute threshold
                if score >= self.config.classification.share_threshold:
                    # Check diversity with existing Share candidates
                    if self.config.classification.enable_diversity_check and share_candidates:
                        is_diverse = self._check_diversity_with_candidates(
                            photo_path,
                            share_candidates,
                            embeddings,
                            i,
                            ranked_indices
                        )
                        if is_diverse:
                            destinations.append("Share")
                            share_candidates.append((i, photo_path))
                        else:
                            # Too similar to existing Share — borderline goes to Review
                            if score >= self.config.classification.review_threshold:
                                destinations.append("Review")
                            else:
                                destinations.append("Storage")
                    else:
                        # No diversity check needed (first candidate or disabled)
                        destinations.append("Share")
                        share_candidates.append((i, photo_path))
                elif score >= self.config.classification.review_threshold:
                    destinations.append("Review")
                else:
                    destinations.append("Storage")

            elif score >= self.config.classification.review_threshold:
                # Borderline score — Review regardless of rank
                destinations.append("Review")

            else:
                # Low score — Storage
                destinations.append("Storage")

        return destinations

    def _check_diversity_with_candidates(
        self,
        photo_path: Path,
        share_candidates: List[Tuple[int, Path]],
        embeddings: Optional[np.ndarray],
        current_index: int,
        ranked_indices: List[int]
    ) -> bool:
        """
        Check if photo is diverse enough from existing Share candidates.

        Args:
            photo_path: Path to current photo.
            share_candidates: List of (index, path) tuples for Share candidates.
            embeddings: Optional CLIP embeddings.
            current_index: Current photo index.
            ranked_indices: Sorted indices by rank.

        Returns:
            True if diverse enough, False if too similar.
        """
        if embeddings is None:
            # No embeddings available, assume diverse
            return True

        if not share_candidates:
            # First candidate, always diverse
            return True

        # Get current embedding
        current_embedding = embeddings[current_index]

        # Check similarity with each Share candidate
        for candidate_idx, candidate_path in share_candidates:
            candidate_embedding = embeddings[candidate_idx]

            # Compute cosine similarity
            similarity = float(np.dot(current_embedding, candidate_embedding))

            # If too similar, not diverse
            if similarity >= self.config.classification.diversity_similarity_threshold:
                print(f"   Diversity check: {photo_path.name} too similar to {candidate_path.name} (sim={similarity:.3f})")
                return False

        # Diverse from all candidates
        return True

    def check_diversity(
        self,
        photo_a_path: Path,
        photo_b_path: Path,
        gemini_check: bool = False
    ) -> Tuple[bool, float, str]:
        """
        Check if two photos are meaningfully different.

        Args:
            photo_a_path: Path to first photo.
            photo_b_path: Path to second photo.
            gemini_check: Whether to use Gemini for semantic diversity check.

        Returns:
            Tuple of (is_diverse, confidence, reasoning).
        """
        if not gemini_check or self.client is None:
            # Simple check: assume diverse
            return (True, 0.5, "Diversity checking disabled")

        # Use Gemini to check diversity
        try:
            img_a = ImageUtils.load_and_fix_orientation(photo_a_path, max_size=800)
            img_b = ImageUtils.load_and_fix_orientation(photo_b_path, max_size=800)

            if img_a is None or img_b is None:
                return (True, 0.5, "Could not load images")

            prompt = """Compare these two photos from the same burst. Are they meaningfully DIFFERENT enough to both be worth keeping?

**Context:** These are consecutive photos from the same moment. We want to keep multiple photos from a burst ONLY if they capture genuinely different moments or expressions.

**"Meaningfully different" means (keep BOTH):**
- Clearly different facial expressions (smiling → laughing, eyes closed → open, neutral → big smile)
- Significantly different composition (different crop, different angle, different framing)
- Different action/moment (before → after movement, different interaction between people)
- Different pose (not just micro-adjustments)

**NOT meaningfully different (keep ONLY the best one):**
- Nearly identical - just camera shake or tiny position shifts (1-2cm)
- Same general pose, same general expression (minor micro-expression changes don't count)
- Same composition with fractional timing difference
- One is just a worse/blurrier version of the other
- Baby lying in same position on blanket with no real expression change

**Decision rule:** If a regular person looking at both photos would say "these look the same to me", mark as NOT diverse. Only keep both if there's a clear, human-noticeable difference. When uncertain, be slightly conservative (lean toward NOT diverse to avoid too many duplicates).

Respond with JSON:
{
    "is_diverse": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (10 words max)",
    "key_difference": "What's different? (5 words max)"
}"""

            result = self.client.generate_json(
                prompt=[prompt, img_a, img_b],
                fallback={"is_diverse": True, "confidence": 0.5, "reasoning": "Error checking diversity"},
                generation_config={"max_output_tokens": 512},
                handle_safety_errors=True
            )

            is_diverse = result.get("is_diverse", True)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")

            return (is_diverse, confidence, reasoning)

        except Exception as e:
            print(f"Diversity check error: {e}")
            return (True, 0.5, f"Error: {e}")

    def route_video(self, classification: Dict[str, Any]) -> str:
        """
        Route a video based on classification.

        Args:
            classification: Classification result dict.

        Returns:
            Destination folder: "Share", "Storage", "Review", or "Ignore".
        """
        # Videos use same logic as singletons
        return self.route_singleton(classification)

    def route_audio(self, classification: Dict[str, Any]) -> str:
        """
        Route an audio file based on classification.

        Uses profile-threshold routing (same as documents). No children/appropriateness checks.

        Args:
            classification: Classification result dict.

        Returns:
            Destination category name.
        """
        return self.route_document(classification)

    def route_document(self, classification: Dict[str, Any]) -> str:
        """
        Route a document based on classification using profile-defined categories.

        If a TasteProfile is set, uses profile thresholds and categories.
        Otherwise falls back to simple threshold-based routing.

        Args:
            classification: Classification result dict.

        Returns:
            Destination category name.
        """
        error_dest = self._error_destination(classification)
        if error_dest:
            return error_dest

        category = classification.get("classification", "Review")
        score = classification.get("score", 2)

        if self.profile:
            # Validate category is in profile
            valid_names = self.profile.category_names
            if category not in valid_names:
                return self.profile.default_category

            # Check thresholds for the category
            threshold = self.profile.thresholds.get(category)
            if threshold is not None and score < threshold:
                # Score too low for this category, fall back
                return self.profile.default_category

            return category

        # Fallback: use photo routing logic
        return self.route_singleton(classification)
