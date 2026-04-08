"""
Deduplicate images/videos in folders using perceptual hashing (phash).
Only catches true duplicates — same image recompressed, resized, or reformatted.
Moves duplicates to a _dupes subfolder for manual review.
For videos, extracts a frame at 1s for hashing.
"""

import shutil
import argparse
from pathlib import Path

import imagehash
import cv2
from PIL import Image


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.3gp'}


def extract_video_frame(path, timestamp_sec=1.0):
    """Extract a single frame from a video at the given timestamp."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None


def get_phash(path, hash_size=16):
    """Get perceptual hash for an image or video file."""
    ext = path.suffix.lower()
    img = None

    if ext in IMAGE_EXTS:
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"  [skip] Can't open {path.name}: {e}")
            return None
    elif ext in VIDEO_EXTS:
        img = extract_video_frame(path)
        if img is None:
            print(f"  [skip] Can't extract frame from {path.name}")
            return None
    else:
        return None

    return imagehash.phash(img, hash_size=hash_size)


def find_duplicates(hashes, max_distance):
    """Find duplicate groups using hamming distance on perceptual hashes."""
    if len(hashes) == 0:
        return []

    items = list(hashes.items())
    seen = set()
    groups = []

    for i, (path_i, hash_i) in enumerate(items):
        if path_i in seen:
            continue
        group = [path_i]
        for j in range(i + 1, len(items)):
            path_j, hash_j = items[j]
            if path_j in seen:
                continue
            if hash_i - hash_j <= max_distance:
                group.append(path_j)
                seen.add(path_j)
        if len(group) > 1:
            groups.append(group)

    return groups


def dedupe_folder(folder, max_distance, hash_size, dry_run):
    folder = Path(folder)
    print(f"\n{'='*60}")
    print(f"Processing: {folder}")

    files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    ])
    print(f"  {len(files)} media files found")

    if len(files) < 2:
        print("  Nothing to dedupe.")
        return 0

    # Compute perceptual hashes
    hashes = {}
    for i, f in enumerate(files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Hashing {i+1}/{len(files)}...")
        h = get_phash(f, hash_size=hash_size)
        if h is not None:
            hashes[f] = h

    print(f"  {len(hashes)} hashes computed")

    # Find duplicates
    dupe_groups = find_duplicates(hashes, max_distance)

    if not dupe_groups:
        print("  No duplicates found.")
        return 0

    # Move duplicates (keep first in each group by filename)
    dupes_dir = folder / "_dupes"
    moved = 0
    for group in dupe_groups:
        group.sort(key=lambda p: p.name)
        keep = group[0]
        to_move = group[1:]
        print(f"  Keep: {keep.name}")
        for f in to_move:
            print(f"    -> dupe: {f.name}")
            if not dry_run:
                dupes_dir.mkdir(exist_ok=True)
                dest = dupes_dir / f.name
                if dest.exists():
                    dest = dupes_dir / f"{f.stem}_dup{f.suffix}"
                shutil.move(str(f), str(dest))
            moved += 1

    print(f"  {'Would move' if dry_run else 'Moved'} {moved} duplicates")
    return moved


def main():
    parser = argparse.ArgumentParser(description="Deduplicate media using perceptual hashing")
    parser.add_argument("folders", nargs="+", help="Folders to deduplicate")
    parser.add_argument("--max-distance", type=int, default=4,
                        help="Max hamming distance to consider duplicate (default: 4)")
    parser.add_argument("--hash-size", type=int, default=16,
                        help="Phash size — 16 gives 256-bit hashes (default: 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be moved without moving")
    args = parser.parse_args()

    print(f"Max hamming distance: {args.max_distance}")
    print(f"Hash size: {args.hash_size} ({args.hash_size**2}-bit hashes)")
    if args.dry_run:
        print("DRY RUN - no files will be moved")

    total_moved = 0
    for folder in args.folders:
        total_moved += dedupe_folder(folder, args.max_distance, args.hash_size, args.dry_run)

    print(f"\n{'='*60}")
    print(f"Total duplicates {'found' if args.dry_run else 'moved'}: {total_moved}")


if __name__ == "__main__":
    main()
