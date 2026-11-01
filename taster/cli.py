"""Taster CLI — ``taster <command>``."""

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    from taster import __version__

    parser = argparse.ArgumentParser(
        prog="taster",
        description="Teach AI your taste. Apply it to everything.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ────────────────────────────────────────────────────────
    sub.add_parser("init", help="Interactive first-run setup")

    # ── classify ────────────────────────────────────────────────────
    cls = sub.add_parser("classify", help="Classify files against a profile")
    cls.add_argument("folder", type=str, help="Folder to classify")
    cls.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    cls.add_argument("--profile", type=str, default=None, help="Taste profile name")
    cls.add_argument("--provider", choices=["gemini", "openai", "anthropic", "local"], default=None)
    cls.add_argument("--dry-run", action="store_true", help="Preview without moving files")
    cls.add_argument("--output", type=str, default=None, help="Output directory")
    cls.add_argument("--classify-videos", action="store_true")
    cls.add_argument("--no-classify-videos", action="store_true")
    cls.add_argument("--parallel-videos", type=int, default=None)
    cls.add_argument("--parallel-photos", type=int, default=None)
    cls.add_argument("--cache-dir", type=str, default=None)
    cls.add_argument("--no-cache", action="store_true", help="Skip classification cache (re-query AI for every item)")
    cls.add_argument("--bundles", action="store_true", help="Force bundle mode (one classification per subfolder)")
    cls.add_argument("--no-bundles", action="store_true", help="Disable bundle auto-detection")

    # ── train ───────────────────────────────────────────────────────
    trn = sub.add_parser("train", help="Launch Gradio pairwise trainer")
    trn.add_argument("folder", type=str, help="Folder of images to train on")

    # ── serve ───────────────────────────────────────────────────────
    srv = sub.add_parser("serve", help="Start MCP server for Claude Desktop")

    # ── status ──────────────────────────────────────────────────────
    sub.add_parser("status", help="Show config location, profiles, and API key status")

    return parser


# ── subcommand handlers ─────────────────────────────────────────────


def _cmd_init(args: argparse.Namespace) -> None:
    """Interactive first-run setup."""
    import json
    import os
    import platform
    import subprocess

    from taster.dirs import ensure_dirs, get_config_dir, get_env_path

    print("\nTaster Setup")
    print("=" * 40)

    # Step 1: Python version
    print("\n[1/4] Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 12):
        print(f"  Python 3.12+ required, found {major}.{minor}")
        print("  Download from https://www.python.org/downloads/")
        sys.exit(1)
    print(f"  Python {major}.{minor} OK")

    # Step 2: Create config directory
    print("\n[2/4] Creating config directory...")
    config_dir = ensure_dirs()
    print(f"  {config_dir}")

    # Step 3: API keys
    print("\n[3/4] Configuring API keys...")
    env_file = get_env_path()
    existing: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()

    providers = [
        ("GEMINI_API_KEY", "Gemini (recommended -- cheapest, native video/PDF)"),
        ("OPENAI_API_KEY", "OpenAI (GPT-5)"),
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("LOCAL_LLM_URL", "Local LLM (Ollama, LM Studio, vLLM -- e.g. http://localhost:11434/v1)"),
    ]

    print("\n  At least one AI provider API key is required.")
    if existing:
        configured = [name for name, _ in providers if existing.get(name)]
        if configured:
            print(f"  Already configured: {', '.join(configured)}")

    keys = dict(existing)
    for env_var, label in providers:
        current = existing.get(env_var, "")
        if current:
            change = input(f"  {label}: [keep current] ").strip()
            if change:
                keys[env_var] = change
        else:
            value = input(f"  {label}: [skip] ").strip()
            if value:
                keys[env_var] = value

    has_any = any(keys.get(name) for name, _ in providers)
    if not has_any:
        print("\n  No API keys configured. You'll need at least one to use Taster.")
    else:
        lines = []
        for env_var, label in providers:
            if keys.get(env_var):
                lines.append(f"{env_var}={keys[env_var]}")
        for key, value in keys.items():
            if key not in {name for name, _ in providers}:
                lines.append(f"{key}={value}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Saved to {env_file}")

    # Step 4: Claude Desktop
    print("\n[4/4] Claude Desktop integration...")
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        cd_config = Path(appdata) / "Claude" / "claude_desktop_config.json" if appdata else None
    elif system == "Darwin":
        cd_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        cd_config = Path(xdg) / "Claude" / "claude_desktop_config.json"

    if cd_config:
        print(f"  Config location: {cd_config}")
        setup_claude = input("  Add Taster to Claude Desktop? [Y/n] ").strip().lower()
        if setup_claude != "n":
            mcp_entry = {
                "command": "taster",
                "args": ["serve"],
                "env": {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                },
            }
            if cd_config.exists():
                try:
                    config = json.loads(cd_config.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    config = {}
            else:
                config = {}
            servers = config.setdefault("mcpServers", {})
            if "taster" in servers:
                print(f"  Taster already in {cd_config}")
                update = input("  Overwrite? [y/N] ").strip().lower()
                if update != "y":
                    print("  Kept existing entry.")
                else:
                    servers["taster"] = mcp_entry
            else:
                servers["taster"] = mcp_entry
            cd_config.parent.mkdir(parents=True, exist_ok=True)
            cd_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"  Added Taster to {cd_config}")
        else:
            print("  Skipped.")
    else:
        print("  Claude Desktop config not found -- skipping.")

    print("\n" + "=" * 40)
    print("Setup complete!")
    print()
    print('Open Claude Desktop and say: "Check my Taster status"')
    print()


def _print_fatal_api_error(e: Exception) -> None:
    """Explain why a run was aborted and how to recover."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION ABORTED")
    print("=" * 60)
    print(f"  {e}")
    print()
    print("  Likely causes: API credits depleted or not yet active, invalid API key,")
    print("  or the configured model was retired. Fix the cause and re-run; files")
    print("  already classified are cached and won't be re-billed.")
    print("=" * 60)


def _cmd_classify(args: argparse.Namespace) -> None:
    """Classify files in a folder."""
    from dotenv import load_dotenv

    load_dotenv()

    from taster.core.config import load_config
    from taster.core.cache import CacheManager
    from taster.core.file_utils import FileTypeRegistry
    from taster.core.profiles import ProfileManager
    from taster.core.ai_client import FatalAPIError
    from taster.core.provider_factory import create_ai_client
    from taster.pipelines import MixedPipeline

    # Also load user-level .env
    from taster.dirs import get_env_path

    user_env = get_env_path()
    if user_env.exists():
        load_dotenv(user_env)

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    config_path = Path(args.config) if args.config else None
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.no_classify_videos:
        config.classification.classify_videos = False
    elif args.classify_videos:
        config.classification.classify_videos = True
    if args.parallel_videos:
        config.classification.parallel_video_workers = args.parallel_videos
    if args.parallel_photos:
        config.classification.parallel_photo_workers = args.parallel_photos
    if args.dry_run:
        config.system.dry_run = True
    if args.cache_dir:
        config.paths.cache_root = Path(args.cache_dir)

    # Load profile
    from taster.dirs import find_profiles_dir

    profiles_dir = find_profiles_dir()
    profile_name = args.profile or config.profiles.active_profile
    profile_manager = ProfileManager(str(profiles_dir))

    profile = None
    if profile_manager.profile_exists(profile_name):
        profile = profile_manager.load_profile(profile_name)
        print(f"Using taste profile: {profile.name} ({profile.description})")
    elif config.profiles.auto_detect_media_type:
        media_type = FileTypeRegistry.detect_media_type(folder)
        print(f"Auto-detected media type: {media_type}")
        profile = profile_manager.get_default_profile(media_type)
        print(f"Using default profile: {profile.name}")

    output_base = Path(args.output) if args.output else folder.parent / f"{folder.name}_sorted"
    print(f"Output directory: {output_base}")

    # Create output dirs
    if profile:
        dirs = {cat.name: output_base / cat.name for cat in profile.categories}
    else:
        dirs = {n: output_base / n for n in ("Share", "Storage", "Review", "Ignore")}
    dirs["Reports"] = output_base / "Reports"
    if not config.classification.classify_videos:
        dirs["Videos"] = output_base / "Videos"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    skip_cache = {"gemini", "burst_context"} if getattr(args, 'no_cache', False) else set()
    if skip_cache:
        print("Classification cache disabled (--no-cache)")
    cache_manager = CacheManager(
        config.paths.cache_root,
        ttl_days=config.caching.ttl_days,
        enabled=config.caching.enabled,
        skip_types=skip_cache,
    )

    try:
        client = create_ai_client(config, provider=args.provider)
        print(f"AI client initialized: {client.provider_name}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    media = FileTypeRegistry.list_all_media(folder)
    total = len(media["images"]) + len(media["videos"]) + len(media["documents"])
    print(f"\nScanning folder: {folder}")
    print(f"   Found {len(media['images'])} images, {len(media['videos'])} videos, {len(media['documents'])} documents")
    if total == 0:
        print("\nNo media files found to process.")
        sys.exit(0)

    import time
    pipeline = MixedPipeline(config, profile, cache_manager, client)
    start = time.time()
    try:
        result = pipeline.run(
            folder, output_base, config.system.dry_run,
            classify_videos=config.classification.classify_videos,
            classify_bundles=True if getattr(args, 'bundles', False) else (False if getattr(args, 'no_bundles', False) else None),
        )
    except FatalAPIError as e:
        _print_fatal_api_error(e)
        sys.exit(1)
    result.elapsed_seconds = time.time() - start

    if result.results:
        report_dir = dirs.get("Reports", output_base / "Reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        # Inline CSV report
        import csv
        from datetime import datetime

        rows = []
        for r in result.results:
            c = r.get("classification", {})
            rows.append({
                "source": str(r["path"]),
                "destination": r.get("destination", "Unknown"),
                "classification": c.get("classification", "Unknown"),
                "score": c.get("score", 0),
                "reasoning": c.get("reasoning", ""),
            })
        report_path = report_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if rows:
            import pandas as pd
            pd.DataFrame(rows).to_csv(report_path, index=False)
            print(f"\nReport saved: {report_path}")
        result.generate_summary_report(report_dir, provider_name=client.provider_name)

    result.print_summary("media files")
    if config.system.dry_run:
        print("\nThis was a dry run. No files were moved.")
    print("\nClassification complete!")


def _cmd_train(args: argparse.Namespace) -> None:
    """Launch Gradio pairwise trainer."""
    from taster.compat import require

    require("gradio", "train")

    # Delegate to existing trainer
    sys.argv = ["taste_trainer.py", args.folder]
    from taster.dirs import get_env_path
    from dotenv import load_dotenv

    load_dotenv()
    user_env = get_env_path()
    if user_env.exists():
        load_dotenv(user_env)

    # Import and run the trainer UI
    from taster.core.config import load_config, BurstDetectionConfig
    from taster.core.file_utils import FileTypeRegistry, ImageUtils
    from taster.core.profiles import ProfileManager
    from taster.features.burst_detector import BurstDetector
    from taster.training.session import TrainingSession
    from taster.training.sampler import ComparisonSampler

    print(f"Launching Gradio trainer for: {args.folder}")
    print("This will open a browser window for pairwise training.")

    # The taste_trainer.py script creates the Gradio UI - import it
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "taste_trainer", Path(__file__).parent.parent / "taste_trainer.py"
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        print("Error: Could not find taste_trainer.py")
        sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    import asyncio
    import os

    from taster.compat import require

    require("mcp", "mcp")

    from dotenv import load_dotenv

    load_dotenv()
    from taster.dirs import get_env_path

    user_env = get_env_path()
    if user_env.exists():
        load_dotenv(user_env)

    # Windows encoding fix
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    from taster.mcp.server import create_mcp_server

    server = create_mcp_server()
    init_options = server.create_initialization_options()

    async def run():
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)

    asyncio.run(run())


def _cmd_status(args: argparse.Namespace) -> None:
    """Show config location, profiles, and API key status."""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    from taster import __version__
    from taster.dirs import get_config_dir, find_config, find_profiles_dir, get_env_path

    user_env = get_env_path()
    if user_env.exists():
        load_dotenv(user_env)

    print(f"Taster v{__version__}")
    print()

    # Config location
    config_dir = get_config_dir()
    print(f"Config directory:  {config_dir}")
    print(f"  exists: {'yes' if config_dir.exists() else 'no'}")

    cfg_path = find_config()
    if cfg_path:
        print(f"Active config:     {cfg_path}")
    else:
        print("Active config:     (defaults -- no config file found)")

    # Profiles
    profiles_dir = find_profiles_dir()
    print(f"Profiles directory: {profiles_dir}")
    if profiles_dir.exists():
        profiles = list(profiles_dir.glob("*.json"))
        print(f"  {len(profiles)} profile(s): {', '.join(p.stem for p in profiles)}")
    else:
        print("  (not created yet)")

    # API keys
    print()
    keys = {
        "GEMINI_API_KEY": "Gemini",
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "LOCAL_LLM_URL": "Local LLM",
    }
    for env_var, label in keys.items():
        val = os.environ.get(env_var, "")
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 10 else "***"
            print(f"  {label}: {masked}")
        else:
            print(f"  {label}: not set")

    # .env location
    if user_env.exists():
        print(f"\n.env file: {user_env}")
    local_env = Path(".env")
    if local_env.exists():
        print(f".env file: {local_env.resolve()} (local)")


# ── main ────────────────────────────────────────────────────────────


_COMMANDS = {
    "init": _cmd_init,
    "classify": _cmd_classify,
    "train": _cmd_train,
    "serve": _cmd_serve,
    "status": _cmd_status,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = _COMMANDS.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
