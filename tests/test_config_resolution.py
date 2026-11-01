"""Tests for config resolution — load_config() with the new resolution order."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from taster.core.config import load_config, Config
from taster.core.profiles import ProfileManager


class TestLoadConfigResolution:
    """Test that load_config() uses the correct resolution order."""

    def test_returns_defaults_when_no_config_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("taster.dirs.get_config_dir", return_value=tmp_path / "empty"):
            config = load_config()
        assert isinstance(config, Config)
        assert config.model.name == "gemini-3.8-flash"

    def test_loads_from_explicit_path(self, tmp_path):
        cfg = tmp_path / "custom.yaml"
        cfg.write_text(yaml.dump({"model": {"name": "custom-model"}}))
        config = load_config(cfg)
        assert config.model.name == "custom-model"

    def test_explicit_path_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")

    def test_finds_local_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "config.yaml"
        local.write_text(yaml.dump({"model": {"name": "local-model"}}))
        config = load_config()
        assert config.model.name == "local-model"

    def test_local_over_user_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "config.yaml"
        local.write_text(yaml.dump({"model": {"name": "local"}}))
        user_cfg = tmp_path / "user" / "config.yaml"
        user_cfg.parent.mkdir()
        user_cfg.write_text(yaml.dump({"model": {"name": "user"}}))
        with patch("taster.dirs.get_config_dir", return_value=tmp_path / "user"):
            config = load_config()
        assert config.model.name == "local"

    def test_falls_back_to_user_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        user_cfg = tmp_path / "user_config" / "config.yaml"
        user_cfg.parent.mkdir()
        user_cfg.write_text(yaml.dump({"model": {"name": "user-model"}}))
        with patch("taster.dirs.get_config_dir", return_value=tmp_path / "user_config"):
            config = load_config()
        assert config.model.name == "user-model"

    def test_defaults_are_valid_config(self):
        """Defaults (no file) produce a fully usable Config."""
        with (
            patch("taster.dirs.find_config", return_value=None),
        ):
            config = load_config()
        assert isinstance(config, Config)
        assert 1 <= config.classification.share_threshold <= 5
        assert config.profiles.active_profile == "default-photos"

    def test_partial_config_fills_defaults(self, tmp_path):
        cfg = tmp_path / "partial.yaml"
        cfg.write_text(yaml.dump({"model": {"name": "partial-model"}}))
        config = load_config(cfg)
        assert config.model.name == "partial-model"
        # Other sections use defaults
        assert config.classification.share_threshold == 4
        assert config.caching.enabled is True


class TestProfileManagerResolution:
    """Test that ProfileManager finds profiles in the right directory."""

    def test_loads_from_local_profiles_dir(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        import json
        profile_data = {
            "name": "test",
            "description": "Test profile",
            "media_types": ["image"],
            "categories": [{"name": "Good", "description": "Good"}],
        }
        (profiles_dir / "test.json").write_text(json.dumps(profile_data))
        pm = ProfileManager(str(profiles_dir))
        assert pm.profile_exists("test")
        profile = pm.load_profile("test")
        assert profile.name == "test"

    def test_profile_not_found_raises(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        pm = ProfileManager(str(profiles_dir))
        with pytest.raises(FileNotFoundError):
            pm.load_profile("nonexistent")

    def test_list_profiles(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        import json
        for name in ("alpha", "beta"):
            data = {
                "name": name,
                "description": f"{name} profile",
                "media_types": ["image"],
                "categories": [{"name": "Good", "description": "Good"}],
            }
            (profiles_dir / f"{name}.json").write_text(json.dumps(data))
        pm = ProfileManager(str(profiles_dir))
        profiles = pm.list_profiles()
        names = [p.name for p in profiles]
        assert "alpha" in names
        assert "beta" in names

    def test_invalid_json_raises(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "bad.json").write_text("not valid json{{{")
        pm = ProfileManager(str(profiles_dir))
        with pytest.raises(Exception):
            pm.load_profile("bad")
