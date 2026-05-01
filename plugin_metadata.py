from __future__ import annotations

from pathlib import Path

import yaml

PLUGIN_ROOT = Path(__file__).resolve().parent


def _load_plugin_metadata() -> dict:
    """从 metadata.yaml 读取插件发布元信息，保证注册信息与发布配置同源。"""
    metadata_path = PLUGIN_ROOT / "metadata.yaml"
    return yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}


PLUGIN_METADATA = _load_plugin_metadata()
PLUGIN_NAME = str(PLUGIN_METADATA.get("name") or PLUGIN_ROOT.name)
PLUGIN_AUTHOR = str(PLUGIN_METADATA.get("author") or "")
PLUGIN_DESC = str(PLUGIN_METADATA.get("desc") or "")
PLUGIN_VERSION = str(PLUGIN_METADATA.get("version") or "")
PLUGIN_REPO = str(PLUGIN_METADATA.get("repo") or "")
