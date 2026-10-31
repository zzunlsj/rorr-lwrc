"""스펙 설정(config/*.toml) 로더. 코드에 상수를 두지 않는다 (정본 v1.0 §9)."""

import hashlib
import tomllib
from pathlib import Path

from common.paths import MANIFEST, ROOT

CONFIG_DIR = ROOT / "config"


def load(name: str) -> dict:
    path = Path(name) if Path(name).suffix else CONFIG_DIR / f"{name}.toml"
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = path.as_posix()
    return cfg


def data_snapshot() -> str:
    """manifest/received.csv 해시 — 모든 L3 출력에 남긴다 (CLAUDE.md §4)."""
    return hashlib.sha256((MANIFEST / "received.csv").read_bytes()).hexdigest()[:16]
