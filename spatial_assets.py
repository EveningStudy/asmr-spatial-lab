"""Validate pinned downloaded code/weights before subprocess import."""

import hashlib
import json
from pathlib import Path


def verify_assets():
    root = Path(__file__).resolve().parent
    lock = json.loads((root / "spatial-assets.lock.json").read_text(encoding="utf-8"))
    for name, details in lock["files"].items():
        path = root / "work/spatial-assets" / name
        if not path.is_file():
            raise ValueError(f"缺少资源 {name}；先运行 setup-spatial.ps1")
        if hashlib.sha256(path.read_bytes()).hexdigest() != details["sha256"]:
            raise ValueError(f"资源校验失败 {name}；请恢复锁定版本")
    return lock
