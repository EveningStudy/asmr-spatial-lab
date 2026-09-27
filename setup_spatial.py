"""Download official inference assets only; never execute upstream setup scripts."""

import hashlib
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent / "work/spatial-assets"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        locked = json.loads(
            (Path(__file__).parent / "spatial-assets.lock.json").read_text(encoding="utf-8")
        )
        urls = {name: data["url"] for name, data in locked["files"].items()}
        manifest = {"meta_revision": locked["meta_revision"], "files": {}}
        for name, url in urls.items():
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                response = client.get(url)
                if name.endswith("__init__.py") and response.status_code == 404:
                    continue
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != locked["files"][name]["sha256"]:
                    raise ValueError(f"Download hash mismatch: {name}")
                temporary = target.with_suffix(target.suffix + ".download")
                temporary.write_bytes(response.content)
                temporary.replace(target)
            payload = target.read_bytes()
            if hashlib.sha256(payload).hexdigest() != locked["files"][name]["sha256"]:
                raise ValueError(f"Existing asset hash mismatch: {name}")
            manifest["files"][name] = {
                "url": url,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            print(name, len(payload), flush=True)
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
