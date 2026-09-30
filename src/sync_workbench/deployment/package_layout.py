"""Shared package identity, path and integrity rules (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

SCHEMA = "syncwb.student_package.v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def local_path(root, ref):
    """Resolve a portable reference without allowing traversal or drive paths."""
    if not isinstance(ref, str) or not ref or "\\" in ref or ":" in ref:
        raise ValueError(f"Non-portable package path: {ref!r}")
    rel = PurePosixPath(ref)
    if rel.is_absolute() or PureWindowsPath(ref).drive or ".." in rel.parts:
        raise ValueError(f"Non-portable package path: {ref!r}")
    root = Path(root).resolve()
    path = (root / ref).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path escapes package root: {ref!r}")
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported student-package format")
    expected = manifest.get("manifest_sha256")
    if expected != json_digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Package manifest checksum mismatch")
    return manifest


def verify_package(root, *, full=True):
    """Immutable files only; student work and identity are intentionally mutable."""
    root = Path(root).resolve()
    manifest = read_manifest(root)
    seen = set()
    for entry in manifest["files"]:
        ref = entry["path"]
        if ref in seen or ref.startswith("work/") or ref == "manifest.json":
            raise ValueError("Invalid immutable-file inventory")
        seen.add(ref)
        path = local_path(root, ref)
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Package file missing or size changed: {ref}")
        if full or entry["role"] not in {"asset", "database_template"}:
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Package file checksum mismatch: {ref}")
    if not {"config.json", "database/template.sqlite"}.issubset(seen):
        raise ValueError("Incomplete package inventory")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("package_id") != manifest["package_id"] or config.get("assignment") != manifest["assignment"]:
        raise ValueError("Configuration/manifest assignment mismatch")
    expected_paths = {"database_template": "database/template.sqlite", "working_database": "work/workbench.sqlite",
                      "artifact_root": "assets/artifacts", "rgb_root": "assets/rgb"}
    if config.get("paths") != expected_paths:
        raise ValueError("Unsupported student package paths")
    for ref in config["paths"].values():
        local_path(root, ref)
    for asset in manifest["assets"]:
        if asset["path"] not in seen:
            raise ValueError("Asset omitted from immutable inventory")
    return {"package_id": manifest["package_id"], "manifest_sha256": manifest["manifest_sha256"],
            "verified_files": len(seen), "full_checksums": full}
