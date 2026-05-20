"""Preflight HuggingFace backbone cache before long CAD workflows.

Kaggle and other notebook runtimes can occasionally stall while a child process
downloads or materializes a HuggingFace checkpoint. This script performs that
work once, up front, so the expensive demo workflow can switch to
``METANATH_LOCAL_FILES_ONLY=1`` afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _cache_roots() -> list[Path]:
    roots = []
    for env_name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())

    if "HF_HOME" not in os.environ and "HUGGINGFACE_HUB_CACHE" not in os.environ:
        roots.append(Path.home() / ".cache" / "huggingface")

    deduped: list[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if str(resolved) not in seen:
            deduped.append(root)
            seen.add(str(resolved))
    return deduped


def clean_hf_locks(max_age_seconds: float = 0.0) -> list[str]:
    """Remove HuggingFace lock files under known cache roots.

    This is deliberately scoped to the HF cache directories. The demo workflow
    runs one model-loading process at a time, so removing stale notebook-era
    locks is safer than letting a 4-hour workflow stall before task 0.
    """

    removed: list[str] = []
    now = time.time()
    for root in _cache_roots():
        if not root.exists():
            continue
        for lock_path in root.rglob("*.lock"):
            try:
                if max_age_seconds > 0:
                    age = now - lock_path.stat().st_mtime
                    if age < max_age_seconds:
                        continue
                lock_path.unlink()
                removed.append(str(lock_path))
            except OSError as exc:
                print(f"[preflight] could not remove lock {lock_path}: {exc}", flush=True)
    return removed


def backbones_from_configs(config_paths: Iterable[str]) -> list[str]:
    backbones: list[str] = []
    for config_text in config_paths:
        config_path = Path(config_text)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        backbone = str(config.get("model", {}).get("backbone", "")).strip()
        if backbone:
            backbones.append(backbone)
    return backbones


def load_backbone(backbone: str, local_files_only: bool, retry_force_download: bool) -> None:
    # Set these before importing transformers/huggingface_hub.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoModel

    print(
        f"[preflight] loading backbone={backbone!r} local_files_only={local_files_only}",
        flush=True,
    )
    try:
        model = AutoModel.from_pretrained(
            backbone,
            local_files_only=local_files_only,
        )
    except Exception:
        if local_files_only or not retry_force_download:
            raise
        print("[preflight] first load failed; retrying once with force_download=True", flush=True)
        model = AutoModel.from_pretrained(
            backbone,
            local_files_only=False,
            force_download=True,
        )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[preflight] cached OK: {backbone} params={param_count}", flush=True)
    del model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight/cache HF vision backbone.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config to inspect for model.backbone. Can be passed more than once.",
    )
    parser.add_argument(
        "--backbone",
        action="append",
        default=[],
        help="Explicit HuggingFace backbone id. Can be passed more than once.",
    )
    parser.add_argument(
        "--local-files-only",
        choices=["auto", "0", "1"],
        default="auto",
        help="auto follows METANATH_LOCAL_FILES_ONLY; 0 may download; 1 verifies cache only.",
    )
    parser.add_argument(
        "--clean-locks",
        action="store_true",
        help="Remove HuggingFace *.lock files under known cache roots before loading.",
    )
    parser.add_argument(
        "--lock-max-age-seconds",
        type=float,
        default=0.0,
        help="Only remove lock files older than this many seconds. 0 removes all locks.",
    )
    parser.add_argument(
        "--retry-force-download",
        action="store_true",
        help="If online load fails, retry once with force_download=True.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backbones = []
    backbones.extend(backbones_from_configs(args.config))
    backbones.extend(args.backbone)
    if not backbones:
        backbones = ["facebook/dinov2-base"]

    unique_backbones = []
    seen = set()
    for backbone in backbones:
        if backbone not in seen:
            unique_backbones.append(backbone)
            seen.add(backbone)

    if args.clean_locks:
        removed = clean_hf_locks(max_age_seconds=args.lock_max_age_seconds)
        print(f"[preflight] removed_hf_locks={len(removed)}", flush=True)
        for lock_path in removed[:20]:
            print(f"[preflight] removed lock: {lock_path}", flush=True)
        if len(removed) > 20:
            print(f"[preflight] ... {len(removed) - 20} more lock(s)", flush=True)

    if args.local_files_only == "auto":
        local_files_only = _truthy(os.environ.get("METANATH_LOCAL_FILES_ONLY"))
    else:
        local_files_only = args.local_files_only == "1"

    for backbone in unique_backbones:
        load_backbone(
            backbone=backbone,
            local_files_only=local_files_only,
            retry_force_download=args.retry_force_download,
        )

    print("[preflight] all requested backbones are ready", flush=True)


if __name__ == "__main__":
    main()
