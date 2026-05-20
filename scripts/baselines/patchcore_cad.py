"""PatchCore-style continual anomaly detection baseline.

This adapter is intentionally separate from the main Meta-NATH workflow. It
uses the repository's continual task protocol, frozen backbone feature extractor,
bounded normal-only memory, and patch nearest-neighbor scoring so its artifacts
can be fed into ``scripts/benchmark/compute_replaycad_metrics.py``.

It is not an official PatchCore reproduction. Use it as a local lightweight
baseline row unless you later replace it with the upstream PatchCore code and
the exact paper settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import yaml
from tqdm import tqdm

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager
from models.meta_nath_core import MetaNATHCore
from training.meta_nath_engine import MetaNATHEngine
from training.run_experiment import (
    _compute_forgetting_metrics,
    _model_geometry,
    _truncate_tasks_if_needed,
    apply_profile,
)
from utils.global_seed import set_seed


class PatchCoreCADCore(MetaNATHCore):
    """MetaNATH feature extractor with PatchCore-style normal memory updates."""

    def forward(
        self,
        x: torch.Tensor,
        task_id: int = 0,
        update_coreset: bool = True,
    ) -> Dict[str, Any]:
        z_cls, z_patches, patch_grid = self.extract_features(x)

        n_updated = 0
        if update_coreset:
            n_updated = self.coreset.update_batch(
                cls_embs=z_cls.detach(),
                patch_embs_batch=z_patches.detach(),
                images=None,
                task_id=task_id,
            )

        return {
            "z_cls": z_cls,
            "z_updated": z_cls,
            "z_patches": z_patches,
            "surprise": 0.0,
            "acc_score": 1.0,
            "approved": True,
            "coreset_updated": n_updated > 0,
            "coreset_n_updated": n_updated,
            "patch_grid": patch_grid,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight PatchCore-style CAD baseline."
    )
    parser.add_argument("--config", type=str, default="conf/full_demo.yaml")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "tiny", "small"],
        help="Runtime profile matching training/run_experiment.py.",
    )
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--run_suffix", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--synthetic_train_probability",
        type=float,
        default=0.0,
        help=(
            "Probability of synthetic train anomalies. PatchCore-style memory "
            "defaults to 0.0 so it indexes all normal train images."
        ),
    )
    return parser.parse_args()


def build_model(config: Dict[str, Any]) -> PatchCoreCADCore:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    n_patch_cfg = model_cfg.get("n_patch", None)
    n_patch = None if n_patch_cfg is None else int(n_patch_cfg)

    return PatchCoreCADCore(
        d=int(model_cfg.get("embed_dim", 768)),
        tau_acc=float(model_cfg.get("tau_acc", 0.25)),
        max_coreset_size=int(model_cfg.get("max_coreset_size", 1000)),
        n_patch=n_patch,
        store_images=False,
        device=str(training_cfg.get("device", "cuda")),
        backbone_name=model_cfg.get("backbone", "facebook/dinov2-base"),
    )


def _resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return requested_device


def _prepare_config(
    base_config: Dict[str, Any],
    profile: str,
    max_tasks: Optional[int],
    synthetic_train_probability: float,
) -> Dict[str, Any]:
    config = apply_profile(base_config, profile)
    _truncate_tasks_if_needed(config, max_tasks)
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["synthetic_train_probability"] = float(
        min(max(synthetic_train_probability, 0.0), 1.0)
    )
    return config


def run_patchcore_cad(
    config: Dict[str, Any],
    run_suffix: str = "",
    quiet: bool = False,
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})
    evaluation_cfg = config.get("evaluation", {})
    memory_cfg = config.get("memory", {})

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PatchCoreCAD_{timestamp}"
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"

    results_dir = logging_cfg.get("results_dir", "results")
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    requested_device = str(training_cfg.get("device", "cuda"))
    device = _resolve_device(requested_device)
    config.setdefault("training", {})["device"] = device

    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = {
        "method": "PatchCore-style CAD",
        "claim_boundary": (
            "Local lightweight baseline adapter, not an official PatchCore "
            "paper reproduction."
        ),
        "normal_memory": True,
        "phase3": False,
        "synthetic_train_probability": config.get("dataset", {}).get(
            "synthetic_train_probability", 0.0
        ),
    }
    with open(os.path.join(run_dir, "baseline_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    nearest_neighbors = int(memory_cfg.get("nearest_neighbors", 2))
    pixel_score_norm = str(evaluation_cfg.get("pixel_score_norm", "none")).lower()
    gaussian_smoothing_sigma = float(evaluation_cfg.get("gaussian_smoothing_sigma", 0.0))

    model = build_model(config)
    engine = MetaNATHEngine(
        model=model,
        device=device,
        nearest_neighbors=nearest_neighbors,
        pixel_score_norm=pixel_score_norm,
        gaussian_smoothing_sigma=gaussian_smoothing_sigma,
    )
    stream_manager = ContinualStreamingManager(config)

    epochs_per_task = int(training_cfg.get("stream_repeats", training_cfg.get("epochs_per_task", 1)))
    pixel_sample_limit = int(evaluation_cfg.get("pixel_sample_limit", 10000))
    eval_mode = str(evaluation_cfg.get("mode", "current")).lower()
    cumulative_frequency = int(evaluation_cfg.get("cumulative_frequency", 5))
    final_cumulative = bool(evaluation_cfg.get("final_cumulative", True))
    forgetting_matrix_enabled = bool(evaluation_cfg.get("forgetting_matrix", False))
    forgetting_metric = str(evaluation_cfg.get("forgetting_metric", "image_auroc"))

    task_records = []
    forgetting_matrix: Dict[str, Dict[str, float]] = {}

    total_tasks = len(stream_manager.categories)
    task_pbar = tqdm(total=total_tasks, desc="PatchCore CAD tasks", unit="task", disable=quiet)
    while True:
        train_loader, test_loader, task_info = stream_manager.get_next_task()
        if train_loader is None:
            break

        task_id = int(task_info["task_id"])
        category = str(task_info["category"])

        task_pbar.set_description(f"PatchCore task {task_id}: {category}")
        if not quiet:
            print(f"\n===== PatchCore-style Task {task_id}: {category} =====")

        train_metrics = engine.train_task(
            train_loader=train_loader,
            task_id=task_id,
            epochs=epochs_per_task,
            verbose=not quiet,
        )
        eval_metrics = engine.evaluate_task(
            test_loader=test_loader,
            task_id=task_id,
            verbose=not quiet,
            pixel_sample_limit=pixel_sample_limit,
        )

        cumulative_eval_metrics = None
        should_cumulative_eval = (
            eval_mode != "cumulative"
            and cumulative_frequency > 0
            and (task_id + 1) % cumulative_frequency == 0
        )
        if should_cumulative_eval:
            cumulative_loader = stream_manager.get_cumulative_test_loader()
            if cumulative_loader is not None:
                cumulative_eval_metrics = engine.evaluate_task(
                    test_loader=cumulative_loader,
                    task_id=task_id,
                    verbose=not quiet,
                    pixel_sample_limit=pixel_sample_limit,
                )

        record = {
            "task_id": task_id,
            "category": category,
            "method": "PatchCore-style CAD",
            "model": _model_geometry(model),
            "train": train_metrics,
            "eval": eval_metrics,
        }
        if cumulative_eval_metrics is not None:
            record["cumulative_eval"] = cumulative_eval_metrics

        if forgetting_matrix_enabled:
            row_metrics: Dict[str, Dict[str, Any]] = {}
            row_scores: Dict[str, float] = {}
            n_prev = len(stream_manager.test_datasets_history)
            fm_pbar = tqdm(
                range(n_prev),
                desc=f"  PatchCore forgetting matrix (task {task_id})",
                unit="eval",
                leave=False,
                disable=quiet,
            )
            for prev_task_id in fm_pbar:
                if prev_task_id == task_id and eval_mode != "cumulative":
                    prev_metrics = eval_metrics
                else:
                    prev_loader = stream_manager.get_test_loader_for_task(prev_task_id)
                    if prev_loader is None:
                        continue
                    prev_metrics = engine.evaluate_task(
                        test_loader=prev_loader,
                        task_id=prev_task_id,
                        verbose=False,
                        pixel_sample_limit=pixel_sample_limit,
                    )

                row_metrics[str(prev_task_id)] = prev_metrics
                row_scores[str(prev_task_id)] = float(prev_metrics.get(forgetting_metric, 0.0))

            record["forgetting_eval"] = row_metrics
            forgetting_matrix[str(task_id)] = row_scores

        task_records.append(record)
        task_pbar.update(1)

        with open(os.path.join(run_dir, f"task_{task_id:02d}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    task_pbar.close()

    final_cumulative_metrics = None
    if task_records and final_cumulative and eval_mode != "cumulative":
        if len(task_records) == 1:
            final_cumulative_metrics = task_records[-1]["eval"]
        elif "cumulative_eval" in task_records[-1]:
            final_cumulative_metrics = task_records[-1]["cumulative_eval"]
        else:
            if not quiet:
                print("\n===== PatchCore Final Cumulative Evaluation =====")
            cumulative_loader = stream_manager.get_cumulative_test_loader()
            if cumulative_loader is not None:
                final_cumulative_metrics = engine.evaluate_task(
                    test_loader=cumulative_loader,
                    task_id=int(task_records[-1]["task_id"]),
                    verbose=not quiet,
                    pixel_sample_limit=pixel_sample_limit,
                )

        if final_cumulative_metrics is not None:
            with open(os.path.join(run_dir, "final_cumulative_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(final_cumulative_metrics, f, indent=2)

    eval_image_aurocs = [float(rec["eval"].get("image_auroc", 0.0)) for rec in task_records]
    eval_pixel_aurocs = [float(rec["eval"].get("pixel_auroc", 0.0)) for rec in task_records]
    eval_pixel_auprs = [float(rec["eval"].get("pixel_aupr", 0.0)) for rec in task_records]
    eval_image_aps = [float(rec["eval"].get("image_ap", 0.0)) for rec in task_records]

    avg_eval_image_auroc = float(np.mean(eval_image_aurocs)) if eval_image_aurocs else 0.0
    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "method": "PatchCore-style CAD",
        "tasks_completed": len(task_records),
        "nearest_neighbors": nearest_neighbors,
        "pixel_score_norm": pixel_score_norm,
        "gaussian_smoothing_sigma": gaussian_smoothing_sigma,
        **_model_geometry(model),
        "avg_eval_image_auroc": avg_eval_image_auroc,
        "avg_eval_auroc": avg_eval_image_auroc,
        "avg_eval_pixel_auroc": float(np.mean(eval_pixel_aurocs)) if eval_pixel_aurocs else 0.0,
        "avg_eval_pixel_aupr": float(np.mean(eval_pixel_auprs)) if eval_pixel_auprs else 0.0,
        "avg_eval_image_ap": float(np.mean(eval_image_aps)) if eval_image_aps else 0.0,
    }
    if final_cumulative_metrics is not None:
        summary["final_cumulative_image_auroc"] = float(final_cumulative_metrics.get("image_auroc", 0.0))
        summary["final_cumulative_pixel_aupr"] = float(final_cumulative_metrics.get("pixel_aupr", 0.0))

    if forgetting_matrix_enabled:
        final_task_id = int(task_records[-1]["task_id"]) if task_records else 0
        forgetting_metrics = _compute_forgetting_metrics(forgetting_matrix, final_task_id)
        summary["forgetting_metric"] = forgetting_metric
        summary["forgetting_measure"] = float(forgetting_metrics["forgetting_measure"])
        with open(os.path.join(run_dir, "forgetting_matrix.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": forgetting_metric,
                    "matrix": forgetting_matrix,
                    **forgetting_metrics,
                },
                f,
                indent=2,
            )

    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "task_records.json"), "w", encoding="utf-8") as f:
        json.dump(task_records, f, indent=2)

    print(f"PatchCore-style CAD run completed. Summary: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config = _prepare_config(
        base_config=copy.deepcopy(base_config),
        profile=args.profile,
        max_tasks=args.max_tasks,
        synthetic_train_probability=args.synthetic_train_probability,
    )
    run_suffix = args.run_suffix if args.run_suffix else args.profile
    run_patchcore_cad(config=config, run_suffix=run_suffix, quiet=args.quiet)


if __name__ == "__main__":
    main()
