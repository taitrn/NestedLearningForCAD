# Scripts

Start here for submission:

```bash
bash scripts/run_full_demo.sh
```

## Layout

- `run_full_demo.sh`: primary 3-tier MVTec demo workflow.
- `run_server_phase3.sh`: conservative-only MVTec Phase 3 workflow wrapper.
- `run_server_visa.sh`: VisA conservative + mechanism + experimental/max-power Phase 3 workflow wrapper.
- `baselines/`: local baseline adapters that emit benchmark-compatible artifacts.
- `benchmark/`: paper-style CAD Avg/FM metric collection from saved artifacts.
- `pipeline/`: reproducible CLIs used by the full demo.
- `diagnostics/`: smoke tests, summaries, GPU checks, and metric helpers.
- `workflows/`: optional server workflows. Root-level `run_server_*.sh` files are thin compatibility wrappers.

## Pipeline CLIs

- `pipeline/run_phase3_consolidation.py`: Phase 3 N2B-NC consolidation.
- `pipeline/evaluate_checkpoint.py`: evaluate a saved checkpoint without retraining.
- `pipeline/phase3_acceptance.py`: before/after metric-gated acceptance report.
- `pipeline/compare_checkpoint_scores.py`: slow score-distribution diagnostics, not part of the default demo.

## Diagnostics

- `diagnostics/mechanism_smoke.py`: TITANS, CADIC, ACC, NSP2, CBP, and Subspace Recycling smoke test.
- `diagnostics/preflight_backbone.py`: pre-cache/verify the HuggingFace backbone before long Kaggle/server workflows.
- `diagnostics/summarize_run.py`: markdown summary for a result directory.
- `diagnostics/compute_forgetting.py`: forgetting metric from an evaluation matrix.
- `diagnostics/check_gpu.py`: environment/GPU check.

## Benchmark

- `benchmark/compute_replaycad_metrics.py`: compute Image-AUROC Avg/FM and Pixel-AP Avg/FM from `task_records.json` or `forgetting_matrix.json`.
- `benchmark/merge_phase3_benchmark.py`: merge sequential warmup history with a post-Phase-3 final evaluation row before computing Phase 3 Avg/FM.
- `benchmark/collect_benchmark_table.py`: collect local benchmark rows into a Markdown table.
- `baselines/patchcore_cad.py`: lightweight PatchCore-style CAD baseline adapter. This is not an official PatchCore reproduction, but it follows the local sequential task protocol and writes `task_records.json`.

Example:

```bash
python scripts/benchmark/compute_replaycad_metrics.py results/<run_dir>/task_records.json --dataset MVTec --method Meta-NATH --print-markdown-row
```

Phase 3 benchmark row:

```bash
python scripts/benchmark/merge_phase3_benchmark.py --history-task-records results/<warmup_dir>/task_records.json --final-eval-task-records results/<after_eval_dir>/task_records.json --output results/<phase3_dir>/phase3_benchmark_task_records.json
python scripts/benchmark/compute_replaycad_metrics.py results/<phase3_dir>/phase3_benchmark_task_records.json --dataset MVTec --method Meta-NATH-Phase3 --print-markdown-row
```

PatchCore-style local baseline:

```bash
python scripts/baselines/patchcore_cad.py --config conf/full_demo.yaml --max_tasks 15 --run_suffix mvtec_full15
python scripts/benchmark/compute_replaycad_metrics.py results/<PatchCoreCAD_run>/task_records.json --dataset MVTec --method "PatchCore-style" --print-markdown-row
```

## Workflows

- `workflows/run_server_phase3.sh`: conservative-only MVTec Phase 3 workflow.
- `workflows/run_server_visa.sh`: optional VisA full Phase 3 workflow once `data/visa` is mounted.

For a 15-task max-power MVTec stress test:

```bash
MAIN_MAX_TASKS=15 EXPERIMENTAL_MAX_TASKS=15 EXPERIMENTAL_CONFIG=conf/mvtec_max_power.yaml bash scripts/run_full_demo.sh
```

`run_full_demo.sh` preflights the configured HuggingFace backbone by default,
then switches subsequent steps to local cache mode. Disable only for debugging:

```bash
PREFLIGHT_HF_BACKBONE=0 bash scripts/run_full_demo.sh
```

For a VisA max-power stress test:

```bash
EXPERIMENTAL_CONFIG=conf/visa_max_power.yaml bash scripts/run_server_visa.sh
```
