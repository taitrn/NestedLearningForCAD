# Paper Benchmark Plan

This plan turns the current Meta-NATH CAD demo into a benchmark that can be
defended in a paper-style discussion.

## Why This Is Needed

The current full demo validates that the method runs on all MVTec tasks, saves
checkpoints, evaluates before/after Phase 3, and passes the configured
acceptance gates. That is necessary, but it is not enough for a claim such as
"outperforms UCAD" or "beats ReplayCAD".

For that kind of claim, every method must be compared under the same continual
task protocol, metric definitions, data-access rules, and task order. Otherwise
the table is only contextual, not a direct benchmark.

## Benchmark Stages

### Stage 1 - Lock the Protocol

Use `docs/benchmark_protocol.md` as the source of truth.

Required decisions:

| Item | Current recommendation |
| :--- | :--- |
| Main dataset | MVTec AD |
| Task count | 15 |
| Task order | `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper` |
| Backbone | `facebook/dinov2-base` |
| Memory | CADIC normal-only bounded coreset |
| Replay | No diffusion replay; utility-guided anchor replay during Phase 3 |
| Metrics | Image-AUROC Avg/FM and Pixel-AP Avg/FM |

### Stage 2 - Generate Ours Rows

For a benchmark row, run the full task stream with `forgetting_matrix` enabled.

MVTec 15-task command:

```bash
MAIN_MAX_TASKS=15 EXPERIMENTAL_MAX_TASKS=15 bash scripts/run_full_demo.sh
```

Then compute the paper-style metrics from the run artifact:

```bash
python scripts/benchmark/compute_replaycad_metrics.py \
  results/<run_dir>/task_records.json \
  --dataset MVTec \
  --method Meta-NATH \
  --metrics image_auroc pixel_aupr \
  --output results/<run_dir>/benchmark_metrics.json
```

For a Phase 3 row, do not compute FM directly from `after_eval_dir`, because
that artifact is a fixed-checkpoint evaluation sweep. Merge the sequential
warmup history with the post-Phase-3 final row first:

```bash
python scripts/benchmark/merge_phase3_benchmark.py \
  --history-task-records results/<warmup_dir>/task_records.json \
  --final-eval-task-records results/<after_eval_dir>/task_records.json \
  --output results/<phase3_dir>/phase3_benchmark_task_records.json \
  --label Meta-NATH-Phase3

python scripts/benchmark/compute_replaycad_metrics.py \
  results/<phase3_dir>/phase3_benchmark_task_records.json \
  --dataset MVTec \
  --method Meta-NATH-Phase3 \
  --metrics image_auroc pixel_aupr \
  --output results/<phase3_dir>/benchmark_metrics.json
```

Use the Phase 1-2 anchor run and the merged Phase 3 artifacts as separate rows
only if the report clearly explains what each row represents.

### Stage 3 - Add Ablations

At minimum, prepare:

| Row | Purpose |
| :--- | :--- |
| Frozen DINOv2 + patch NN only | Base anomaly detector |
| + TITANS memory | Fast-memory contribution |
| + ACC gating | Memory admission contribution |
| + CADIC coreset | Bounded normal memory contribution |
| + Conservative Phase 3 | Reportable consolidation |
| + NSP2/CBP/Subspace Recycling | Experimental max-power stack |

The max-power row should not be used to assign credit to NSP2, CBP, or Subspace
Recycling until these ablations exist.

### Stage 4 - Baselines

There are two acceptable baseline tiers.

Reproduced baselines:

- Run the baseline code locally with the same task protocol.
- Compute the same Avg/FM metrics.
- Put them in the direct comparison table.

Paper-reported references:

- Copy values from the original paper table only for context.
- Label them as "paper-reported".
- Do not describe them as reproduced results.

Recommended first reference methods:

| Method family | Methods |
| :--- | :--- |
| CAD-specific | UCAD, IUF, CDAD, DNE |
| AD adapted to CAD | PatchCore, SimpleNet, MambaAD, InvAD |
| AD + continual learning | SimpleNet+EWC/MAS, MambaAD+EWC/MAS |
| Generative replay | ReplayCAD |

### Stage 5 - VisA

Do not use VisA in the main claim until:

1. VisA data is mounted under the configured path.
2. The full 12-task run completes.
3. `task_records.json` contains the final row for all 12 tasks.
4. Image-AUROC Avg/FM and Pixel-AP Avg/FM are computed.
5. The artifact zip is archived.

### Stage 6 - Multi-Order Robustness

For a stronger paper submission, repeat MVTec with multiple task orders. The
minimum defensible version is:

- Default semantic order.
- At least three random class orders.
- Mean and standard deviation for Avg/FM.

This is not required for a class report, but it is important for a serious
paper-style benchmark.

## Claim Boundaries

Safe current claim:

```text
Meta-NATH CAD is a lightweight, bounded-memory continual anomaly detection
pipeline. On the verified MVTec 15-task run, it completes the full task stream
and passes before/after Phase 3 acceptance gates.
```

Unsafe claim until benchmark baselines are reproduced:

```text
Meta-NATH outperforms ReplayCAD, UCAD, CDAD, IUF, or other CAD baselines.
```

Safer comparison phrasing:

```text
Compared with paper-reported CAD baselines, our current MVTec numbers are
reported as contextual references. Direct superiority claims require rerunning
the baselines under the same protocol.
```

## Deliverables

The benchmark-ready package should contain:

1. `task_records.json` for each reported row.
2. `benchmark_metrics.json` generated by `compute_replaycad_metrics.py`.
3. `phase3_benchmark_task_records.json` for each reported Phase 3 row.
4. A Markdown or LaTeX benchmark table generated from local rows.
5. A short protocol paragraph in the report.
6. A limitation paragraph distinguishing demo validation, local benchmark rows,
   reproduced baselines, and paper-reported references.
