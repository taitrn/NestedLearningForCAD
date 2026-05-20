# Paper-Grade Continual Anomaly Detection Benchmark Protocol

This document separates the project demo workflow from a paper-grade benchmark
workflow. The demo proves that the pipeline runs end to end. A paper benchmark
must additionally lock the task protocol, metrics, comparison scope, and claim
boundaries.

## Scope

The active benchmark target is continual anomaly detection (CAD) on industrial
inspection datasets:

| Dataset | Tasks | Benchmark role |
| :--- | ---: | :--- |
| MVTec AD | 15 | Main verified dataset path |
| VisA | 12 | Prepared optional dataset path |

MVTec AD is currently the verified path in this repository. VisA should be
reported only after the corresponding 12-task run has been executed, inspected,
and archived.

## Continual Task Protocol

For a dataset with `N` object categories, each category is treated as one
continual task.

At training step `t`:

1. The learner may load only the training split of the current class `X_t`.
2. Full training splits from old tasks `X_0 ... X_{t-1}` must not be loaded into
   the active training dataloader.
3. Old-task knowledge may be represented only through explicitly declared
   bounded memory, such as the CADIC normal-only coreset, or through saved model
   state.
4. Test sets from seen tasks may be used for evaluation only, not for model
   updates.

This means the project is a nested-learning-inspired hybrid CAD system rather
than a pure "no memory" continual learner. The bounded normal-only coreset is
part of the method and must be declared in every benchmark table.

## Training and Evaluation Matrix

After learning task `t`, evaluate the model on every seen test task
`0 ... t`. This creates a lower-triangular matrix:

```text
R[t][j] = score after training task t, evaluated on test task j
```

For a full paper-grade run, the final row `R[N-1]` must contain all tasks.

The repository stores this information in `task_records.json` when
`evaluation.forgetting_matrix: true` is enabled. Each record contains:

- `eval`: metrics for the current task.
- `forgetting_eval`: metrics on all seen tasks after the current training step.

## Metrics

Report the same metric families as ReplayCAD-style CAD tables:

| Metric | Direction | Repository key |
| :--- | :---: | :--- |
| Image-AUROC Avg | higher is better | `image_auroc` |
| Image-AUROC FM | lower is better | `image_auroc` |
| Pixel-AP Avg | higher is better | `pixel_aupr` |
| Pixel-AP FM | lower is better | `pixel_aupr` |

`pixel_aupr` is average precision over pixel anomaly scores and is reported as
Pixel-AP in benchmark tables.

Optional diagnostic metrics such as `pixel_auroc`, `image_ap`, final cumulative
metrics, acceptance deltas, and drift are useful for engineering, but they do
not replace the four benchmark fields above.

## Avg and Forgetting Measure

Let the final task index be `N - 1`.

Final average for metric `m`:

```text
Avg_m = mean_j R_m[N-1][j], for j in 0 ... N-1
```

Forgetting Measure (FM) for metric `m`:

```text
FM_m = mean_j max(0, max_t R_m[t][j] - R_m[N-1][j])
       for j in 0 ... N-2 and t in j ... N-1
```

The reported table values are usually percentages:

```text
Image-AUROC Avg = 100 * Avg_image_auroc
Image-AUROC FM  = 100 * FM_image_auroc
Pixel-AP Avg    = 100 * Avg_pixel_aupr
Pixel-AP FM     = 100 * FM_pixel_aupr
```

## Result Tiers

Use separate result tiers to avoid overclaiming:

| Tier | Meaning | Allowed claim |
| :--- | :--- | :--- |
| Demo run | End-to-end workflow completed | Pipeline is executable and accepted by configured gate |
| Benchmark run | Full CAD task protocol with Avg/FM | Comparable under the stated protocol |
| Reproduced baseline | Baseline code run locally under same protocol | Direct comparison to that baseline |
| Paper-reported reference | Numbers copied from a paper | Context only, not a reproduced comparison |

Do not claim that Meta-NATH outperforms ReplayCAD, UCAD, CDAD, IUF, PatchCore,
SimpleNet, MambaAD, or InvAD unless those methods were evaluated under the same
dataset order, metric definitions, and data-access rules, or the table clearly
labels their values as paper-reported references.

## Required Benchmark Checklist

Before presenting a paper-style table, verify:

1. The run uses the full task count: 15 for MVTec AD or 12 for VisA.
2. `evaluation.forgetting_matrix: true` is enabled.
3. The final row in `task_records.json` contains every task.
4. Image-AUROC Avg/FM and Pixel-AP Avg/FM are computed from the same matrix.
5. The table declares the backbone, memory type, replay type, task order, and
   whether baselines are reproduced or paper-reported.
6. The report states that the current method uses bounded normal-only coreset
   memory and utility-guided anchor replay, not diffusion-based generative
   replay.

## Recommended Commands

Compute ReplayCAD-style metrics for one run:

```bash
python scripts/benchmark/compute_replaycad_metrics.py \
  results/<run_dir>/task_records.json \
  --dataset MVTec \
  --method Meta-NATH \
  --metrics image_auroc pixel_aupr \
  --output results/<run_dir>/benchmark_metrics.json
```

Collect one or more local rows into a Markdown table:

```bash
python scripts/benchmark/collect_benchmark_table.py \
  --dataset MVTec \
  --row "Meta-NATH=results/<run_dir>/task_records.json" \
  --output docs/benchmark_results_mvtec.md
```

The generated table is a local benchmark table for the rows supplied to the
script. If external paper numbers are added manually, label them as
paper-reported references.

By default, the benchmark scripts fail if the final evaluation row does not
contain every task. This is intentional: an incomplete final row is useful for a
diagnostic smoke run, but it is not a paper-grade CAD benchmark row. Use
`--allow-incomplete-final-row` only when debugging partial artifacts.
