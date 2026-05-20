"""Compute ReplayCAD-style CAD benchmark metrics from Meta-NATH artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MetricMatrix = Dict[str, Dict[str, float]]


TABLE_METRIC_NAMES = {
    "image_auroc": "Image-AUROC",
    "pixel_aupr": "Pixel-AP",
    "pixel_ap": "Pixel-AP",
    "image_ap": "Image-AP",
    "pixel_auroc": "Pixel-AUROC",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_from_mapping(metrics: Mapping[str, Any], metric: str) -> Optional[float]:
    if metric == "pixel_ap" and "pixel_ap" not in metrics:
        metric = "pixel_aupr"
    return _as_float(metrics.get(metric))


def extract_task_record_matrix(
    records: Iterable[Mapping[str, Any]],
    metric: str,
) -> Tuple[MetricMatrix, Dict[str, str]]:
    """Extract R[t][j] for one metric from task_records.json."""

    matrix: MetricMatrix = {}
    categories: Dict[str, str] = {}

    for record in records:
        task_id_raw = record.get("task_id")
        if task_id_raw is None:
            continue
        task_id = str(int(task_id_raw))
        categories[task_id] = str(record.get("category", task_id))

        row: Dict[str, float] = {}
        forgetting_eval = record.get("forgetting_eval")
        if isinstance(forgetting_eval, Mapping):
            for eval_task_id, eval_metrics in forgetting_eval.items():
                if not isinstance(eval_metrics, Mapping):
                    continue
                score = _metric_from_mapping(eval_metrics, metric)
                if score is not None:
                    row[str(int(eval_task_id))] = score

        current_eval = record.get("eval")
        if isinstance(current_eval, Mapping):
            score = _metric_from_mapping(current_eval, metric)
            if score is not None:
                row.setdefault(task_id, score)

        if row:
            matrix[task_id] = row

    return matrix, categories


def extract_forgetting_matrix(payload: Mapping[str, Any], metric: str) -> MetricMatrix:
    """Extract a numeric matrix from forgetting_matrix.json.

    The standard repository forgetting_matrix.json stores one metric only. If a
    caller requests a different metric, the file cannot provide it.
    """

    stored_metric = str(payload.get("metric", metric))
    if stored_metric != metric:
        raise ValueError(
            f"{payload.get('metric')} matrix cannot provide requested metric {metric!r}. "
            "Use task_records.json for multi-metric benchmark extraction."
        )

    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("forgetting_matrix.json must contain a 'matrix' object.")

    extracted: MetricMatrix = {}
    for after_task_id, row in matrix.items():
        if not isinstance(row, Mapping):
            continue
        extracted[str(int(after_task_id))] = {}
        for eval_task_id, score in row.items():
            value = _as_float(score)
            if value is not None:
                extracted[str(int(after_task_id))][str(int(eval_task_id))] = value
    return extracted


def extract_metric_matrix(payload: Any, metric: str) -> Tuple[MetricMatrix, Dict[str, str]]:
    if isinstance(payload, list):
        return extract_task_record_matrix(payload, metric)
    if isinstance(payload, Mapping) and "matrix" in payload:
        return extract_forgetting_matrix(payload, metric), {}
    raise ValueError("Input must be task_records.json or forgetting_matrix.json.")


def _sorted_int_keys(mapping: Mapping[str, Any]) -> list[int]:
    return sorted(int(key) for key in mapping.keys())


def compute_avg_and_fm(matrix: MetricMatrix) -> Dict[str, Any]:
    if not matrix:
        raise ValueError("Metric matrix is empty.")

    final_task = max(_sorted_int_keys(matrix))
    final_key = str(final_task)
    final_row = matrix.get(final_key, {})
    if not final_row:
        raise ValueError(f"Final row {final_key!r} is missing or empty.")

    eval_tasks = _sorted_int_keys(final_row)
    final_scores = {str(task_id): float(final_row[str(task_id)]) for task_id in eval_tasks}
    avg = sum(final_scores.values()) / len(final_scores)

    per_task_forgetting: Dict[str, float] = {}
    peak_scores: Dict[str, float] = {}

    for eval_task in eval_tasks:
        if eval_task == final_task:
            continue
        eval_key = str(eval_task)
        history = []
        for after_task in range(eval_task, final_task + 1):
            score = matrix.get(str(after_task), {}).get(eval_key)
            if score is not None:
                history.append(float(score))
        if not history:
            continue

        peak = max(history)
        final = final_scores[eval_key]
        peak_scores[eval_key] = peak
        per_task_forgetting[eval_key] = max(0.0, peak - final)

    fm = (
        sum(per_task_forgetting.values()) / len(per_task_forgetting)
        if per_task_forgetting
        else 0.0
    )

    expected_tasks = list(range(final_task + 1))
    missing_final_tasks = [
        str(task_id) for task_id in expected_tasks if str(task_id) not in final_row
    ]

    return {
        "avg": avg,
        "avg_percent": avg * 100.0,
        "fm": fm,
        "fm_percent": fm * 100.0,
        "final_task": final_task,
        "num_final_tasks": len(final_scores),
        "missing_final_tasks": missing_final_tasks,
        "final_scores": final_scores,
        "peak_scores": peak_scores,
        "per_task_forgetting": per_task_forgetting,
    }


def summarize_artifact(
    input_path: Path,
    metrics: Iterable[str],
    dataset: str,
    method: str,
    require_complete_final_row: bool = True,
) -> Dict[str, Any]:
    payload = load_json(input_path)
    metric_summaries: Dict[str, Any] = {}
    categories: Dict[str, str] = {}

    for metric in metrics:
        matrix, metric_categories = extract_metric_matrix(payload, metric)
        if metric_categories:
            categories.update(metric_categories)
        summary = compute_avg_and_fm(matrix)
        if require_complete_final_row and summary["missing_final_tasks"]:
            missing = ", ".join(summary["missing_final_tasks"])
            raise ValueError(
                f"{input_path} is not a complete CAD benchmark matrix for {metric!r}; "
                f"final row is missing task(s): {missing}. Enable evaluation.forgetting_matrix "
                "or pass --allow-incomplete-final-row for diagnostics only."
            )
        summary["matrix"] = matrix
        summary["table_metric_name"] = TABLE_METRIC_NAMES.get(metric, metric)
        metric_summaries[metric] = summary

    row = build_table_row(metric_summaries)
    num_tasks = max(
        (summary["final_task"] + 1 for summary in metric_summaries.values()),
        default=0,
    )

    return {
        "method": method,
        "dataset": dataset,
        "source": str(input_path),
        "num_tasks": num_tasks,
        "task_categories": categories,
        "metrics": metric_summaries,
        "table_row_percent": row,
    }


def build_table_row(metric_summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for metric, summary in metric_summaries.items():
        prefix = "pixel_ap" if metric in {"pixel_aupr", "pixel_ap"} else metric
        row[f"{prefix}_avg"] = float(summary["avg_percent"])
        row[f"{prefix}_fm"] = float(summary["fm_percent"])
    return row


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def markdown_row(summary: Mapping[str, Any]) -> str:
    row = summary["table_row_percent"]
    return (
        "| "
        + " | ".join(
            [
                str(summary["method"]),
                str(summary["dataset"]),
                str(summary["num_tasks"]),
                _format_percent(row.get("image_auroc_avg")),
                _format_percent(row.get("image_auroc_fm")),
                _format_percent(row.get("pixel_ap_avg")),
                _format_percent(row.get("pixel_ap_fm")),
            ]
        )
        + " |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ReplayCAD-style Image-AUROC/Pixel-AP Avg and FM."
    )
    parser.add_argument("input_json", type=Path, help="task_records.json or forgetting_matrix.json")
    parser.add_argument("--dataset", default="MVTec")
    parser.add_argument("--method", default="Meta-NATH")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["image_auroc", "pixel_aupr"],
        help="Metric keys to extract. Use pixel_aupr for Pixel-AP.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--print-markdown-row",
        action="store_true",
        help="Print a one-row Markdown table entry after the JSON summary.",
    )
    parser.add_argument(
        "--allow-incomplete-final-row",
        action="store_true",
        help="Allow diagnostic summaries when the final matrix row does not contain all tasks.",
    )
    args = parser.parse_args()

    summary = summarize_artifact(
        input_path=args.input_json,
        metrics=args.metrics,
        dataset=args.dataset,
        method=args.method,
        require_complete_final_row=not args.allow_incomplete_final_row,
    )

    if args.output is not None:
        write_json(args.output, summary)

    print(json.dumps(summary, indent=2))
    if args.print_markdown_row:
        print()
        print("| Method | Dataset | Tasks | Image-AUROC Avg | Image-AUROC FM | Pixel-AP Avg | Pixel-AP FM |")
        print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
        print(markdown_row(summary))


if __name__ == "__main__":
    main()
