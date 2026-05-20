"""Collect Meta-NATH benchmark metric artifacts into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from compute_replaycad_metrics import markdown_row, summarize_artifact


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_row(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--row must use METHOD=path/to/task_records.json or METHOD=path/to/benchmark_metrics.json"
        )
    method, path_text = value.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("Row method name is empty.")
    return method, Path(path_text.strip())


def resolve_source(path: Path) -> Path:
    if path.is_dir():
        benchmark_metrics = path / "benchmark_metrics.json"
        task_records = path / "task_records.json"
        if benchmark_metrics.exists():
            return benchmark_metrics
        if task_records.exists():
            return task_records
    return path


def row_summary(
    method: str,
    source: Path,
    dataset: str,
    metrics: Iterable[str],
    require_complete_final_row: bool = True,
) -> Mapping[str, Any]:
    source = resolve_source(source)
    if not source.exists():
        raise FileNotFoundError(source)

    payload = load_json(source)
    if isinstance(payload, Mapping) and "table_row_percent" in payload:
        summary: Dict[str, Any] = dict(payload)
        summary["method"] = method
        summary.setdefault("dataset", dataset)
        summary.setdefault("source", str(source))
        if require_complete_final_row:
            validate_complete_final_row(summary, source)
        return summary

    return summarize_artifact(
        source,
        metrics=metrics,
        dataset=dataset,
        method=method,
        require_complete_final_row=require_complete_final_row,
    )


def validate_complete_final_row(summary: Mapping[str, Any], source: Path) -> None:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return
    for metric, metric_summary in metrics.items():
        if not isinstance(metric_summary, Mapping):
            continue
        missing = metric_summary.get("missing_final_tasks", [])
        if missing:
            missing_text = ", ".join(str(item) for item in missing)
            raise ValueError(
                f"{source} is not a complete CAD benchmark matrix for {metric!r}; "
                f"final row is missing task(s): {missing_text}."
            )


def build_markdown(dataset: str, rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        f"# {dataset} Continual Anomaly Detection Benchmark Rows",
        "",
        "| Method | Dataset | Tasks | Image-AUROC Avg | Image-AUROC FM | Pixel-AP Avg | Pixel-AP FM |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in rows:
        lines.append(markdown_row(summary))

    lines.extend(
        [
            "",
            "Values are percentages. Pixel-AP is computed from the repository `pixel_aupr` key.",
            "Only rows generated from local artifacts should be treated as reproduced results.",
            "External paper numbers, if added manually, must be labeled as paper-reported references.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-style benchmark Markdown table.")
    parser.add_argument("--dataset", default="MVTec")
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        type=parse_row,
        help="METHOD=path/to/task_records.json, run dir, or benchmark_metrics.json",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["image_auroc", "pixel_aupr"],
        help="Metric keys to extract when a row points at task_records.json.",
    )
    parser.add_argument(
        "--allow-incomplete-final-row",
        action="store_true",
        help="Allow diagnostic rows when the final matrix row does not contain all tasks.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summaries = [
        row_summary(
            method=method,
            source=source,
            dataset=args.dataset,
            metrics=args.metrics,
            require_complete_final_row=not args.allow_incomplete_final_row,
        )
        for method, source in args.row
    ]
    markdown = build_markdown(args.dataset, summaries)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
