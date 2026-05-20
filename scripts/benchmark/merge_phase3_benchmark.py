"""Merge sequential CAD history with a Phase 3 final evaluation row.

Phase 3 checkpoints are evaluated after the task stream has already finished.
Their ``after_eval_dir/task_records.json`` files are therefore evaluation
sweeps, not fresh sequential-learning histories. For paper-style Avg/FM, keep
the sequential history from the warmup run and replace only the final row with
the post-Phase-3 final evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping


TaskRecord = Dict[str, Any]


def load_records(path: Path) -> list[TaskRecord]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a task_records.json list.")
    return payload


def write_records(path: Path, records: list[TaskRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def task_id(record: Mapping[str, Any]) -> int:
    if "task_id" not in record:
        raise ValueError("Every task record must contain task_id.")
    return int(record["task_id"])


def records_by_task(records: list[TaskRecord], source: Path) -> Dict[int, TaskRecord]:
    indexed: Dict[int, TaskRecord] = {}
    for record in records:
        indexed[task_id(record)] = record
    if not indexed:
        raise ValueError(f"{source} has no task records.")
    return indexed


def normalize_eval_row(row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_task_id, metrics in row.items():
        if not isinstance(metrics, Mapping):
            continue
        normalized[str(int(raw_task_id))] = dict(metrics)
    return normalized


def final_eval_row(final_eval_records: list[TaskRecord]) -> Dict[str, Dict[str, Any]]:
    """Build the final R[N-1][j] row from an after-Phase-3 eval sweep."""

    final_record = max(final_eval_records, key=task_id)
    row: Dict[str, Dict[str, Any]] = {}

    forgetting_eval = final_record.get("forgetting_eval")
    if isinstance(forgetting_eval, Mapping):
        row.update(normalize_eval_row(forgetting_eval))

    # Backfill from each record's direct current-task eval. This covers eval
    # sweeps that did not enable forgetting_eval while still evaluating every
    # task with the same final checkpoint.
    for record in final_eval_records:
        metrics = record.get("eval")
        if isinstance(metrics, Mapping):
            row.setdefault(str(task_id(record)), dict(metrics))

    return row


def validate_complete_row(row: Mapping[str, Any], final_task_id: int, source: Path) -> None:
    missing = [str(i) for i in range(final_task_id + 1) if str(i) not in row]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{source} cannot provide a complete Phase 3 final row; "
            f"missing task(s): {missing_text}."
        )


def merge_phase3_records(
    history_records: list[TaskRecord],
    final_eval_records: list[TaskRecord],
    history_source: Path,
    final_eval_source: Path,
    label: str,
) -> list[TaskRecord]:
    history_by_id = records_by_task(history_records, history_source)
    final_eval_by_id = records_by_task(final_eval_records, final_eval_source)

    final_task_id = max(history_by_id)
    if max(final_eval_by_id) != final_task_id:
        raise ValueError(
            f"Final task mismatch: history final task is {final_task_id}, "
            f"final eval task is {max(final_eval_by_id)}."
        )

    for expected_task_id in range(final_task_id + 1):
        if expected_task_id not in history_by_id:
            raise ValueError(f"{history_source} missing history task {expected_task_id}.")

    row = final_eval_row(final_eval_records)
    validate_complete_row(row, final_task_id, final_eval_source)

    merged = [copy.deepcopy(history_by_id[i]) for i in range(final_task_id + 1)]
    final_record = copy.deepcopy(history_by_id[final_task_id])
    final_record["eval"] = row[str(final_task_id)]
    final_record["forgetting_eval"] = row
    final_record["phase3_benchmark_merge"] = {
        "label": label,
        "history_task_records": str(history_source),
        "final_eval_task_records": str(final_eval_source),
        "meaning": (
            "Sequential history from the warmup run with final row replaced "
            "by post-Phase-3 evaluation metrics."
        ),
    }
    merged[-1] = final_record
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a paper-style Phase 3 CAD benchmark task_records artifact."
    )
    parser.add_argument(
        "--history-task-records",
        type=Path,
        required=True,
        help="Sequential warmup/training task_records.json.",
    )
    parser.add_argument(
        "--final-eval-task-records",
        type=Path,
        required=True,
        help="Post-Phase-3 evaluate_checkpoint task_records.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged task_records.json path.",
    )
    parser.add_argument("--label", default="phase3", help="Short label stored in metadata.")
    args = parser.parse_args()

    history = load_records(args.history_task_records)
    final_eval = load_records(args.final_eval_task_records)
    merged = merge_phase3_records(
        history_records=history,
        final_eval_records=final_eval,
        history_source=args.history_task_records,
        final_eval_source=args.final_eval_task_records,
        label=args.label,
    )
    write_records(args.output, merged)
    final_task_id = max(task_id(record) for record in merged)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tasks": final_task_id + 1,
                "history": str(args.history_task_records),
                "final_eval": str(args.final_eval_task_records),
                "label": args.label,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
