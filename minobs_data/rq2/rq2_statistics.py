#!/usr/bin/env python3
"""
RQ2 post-hoc statistics for MinObs.

Computes point estimates and 95% percentile cluster-bootstrap confidence
intervals from run-level RQ2 results.

Bootstrap unit:
    (user_task_id, injection_task_id)

All repeated runs belonging to a sampled task-goal pair are resampled together.

Default paper protocol:
    B = 2000
    seed = 20260916
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_B = 2000
DEFAULT_SEED = 20260916

PAIR_COLS = ["user_task_id", "injection_task_id"]
REQUIRED_COLS = [
    "user_task_id",
    "injection_task_id",
    "method",
    "status",
    "utility",
    "attack_success",
    "joint_safe_utility",
    "injection_reached",
    "injection_exposed",
    "trajectory_match",
]


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", ""}

    def conv(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, float):
            return bool(int(value))
        s = str(value).strip().lower()
        if s in true_values:
            return True
        if s in false_values:
            return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")

    return series.map(conv).astype(bool)


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in runs CSV: " + ", ".join(missing)
        )

    df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    if df.empty:
        raise ValueError("No status=ok rows found.")

    for col in [
        "utility",
        "attack_success",
        "joint_safe_utility",
        "injection_reached",
        "injection_exposed",
        "trajectory_match",
    ]:
        df[col] = _to_bool_series(df[col])

    return df


def metric_point_estimates(df: pd.DataFrame) -> Dict[str, float | int | None]:
    reached = df["injection_reached"]
    exposed = df["injection_exposed"]

    return {
        "n_runs": int(len(df)),
        "attack_successes": int(df["attack_success"].sum()),
        "utility_successes": int(df["utility"].sum()),
        "joint_safe_successes": int(df["joint_safe_utility"].sum()),
        "n_reached": int(reached.sum()),
        "n_exposed": int(exposed.sum()),
        "trajectory_matches": int(df["trajectory_match"].sum()),
        "asr": float(df["attack_success"].mean()),
        "utility_under_attack": float(df["utility"].mean()),
        "joint_safe_utility": float(df["joint_safe_utility"].mean()),
        "ier_given_reached": (
            float(df.loc[reached, "injection_exposed"].mean())
            if reached.any()
            else None
        ),
        "asr_given_reached": (
            float(df.loc[reached, "attack_success"].mean())
            if reached.any()
            else None
        ),
        "asr_given_exposed": (
            float(df.loc[exposed, "attack_success"].mean())
            if exposed.any()
            else None
        ),
        "trajectory_match": float(df["trajectory_match"].mean()),
    }


def _pair_count_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    One row per task-goal pair:
    [n, attack, utility, joint, reached, exposed, traj]
    """
    pair = (
        df.assign(
            n=1,
            attack=df["attack_success"].astype(int),
            utility_i=df["utility"].astype(int),
            joint=df["joint_safe_utility"].astype(int),
            reached=df["injection_reached"].astype(int),
            exposed=df["injection_exposed"].astype(int),
            traj=df["trajectory_match"].astype(int),
        )
        .groupby(PAIR_COLS, sort=True)[
            ["n", "attack", "utility_i", "joint", "reached", "exposed", "traj"]
        ]
        .sum()
    )
    return pair.to_numpy(dtype=np.int64)


def _bootstrap_one_method(
    df: pd.DataFrame,
    *,
    B: int,
    seed: int,
) -> Dict[str, List[float] | None]:
    rng = np.random.default_rng(seed)
    mat = _pair_count_matrix(df)

    n_pairs = mat.shape[0]
    if n_pairs == 0:
        raise ValueError("No task-goal pairs available for bootstrap.")

    # shape: B x n_pairs
    draw = rng.integers(0, n_pairs, size=(B, n_pairs))
    sampled = mat[draw].sum(axis=1)

    n = sampled[:, 0].astype(float)
    attack = sampled[:, 1].astype(float)
    utility = sampled[:, 2].astype(float)
    joint = sampled[:, 3].astype(float)
    reached = sampled[:, 4].astype(float)
    exposed = sampled[:, 5].astype(float)
    traj = sampled[:, 6].astype(float)

    values = {
        "asr": attack / n,
        "utility_under_attack": utility / n,
        "joint_safe_utility": joint / n,
        "ier_given_reached": np.divide(
            exposed,
            reached,
            out=np.full(B, np.nan, dtype=float),
            where=reached > 0,
        ),
        "trajectory_match": traj / n,
    }

    result: Dict[str, List[float] | None] = {}
    for metric, arr in values.items():
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            result[metric] = None
        else:
            result[metric] = [
                float(np.quantile(arr, 0.025)),
                float(np.quantile(arr, 0.975)),
            ]

    return result


def summarize_all(
    df: pd.DataFrame,
    *,
    B: int,
    seed: int,
) -> Dict[str, Any]:
    methods = list(dict.fromkeys(df["method"].astype(str).tolist()))

    result: Dict[str, Any] = {
        "bootstrap": {
            "unit": "task_x_injection_goal_pair",
            "pair_columns": PAIR_COLS,
            "B": int(B),
            "seed": int(seed),
            "ci": "percentile_95",
            "note": (
                "Repeated runs within each sampled task-goal pair are "
                "resampled together."
            ),
        },
        "methods": {},
    }

    for i, method in enumerate(methods):
        sub = df[df["method"].astype(str).eq(method)].copy()
        point = metric_point_estimates(sub)
        ci = _bootstrap_one_method(
            sub,
            B=B,
            seed=seed + i,
        )
        result["methods"][method] = {
            **point,
            "ci95": ci,
        }

    return result


def write_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    rows = []

    for method, m in summary["methods"].items():
        ier_ci = m["ci95"]["ier_given_reached"]

        rows.append(
            {
                "method": method,
                "n_runs": m["n_runs"],
                "attack_successes": m["attack_successes"],
                "asr": m["asr"],
                "asr_ci95_low": m["ci95"]["asr"][0],
                "asr_ci95_high": m["ci95"]["asr"][1],
                "utility_under_attack": m["utility_under_attack"],
                "ua_ci95_low": m["ci95"]["utility_under_attack"][0],
                "ua_ci95_high": m["ci95"]["utility_under_attack"][1],
                "joint_safe_utility": m["joint_safe_utility"],
                "joint_ci95_low": m["ci95"]["joint_safe_utility"][0],
                "joint_ci95_high": m["ci95"]["joint_safe_utility"][1],
                "ier_given_reached": m["ier_given_reached"],
                "ier_ci95_low": ier_ci[0] if ier_ci else None,
                "ier_ci95_high": ier_ci[1] if ier_ci else None,
                "trajectory_match": m["trajectory_match"],
                "trajectory_ci95_low": m["ci95"]["trajectory_match"][0],
                "trajectory_ci95_high": m["ci95"]["trajectory_match"][1],
                "n_reached": m["n_reached"],
                "n_exposed": m["n_exposed"],
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/rq2_complete/rq2_runs.csv"),
        help="Run-level RQ2 CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/rq2_complete/rq2_statistics.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/rq2_complete/rq2_statistics.csv"),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_B,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    B = max(1, int(args.bootstrap))
    seed = int(args.seed)

    df = load_runs(args.runs)
    summary = summarize_all(df, B=B, seed=seed)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(summary, args.output_csv)

    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")
    print(f"Bootstrap unit: {PAIR_COLS}; B={B}; seed={seed}")

    for method, m in summary["methods"].items():
        lo, hi = m["ci95"]["asr"]
        print(
            f"{method:12s} "
            f"ASR={m['asr']:.4f} "
            f"95% CI=[{lo:.4f}, {hi:.4f}]"
        )


if __name__ == "__main__":
    main()
