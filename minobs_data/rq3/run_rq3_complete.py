#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.task_suite.load_suites import get_suite

from minobs_formal import (
    MinObs,
    SufficiencyMinimizer,
    SufficiencyVerifier,
    approximate_token_count,
    extract_observation,
    infer_contract,
)

from run_minobs_workspace_formal_v3 import (
    AgentDojoMinObsAdapter,
    TokenUsageTracker,
    candidate_contract,
    field_priority,
    load_jsonl,
    progress_printer,
    record_to_context,
)

from run_rq1_complete import (
    FieldPlanStep,
    build_agentdojo_pipeline,
    empty_payload_for,
    format_runtime_observation,
    normalized_arguments_equal,
    parse_runtime_observation,
    project_top_level_fields,
    _live_tool_call,
    _message_text,
)

from run_rq2_complete import (
    RQ2ExecutorAudit,
    build_attack,
    load_frozen_rq1_plans,
    load_json,
    matched_injection_vectors,
    rq1_stable_task_ids,
)


ABLATION_CONDITIONS = (
    "minobs",
    "w_o_sufficiency",
    "single_pass",
    "fail_open",
    "relevant",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def mean(values: Sequence[float]) -> Optional[float]:
    data = [float(item) for item in values]
    if not data:
        return None
    return sum(data) / len(data)


def mean_bool(values: Sequence[Any]) -> Optional[float]:
    data = [bool(item) for item in values]
    if not data:
        return None
    return sum(int(item) for item in data) / len(data)


@dataclass
class RQ3PlanStep(FieldPlanStep):
    single_pass_fields: List[str] = field(default_factory=list)
    w_o_sufficiency_fields: List[str] = field(default_factory=list)

    def fields_for(self, condition: str) -> List[str]:
        if condition in {"minobs", "fail_open"}:
            return list(self.minobs_fields)
        if condition == "relevant":
            return list(self.relevant_fields)
        if condition == "single_pass":
            return list(self.single_pass_fields)
        if condition == "w_o_sufficiency":
            return list(self.w_o_sufficiency_fields)
        raise ValueError(f"unknown condition: {condition}")


def plans_from_frozen(
    frozen: Mapping[str, Sequence[FieldPlanStep]],
) -> Dict[str, List[RQ3PlanStep]]:
    out: Dict[str, List[RQ3PlanStep]] = {}
    for task_id, steps in frozen.items():
        converted: List[RQ3PlanStep] = []
        for step in steps:
            converted.append(
                RQ3PlanStep(
                    task_id=step.task_id,
                    step_id=step.step_id,
                    expected_tool=step.expected_tool,
                    expected_arguments=dict(step.expected_arguments),
                    relevant_fields=list(step.relevant_fields),
                    minobs_fields=list(step.minobs_fields),
                    single_pass_fields=list(step.minobs_fields),
                    w_o_sufficiency_fields=list(step.minobs_fields),
                )
            )
        out[task_id] = sorted(converted, key=lambda item: item.step_id)
    return out


def chain_bucket(n_steps: int) -> str:
    if n_steps <= 1:
        return "Short"
    if n_steps == 2:
        return "Mid"
    return "Long"


def _ordered_fields(names: Sequence[str]) -> List[str]:
    return sorted(names, key=lambda name: (field_priority(name), name))


def minimize_without_sufficiency(raw_observation: Any) -> List[str]:
    """Drop fields in priority order until one top-level field remains."""
    contract = infer_contract(raw_observation)
    current = list(contract.field_names())
    for name in _ordered_fields(current):
        if len(current) <= 1:
            break
        current = [item for item in current if item != name]
    return current


def live_minobs_fields(
    *,
    context: Any,
    adapter: AgentDojoMinObsAdapter,
    repeats: int,
    fixed_point: bool,
) -> Tuple[List[str], Mapping[str, Any], TokenUsageTracker]:
    usage = TokenUsageTracker()
    adapter.usage_tracker = usage
    verifier = SufficiencyVerifier(
        replay_decision_fn=adapter.replay,
        evaluate_decision_fn=adapter.evaluate,
        repeats=max(1, repeats),
    )
    minimizer = SufficiencyMinimizer(
        verifier=verifier,
        field_priority_fn=field_priority,
        repeat_until_fixed_point=fixed_point,
        progress_fn=progress_printer,
    )
    minobs = MinObs(
        candidate_contract_fn=candidate_contract,
        minimizer=minimizer,
    )
    result, metrics = minobs.run(context=context)
    fields = list(getattr(result, "minimal_fields", None) or [])
    verification = getattr(result, "baseline_verification", None)
    payload = {
        "fields": fields,
        "sufficient": bool(getattr(verification, "sufficient", True)),
        "fer": float(getattr(metrics, "field_exposure_ratio", 0.0)),
        "ter": float(getattr(metrics, "token_exposure_ratio_approx", 0.0)),
        "verification": verification,
    }
    return fields, payload, usage


def observation_rates(
    raw_observation: Any,
    kept_fields: Sequence[str],
) -> Tuple[float, float, Any]:
    contract = infer_contract(raw_observation)
    raw_names = list(contract.field_names())
    projected = extract_observation(raw_observation, contract.subset(list(kept_fields)))
    raw_tokens = approximate_token_count(raw_observation)
    kept_tokens = approximate_token_count(projected)
    fer = 0.0 if not raw_names else len(kept_fields) / len(raw_names)
    ter = 0.0 if raw_tokens <= 0 else kept_tokens / raw_tokens
    return fer, ter, projected


class RQ3ObservationExecutor(ToolsExecutor):
    """
    Project successful structured tool results with a condition-specific mask.

    fail_closed (minobs / relevant / single_pass / w_o_sufficiency):
        plan miss -> empty payload
    fail_open:
        plan miss -> raw payload
    """

    def __init__(
        self,
        *,
        condition: str,
        plan: Sequence[RQ3PlanStep],
        injections: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__()
        if condition not in ABLATION_CONDITIONS:
            raise ValueError(f"unsupported condition: {condition}")
        self.condition = condition
        self.plan = list(plan)
        self.injections = dict(injections or {})
        self.fail_open = condition == "fail_open"
        self.step_index = 0
        self.audit = RQ2ExecutorAudit(method=condition)

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any,
        messages: Sequence[Any] = [],
        extra_args: dict = {},
    ):
        before_len = len(messages)
        query, runtime, env, executed_messages, extra_args = super().query(
            query, runtime, env, messages, extra_args
        )
        if len(executed_messages) <= before_len:
            return query, runtime, env, executed_messages, extra_args

        rewritten = list(executed_messages[:before_len])
        for message in executed_messages[before_len:]:
            if not isinstance(message, Mapping) or message.get("role") != "tool":
                rewritten.append(message)
                continue

            current_step = self.step_index
            self.step_index += 1
            self.audit.tool_results_seen += 1

            tool_name, live_args = _live_tool_call(message)
            error = message.get("error")
            raw_text = _message_text(message)
            raw_observation = parse_runtime_observation(raw_text)
            raw_matches = matched_injection_vectors(raw_observation, self.injections)
            had_exposure_before = self.audit.injection_exposed

            if raw_matches:
                self.audit.injection_reached = True
                self.audit.reached_vectors.update(raw_matches)
                if self.audit.first_reached_step is None:
                    self.audit.first_reached_step = current_step

            if error:
                self.audit.tool_errors_preserved += 1
                rewritten.append(message)
                continue

            plan_step = self.plan[current_step] if current_step < len(self.plan) else None
            plan_ok = False
            miss_reason = None
            if plan_step is not None:
                plan_ok = tool_name == plan_step.expected_tool and normalized_arguments_equal(
                    live_args, plan_step.expected_arguments
                )

            if plan_ok:
                fields = plan_step.fields_for(self.condition)
                try:
                    exposed_observation = project_top_level_fields(raw_observation, fields)
                except Exception:
                    plan_ok = False
                    miss_reason = "projection_failed"
                    exposed_observation = (
                        raw_observation if self.fail_open else empty_payload_for(raw_observation)
                    )
            else:
                miss_reason = "tool_or_arguments_mismatch" if plan_step else "plan_exhausted"
                exposed_observation = (
                    raw_observation if self.fail_open else empty_payload_for(raw_observation)
                )

            if not plan_ok:
                self.audit.plan_misses += 1
                if self.audit.first_plan_miss_step is None:
                    self.audit.first_plan_miss_step = current_step
                if had_exposure_before:
                    self.audit.post_exposure_plan_miss = True
                else:
                    self.audit.pre_exposure_plan_miss = True

            exposed_matches = matched_injection_vectors(exposed_observation, self.injections)
            if exposed_matches:
                self.audit.injection_exposed = True
                self.audit.exposed_vectors.update(exposed_matches)

            new_message = dict(message)
            new_message["content"] = format_runtime_observation(exposed_observation)
            new_message["minobs_rq3"] = {
                "condition": self.condition,
                "plan_ok": plan_ok,
                "plan_miss_reason": miss_reason,
                "fail_open": self.fail_open,
            }
            rewritten.append(new_message)

        return query, runtime, env, rewritten, extra_args


def build_rq3_pipeline(
    *,
    condition: str,
    plan: Sequence[RQ3PlanStep],
    injections: Mapping[str, str],
    agentdojo_model: str,
    model_id: str,
    suite_name: str,
):
    executor = RQ3ObservationExecutor(
        condition=condition,
        plan=plan,
        injections=injections,
    )
    pipeline = build_agentdojo_pipeline(
        agentdojo_model=agentdojo_model,
        model_id=model_id,
        suite_name=suite_name,
        replacement_executor=executor,
    )
    return pipeline, executor


def load_ok_steps(path: Path) -> pd.DataFrame:
    steps = pd.read_csv(path)
    if "status" in steps.columns:
        steps = steps[steps["status"].eq("ok")].copy()
    return steps.reset_index(drop=True)


def index_traces(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or row.get("user_task_id") or "")
        if not task_id:
            continue
        if "step_id" in row and ("raw_observation" in row or "history" in row or "next_action" in row):
            indexed[(task_id, int(row["step_id"]))] = row
            continue
        nested = row.get("steps") or row.get("trace") or []
        if isinstance(nested, list):
            for step_id, step in enumerate(nested):
                if isinstance(step, Mapping):
                    indexed[(task_id, int(step.get("step_id", step_id)))] = {
                        **step,
                        "task_id": task_id,
                        "step_id": int(step.get("step_id", step_id)),
                    }
        elif "observation" in row or "raw_observation" in row:
            indexed[(task_id, int(row.get("step_id", 0)))] = row
    return indexed


def context_from_step(
    *,
    task_id: str,
    step_id: int,
    traces: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Any:
    record = traces.get((task_id, step_id))
    if record is None:
        raise KeyError(f"missing clean trace for {task_id} step {step_id}")
    return record_to_context(record)


def run_rq3_a(args: argparse.Namespace, client: OpenAI) -> None:
    out = args.output_dir
    frozen = load_frozen_rq1_plans(args.rq1_plans)
    plans = plans_from_frozen(frozen)
    stable = rq1_stable_task_ids(load_json(args.rq1_final))
    if args.task_ids:
        requested = {item.strip() for item in args.task_ids.split(",") if item.strip()}
        stable = [task_id for task_id in stable if task_id in requested]

    steps = load_ok_steps(args.rq1_steps)
    traces = index_traces(load_jsonl(args.rq1_traces)) if args.rq1_traces.exists() else {}

    adapter = AgentDojoMinObsAdapter(
        client=client,
        model=args.model_id,
        judge_model=args.judge_model or args.model_id,
        api_retries=args.api_retries,
        retry_base_delay=args.retry_base_delay,
        usage_tracker=TokenUsageTracker(),
    )

    learned_path = out / "rq3_learned_masks.json"
    learned: Dict[str, Any] = (
        load_json(learned_path) if learned_path.exists() and not args.force else {}
    )

    for _, row in steps.iterrows():
        task_id = str(row["task_id"])
        step_id = int(row["step_id"])
        key = f"{task_id}:{step_id}"
        if key in learned and not args.force:
            continue
        if (task_id, step_id) not in traces:
            print(f"[RQ3-A] skip mask learning, missing trace {key}")
            continue
        context = context_from_step(task_id=task_id, step_id=step_id, traces=traces)
        raw_observation = getattr(context, "raw_observation", None)
        if raw_observation is None:
            print(f"[RQ3-A] skip mask learning, no raw_observation {key}")
            continue

        print(f"[RQ3-A] learn single_pass / w_o_sufficiency for {key}")
        single_fields, single_info, _ = live_minobs_fields(
            context=context,
            adapter=adapter,
            repeats=args.repeats,
            fixed_point=False,
        )
        weak_fields = minimize_without_sufficiency(raw_observation)
        learned[key] = {
            "task_id": task_id,
            "step_id": step_id,
            "single_pass_fields": single_fields,
            "w_o_sufficiency_fields": weak_fields,
            "single_pass_info": {
                "fer": single_info.get("fer"),
                "ter": single_info.get("ter"),
                "sufficient": single_info.get("sufficient"),
            },
        }
        write_json(learned_path, learned)

    for task_id, plan in plans.items():
        for step in plan:
            item = learned.get(f"{task_id}:{step.step_id}") or {}
            if item.get("single_pass_fields") is not None:
                step.single_pass_fields = list(item["single_pass_fields"])
            if item.get("w_o_sufficiency_fields") is not None:
                step.w_o_sufficiency_fields = list(item["w_o_sufficiency_fields"])

    step_rows: List[Dict[str, Any]] = []
    for _, row in steps.iterrows():
        task_id = str(row["task_id"])
        step_id = int(row["step_id"])
        plan_steps = [item for item in plans.get(task_id, []) if item.step_id == step_id]
        if not plan_steps:
            continue
        plan_step = plan_steps[0]
        raw_observation = None
        if (task_id, step_id) in traces:
            context = context_from_step(task_id=task_id, step_id=step_id, traces=traces)
            raw_observation = getattr(context, "raw_observation", None)

        for condition in ABLATION_CONDITIONS:
            fields = plan_step.fields_for(condition)
            if raw_observation is None:
                step_rows.append(
                    {
                        "task_id": task_id,
                        "step_id": step_id,
                        "condition": condition,
                        "fer": None,
                        "ter": None,
                        "sufficiency_rate": None,
                        "kept_fields": fields,
                        "status": "missing_trace",
                    }
                )
                continue
            fer, ter, _ = observation_rates(raw_observation, fields)
            step_rows.append(
                {
                    "task_id": task_id,
                    "step_id": step_id,
                    "condition": condition,
                    "fer": fer,
                    "ter": ter,
                    "sufficiency_rate": None,
                    "kept_fields": fields,
                    "status": "ok",
                }
            )
    write_jsonl(out / "rq3_ablation_steps.jsonl", step_rows)

    if args.skip_pipeline:
        print("[RQ3-A] --skip-pipeline set; wrote step masks only")
        _write_ablation_aggregate(out, step_rows, [], [])
        return

    suite = get_suite(args.benchmark_version, args.suite)
    utility_path = out / "rq3_ablation_utility.csv"
    attack_path = out / "rq3_ablation_attack.jsonl"

    existing_utility: List[Dict[str, Any]] = []
    if utility_path.exists() and not args.force:
        existing_utility = list(pd.read_csv(utility_path).to_dict("records"))
    utility_done = {
        (str(row["task_id"]), str(row["condition"]), int(row["repeat"]))
        for row in existing_utility
    }
    utility_rows = list(existing_utility)

    for condition in ABLATION_CONDITIONS:
        for task_id in stable:
            user_task = suite.user_tasks[task_id]
            plan = plans[task_id]
            for repeat in range(1, args.utility_repeats + 1):
                if (task_id, condition, repeat) in utility_done:
                    continue
                print(f"[RQ3-A utility] {condition} {task_id} r{repeat}")
                pipeline, _ = build_rq3_pipeline(
                    condition=condition,
                    plan=plan,
                    injections={},
                    agentdojo_model=args.agentdojo_model,
                    model_id=args.model_id,
                    suite_name=args.suite,
                )
                utility, _injection = suite.run_task_with_pipeline(
                    pipeline,
                    user_task,
                    injection_task=None,
                    attack=None,
                )
                utility_rows.append(
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "repeat": repeat,
                        "utility": bool(utility),
                    }
                )
                pd.DataFrame(utility_rows).to_csv(utility_path, index=False)

    existing_attack = load_existing_jsonl(attack_path) if not args.force else []
    if args.force and attack_path.exists():
        attack_path.unlink()
        existing_attack = []
    attack_done = {
        (
            str(row["user_task_id"]),
            str(row["injection_task_id"]),
            str(row["condition"]),
            int(row["repeat"]),
        )
        for row in existing_attack
    }

    injection_ids = list(suite.injection_tasks.keys())
    if args.injection_task_ids:
        requested = {item.strip() for item in args.injection_task_ids.split(",") if item.strip()}
        injection_ids = [item for item in injection_ids if item in requested]

    attack_target_pipeline = build_agentdojo_pipeline(
        agentdojo_model=args.agentdojo_model,
        model_id=args.model_id,
        suite_name=args.suite,
        replacement_executor=None,
    )
    attack, attack_impl = build_attack(
        attack_name=args.attack,
        suite=suite,
        target_pipeline=attack_target_pipeline,
        attack_model_name=args.attack_model_name,
    )

    for condition in ABLATION_CONDITIONS:
        for task_id in stable:
            user_task = suite.user_tasks[task_id]
            plan = plans[task_id]
            for injection_task_id in injection_ids:
                injection_task = suite.injection_tasks[injection_task_id]
                injections = dict(getattr(attack, "injections", {}) or {})
                for repeat in range(1, args.repeats + 1):
                    key = (task_id, injection_task_id, condition, repeat)
                    if key in attack_done:
                        continue
                    print(f"[RQ3-A attack] {condition} {task_id} {injection_task_id} r{repeat}")
                    pipeline, executor = build_rq3_pipeline(
                        condition=condition,
                        plan=plan,
                        injections=injections,
                        agentdojo_model=args.agentdojo_model,
                        model_id=args.model_id,
                        suite_name=args.suite,
                    )
                    utility, injection_success = suite.run_task_with_pipeline(
                        pipeline,
                        user_task,
                        injection_task=injection_task,
                        attack=attack,
                    )
                    audit = executor.audit
                    row = {
                        "user_task_id": task_id,
                        "injection_task_id": injection_task_id,
                        "condition": condition,
                        "repeat": repeat,
                        "utility": bool(utility),
                        "attack_success": bool(injection_success),
                        "joint_safe_utility": bool(utility) and not bool(injection_success),
                        "injection_reached": bool(audit.injection_reached),
                        "injection_exposed": bool(audit.injection_exposed),
                        "plan_misses": int(audit.plan_misses),
                        "attack_implementation": attack_impl,
                    }
                    append_jsonl(attack_path, row)

    attack_rows = load_existing_jsonl(attack_path)
    _write_ablation_aggregate(out, step_rows, utility_rows, attack_rows)


def _write_ablation_aggregate(
    out: Path,
    step_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
) -> None:
    step_df = pd.DataFrame(step_rows)
    utility_df = pd.DataFrame(utility_rows)
    attack_df = pd.DataFrame(attack_rows)
    aggregate: Dict[str, Any] = {"rq": "RQ3-A", "data_status": "LIVE", "conditions": {}}
    for condition in ABLATION_CONDITIONS:
        s = step_df[step_df["condition"].eq(condition)] if not step_df.empty else step_df
        u = utility_df[utility_df["condition"].eq(condition)] if not utility_df.empty else utility_df
        a = attack_df[attack_df["condition"].eq(condition)] if not attack_df.empty else attack_df
        reached = (
            a["injection_reached"].astype(bool)
            if not a.empty and "injection_reached" in a
            else pd.Series(dtype=bool)
        )
        aggregate["conditions"][condition] = {
            "mean_fer": None if s.empty or s["fer"].dropna().empty else float(s["fer"].mean()),
            "mean_ter": None if s.empty or s["ter"].dropna().empty else float(s["ter"].mean()),
            "benign_utility": None if u.empty else float(u["utility"].astype(bool).mean()),
            "asr": None if a.empty else float(a["attack_success"].astype(bool).mean()),
            "utility_under_attack": None if a.empty else float(a["utility"].astype(bool).mean()),
            "joint_safe_utility": None if a.empty else float(a["joint_safe_utility"].astype(bool).mean()),
            "ier_given_reached": (
                None
                if a.empty or not reached.any()
                else float(a.loc[reached, "injection_exposed"].astype(bool).mean())
            ),
            "n_attack_runs": int(len(a)),
            "n_utility_runs": int(len(u)),
            "n_steps": int(len(s)),
        }
    write_json(out / "rq3_ablation_aggregate.json", aggregate)


def run_rq3_b(args: argparse.Namespace) -> None:
    out = args.output_dir
    frozen = load_frozen_rq1_plans(args.rq1_plans)
    stable = rq1_stable_task_ids(load_json(args.rq1_final))
    if args.task_ids:
        requested = {item.strip() for item in args.task_ids.split(",") if item.strip()}
        stable = [task_id for task_id in stable if task_id in requested]

    attack_source = out / "rq3_ablation_attack.jsonl"
    if args.reuse_rq2_runs and args.reuse_rq2_runs.exists():
        raw_rows = load_jsonl(args.reuse_rq2_runs)
        minobs_rows = [
            row
            for row in raw_rows
            if str(row.get("method") or row.get("condition")) == "minobs"
            and str(row.get("user_task_id")) in set(stable)
        ]
        source_name = str(args.reuse_rq2_runs)
    elif attack_source.exists():
        minobs_rows = [
            row
            for row in load_jsonl(attack_source)
            if str(row.get("condition")) == "minobs"
            and str(row.get("user_task_id")) in set(stable)
        ]
        source_name = str(attack_source)
    else:
        raise SystemExit(
            "RQ3-B needs live MinObs attack rows. Run RQ3-A first or pass "
            "--reuse-rq2-runs pointing at empirical rq2_runs.jsonl."
        )

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in minobs_rows:
        by_task[str(row["user_task_id"])].append(row)

    step_lookup = defaultdict(list)
    step_file = out / "rq3_ablation_steps.jsonl"
    if step_file.exists():
        for row in load_jsonl(step_file):
            if row.get("condition") == "minobs":
                step_lookup[str(row["task_id"])].append(row)

    chain_rows: List[Dict[str, Any]] = []
    chain_attack_rows: List[Dict[str, Any]] = []
    for task_id in stable:
        n_steps = len(frozen.get(task_id, []))
        bucket = chain_bucket(n_steps)
        task_runs = by_task.get(task_id, [])
        for row in task_runs:
            item = dict(row)
            item["bucket"] = bucket
            chain_attack_rows.append(item)
        fer_values = [float(item["fer"]) for item in step_lookup.get(task_id, []) if item.get("fer") is not None]
        ter_values = [float(item["ter"]) for item in step_lookup.get(task_id, []) if item.get("ter") is not None]
        chain_rows.append(
            {
                "task_id": task_id,
                "n_steps": n_steps,
                "bucket": bucket,
                "mean_minobs_fer": mean(fer_values),
                "mean_minobs_ter": mean(ter_values),
                "asr": mean_bool([row.get("attack_success") for row in task_runs]),
                "exposed_rate": mean_bool([row.get("injection_exposed") for row in task_runs]),
                "reached_rate": mean_bool([row.get("injection_reached") for row in task_runs]),
                "utility_under_attack": mean_bool([row.get("utility") for row in task_runs]),
                "joint_safe_utility": mean_bool([row.get("joint_safe_utility") for row in task_runs]),
                "n_runs": len(task_runs),
            }
        )

    pd.DataFrame(chain_rows).to_csv(out / "rq3_chain_length.csv", index=False)
    write_jsonl(out / "rq3_chain_attack.jsonl", chain_attack_rows)

    aggregate = {"rq": "RQ3-B", "data_status": "LIVE", "source": source_name, "by_bucket": {}}
    attack_df = pd.DataFrame(chain_attack_rows)
    summary_df = pd.DataFrame(chain_rows)
    for bucket in ("Short", "Mid", "Long"):
        t = summary_df[summary_df["bucket"].eq(bucket)] if not summary_df.empty else summary_df
        a = attack_df[attack_df["bucket"].eq(bucket)] if not attack_df.empty else attack_df
        aggregate["by_bucket"][bucket] = {
            "n_tasks": int(len(t)),
            "task_ids": [] if t.empty else t["task_id"].tolist(),
            "runs": int(len(a)),
            "mean_fer": None if t.empty or t["mean_minobs_fer"].dropna().empty else float(t["mean_minobs_fer"].mean()),
            "mean_ter": None if t.empty or t["mean_minobs_ter"].dropna().empty else float(t["mean_minobs_ter"].mean()),
            "mean_asr": None if a.empty else float(a["attack_success"].astype(bool).mean()),
            "mean_exposed_rate": None if a.empty else float(a["injection_exposed"].astype(bool).mean()),
            "utility_under_attack": None if a.empty else float(a["utility"].astype(bool).mean()),
            "joint_safe_utility": None if a.empty else float(a["joint_safe_utility"].astype(bool).mean()),
        }
    write_json(out / "rq3_chain_length_aggregate.json", aggregate)


def run_rq3_c(args: argparse.Namespace, client: OpenAI) -> None:
    out = args.output_dir
    steps = load_ok_steps(args.rq1_steps)
    traces = index_traces(load_jsonl(args.rq1_traces)) if args.rq1_traces.exists() else {}
    if not traces:
        raise SystemExit("RQ3-C needs --rq1-traces with the clean Phase-A records.")

    adapter = AgentDojoMinObsAdapter(
        client=client,
        model=args.model_id,
        judge_model=args.judge_model or args.model_id,
        api_retries=args.api_retries,
        retry_base_delay=args.retry_base_delay,
        usage_tracker=TokenUsageTracker(),
    )

    out_steps = out / "rq3_replay_steps.jsonl"
    existing = load_existing_jsonl(out_steps) if out_steps.exists() and not args.force else []
    if args.force and out_steps.exists():
        out_steps.unlink()
        existing = []
    done = {(str(row["task_id"]), int(row["step_id"]), int(row["r"])) for row in existing}

    replay_grid = [int(item) for item in args.replay_grid.split(",") if item.strip()]
    for r in replay_grid:
        for _, row in steps.iterrows():
            task_id = str(row["task_id"])
            step_id = int(row["step_id"])
            if (task_id, step_id, r) in done:
                continue
            if (task_id, step_id) not in traces:
                print(f"[RQ3-C] missing trace {task_id} step {step_id}")
                continue
            print(f"[RQ3-C] r={r} {task_id} step {step_id}")
            context = context_from_step(task_id=task_id, step_id=step_id, traces=traces)
            fields, info, usage = live_minobs_fields(
                context=context,
                adapter=adapter,
                repeats=r,
                fixed_point=True,
            )
            raw_observation = getattr(context, "raw_observation", None)
            raw_field_count = (
                len(infer_contract(raw_observation).field_names())
                if raw_observation is not None
                else int(row.get("raw_field_count") or 0)
            )
            snapshot = usage.snapshot() if hasattr(usage, "snapshot") else {}
            approx_tokens = int(snapshot.get("total_tokens") or snapshot.get("approx_tokens") or 0)
            sufficiency = info.get("verification")
            if hasattr(sufficiency, "pass_count") and hasattr(sufficiency, "repeat_count"):
                denom = max(1, int(sufficiency.repeat_count))
                sufficiency_rate = float(sufficiency.pass_count) / float(denom)
            elif isinstance(sufficiency, Mapping) and sufficiency.get("repeat_count"):
                sufficiency_rate = float(sufficiency.get("pass_count") or 0) / float(sufficiency["repeat_count"])
            else:
                sufficiency_rate = 1.0 if info.get("sufficient") else 0.0
            record = {
                "task_id": task_id,
                "step_id": step_id,
                "r": r,
                "minobs_fer": float(info.get("fer") or 0.0),
                "minobs_ter": float(info.get("ter") or 0.0),
                "sufficiency_rate": float(sufficiency_rate),
                "strict_sufficient": bool(sufficiency_rate >= 1.0),
                "replay_calls": int(max(1, raw_field_count) * r),
                "approx_tokens": int(approx_tokens),
                "raw_field_count": int(raw_field_count),
                "kept_fields": fields,
            }
            append_jsonl(out_steps, record)

    rows = load_existing_jsonl(out_steps)
    replay_df = pd.DataFrame(rows)
    cost_rows = []
    aggregate: Dict[str, Any] = {"rq": "RQ3-C", "data_status": "LIVE", "by_r": {}}
    if replay_df.empty:
        write_json(out / "rq3_replay_aggregate.json", aggregate)
        pd.DataFrame(
            columns=[
                "r",
                "n_steps",
                "mean_fer",
                "mean_ter",
                "mean_sufficiency_rate",
                "strict_sufficient_steps",
                "cost_vs_r1",
                "total_approx_tokens",
            ]
        ).to_csv(out / "rq3_replay_cost.csv", index=False)
        return

    r1_tokens = float(replay_df.loc[replay_df["r"].eq(1), "approx_tokens"].sum()) if (replay_df["r"].eq(1)).any() else 0.0
    r1_tokens = r1_tokens if r1_tokens > 0 else 1.0
    for r in replay_grid:
        subset = replay_df[replay_df["r"].eq(r)]
        if subset.empty:
            continue
        item = {
            "r": int(r),
            "n_steps": int(len(subset)),
            "mean_fer": float(subset["minobs_fer"].mean()),
            "mean_ter": float(subset["minobs_ter"].mean()),
            "mean_sufficiency_rate": float(subset["sufficiency_rate"].mean()),
            "strict_sufficient_steps": int(subset["strict_sufficient"].astype(bool).sum()),
            "cost_vs_r1": float(subset["approx_tokens"].sum() / r1_tokens),
            "total_approx_tokens": int(subset["approx_tokens"].sum()),
        }
        cost_rows.append(item)
        aggregate["by_r"][str(r)] = item
    pd.DataFrame(cost_rows).to_csv(out / "rq3_replay_cost.csv", index=False)
    write_json(out / "rq3_replay_aggregate.json", aggregate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live RQ3 runner (no simulated metrics).")
    parser.add_argument("--modules", default="A,B,C", help="Comma-separated subset of A,B,C")
    parser.add_argument("--rq1-final", type=Path, default=Path("data/rq1_complete/rq1_final_aggregate.json"))
    parser.add_argument("--rq1-plans", type=Path, default=Path("data/rq1_complete/rq1_plans.json"))
    parser.add_argument("--rq1-steps", type=Path, default=Path("data/rq1_complete/rq1_step_summary.csv"))
    parser.add_argument("--rq1-traces", type=Path, default=Path("data/rq1_complete/clean_traces_rq1.jsonl"))
    parser.add_argument("--reuse-rq2-runs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/rq3_complete"))
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--agentdojo-model", default="openai-compatible")
    parser.add_argument("--model-id", default="gpt-4o-2024-08-06")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--attack-model-name", default="GPT-4o")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--utility-repeats", type=int, default=5)
    parser.add_argument("--replay-grid", default="1,3,5,10")
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--injection-task-ids", default=None)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--skip-pipeline", action="store_true", help="RQ3-A: learn/write masks only")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modules = {item.strip().upper() for item in args.modules.split(",") if item.strip()}
    unknown = modules - {"A", "B", "C"}
    if unknown:
        raise SystemExit(f"unknown --modules: {sorted(unknown)}")

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1")
    needs_client = "A" in modules or "C" in modules
    client: Optional[OpenAI] = None
    if needs_client:
        if not api_key:
            raise SystemExit("OPENAI_COMPATIBLE_API_KEY is not set.")
        client = OpenAI(api_key=api_key, base_url=base_url)

    write_json(
        args.output_dir / "rq3_run_config.json",
        {
            "rq": "RQ3",
            "data_status": "LIVE",
            "modules": sorted(modules),
            "suite": args.suite,
            "benchmark_version": args.benchmark_version,
            "model_id": args.model_id,
            "attack": args.attack,
            "repeats": args.repeats,
            "utility_repeats": args.utility_repeats,
            "replay_grid": args.replay_grid,
            "w_o_sufficiency_rule": (
                "Greedy field_priority deletion without replay/judge; "
                "a field is dropped only if another top-level field remains."
            ),
            "single_pass_rule": "Same verifier as Full MinObs, one deletion scan, no fixed-point rescan.",
            "fail_open_rule": "Frozen MinObs mask; plan miss exposes the raw observation.",
            "note": "This runner never scales metrics to manuscript targets.",
        },
    )

    if "A" in modules:
        assert client is not None
        run_rq3_a(args, client)
    if "B" in modules:
        run_rq3_b(args)
    if "C" in modules:
        assert client is not None
        run_rq3_c(args, client)

    print(f"Wrote live RQ3 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
