r"""
collect_clean.py
================

Parse AgentDojo benign-run logs into MinObs-friendly JSONL records.

Example usage from the AgentDojo repo root:

    uv run python collect_clean.py ^
        --input runs\deepseek_raw ^
        --output data\clean_traces.jsonl

This version additionally normalizes YAML-parsed datetime/date/time values
before JSON serialization.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import yaml
except ImportError:
    yaml = None


def content_blocks_to_text(content: Any) -> str:
    """Convert AgentDojo content blocks to plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("content")
                if value is not None:
                    parts.append(str(value))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return str(content)


def make_json_serializable(value: Any) -> Any:
    """
    Recursively convert objects produced by YAML parsing into JSON-safe values.

    PyYAML may automatically parse strings such as
    '2024-05-26 19:00:00' into datetime objects, which json.dumps()
    cannot serialize by default.
    """
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): make_json_serializable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(v) for v in value]
    return value


def parse_observation(text: str) -> tuple[Any, str]:
    """
    Parse a tool observation.

    Returns:
        (parsed_value, parser_name)

    parser_name is one of:
        yaml
        json
        text
    """
    stripped = text.strip()

    if not stripped:
        return "", "text"

    # AgentDojo v1.2.2 logs observed here are YAML-like.
    if yaml is not None:
        try:
            value = yaml.safe_load(stripped)
            return make_json_serializable(value), "yaml"
        except Exception:
            pass

    try:
        value = json.loads(stripped)
        return make_json_serializable(value), "json"
    except Exception:
        return text, "text"


def canonical_tool_call(tool_call: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if tool_call is None:
        return None

    return {
        "tool": tool_call.get("function"),
        "arguments": make_json_serializable(tool_call.get("args") or {}),
        "id": tool_call.get("id"),
    }


def assistant_to_action(message: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert the next assistant message into a stable action object."""
    if message is None:
        return {
            "type": "missing",
            "tool_calls": [],
            "final_answer": None,
        }

    tool_calls = message.get("tool_calls")
    text = content_blocks_to_text(message.get("content"))

    if tool_calls:
        return {
            "type": "tool_call",
            "tool_calls": [
                canonical_tool_call(call)
                for call in tool_calls
            ],
            "final_answer": text or None,
        }

    return {
        "type": "final_answer",
        "tool_calls": [],
        "final_answer": text or None,
    }


def find_user_task(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return content_blocks_to_text(message.get("content"))
    return ""


def find_next_assistant(
    messages: List[Dict[str, Any]],
    start_index: int,
) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
    for idx in range(start_index + 1, len(messages)):
        if messages[idx].get("role") == "assistant":
            return messages[idx], idx
    return None, None


def count_consecutive_tool_results_around(
    messages: List[Dict[str, Any]],
    tool_index: int,
) -> int:
    """Count tool messages in the current tool-result block."""
    left = tool_index
    while left - 1 >= 0 and messages[left - 1].get("role") == "tool":
        left -= 1

    right = tool_index
    while right + 1 < len(messages) and messages[right + 1].get("role") == "tool":
        right += 1

    return right - left + 1


def previous_assistant_tool_turn(
    messages: List[Dict[str, Any]],
    tool_index: int,
) -> Optional[Dict[str, Any]]:
    """Find the assistant message that emitted the current tool call(s)."""
    for idx in range(tool_index - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "assistant":
            return messages[idx]
        if role in {"user", "system"}:
            break
    return None


def extract_records_from_log(
    path: Path,
    *,
    include_failed_tasks: bool,
) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    # Only benign runs.
    if data.get("attack_type") is not None:
        return []
    if data.get("injection_task_id") is not None:
        return []

    utility = bool(data.get("utility", False))

    if not include_failed_tasks and not utility:
        return []

    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return []

    user_task = find_user_task(messages)

    records: List[Dict[str, Any]] = []
    tool_step = 0

    for idx, message in enumerate(messages):
        if message.get("role") != "tool":
            continue

        raw_text = content_blocks_to_text(message.get("content"))
        parsed_observation, parser_name = parse_observation(raw_text)

        tool_call = canonical_tool_call(message.get("tool_call"))
        next_assistant, next_assistant_index = find_next_assistant(messages, idx)

        emitting_assistant = previous_assistant_tool_turn(messages, idx)
        emitted_tool_calls = []
        if emitting_assistant is not None:
            emitted_tool_calls = [
                canonical_tool_call(call)
                for call in (emitting_assistant.get("tool_calls") or [])
            ]

        # History excludes the current tool result.
        history = make_json_serializable(messages[:idx])

        record = {
            "source_log": str(path),
            "suite_name": data.get("suite_name"),
            "pipeline_name": data.get("pipeline_name"),
            "benchmark_version": data.get("benchmark_version"),
            "agentdojo_package_version": data.get("agentdojo_package_version"),
            "evaluation_timestamp": data.get("evaluation_timestamp"),

            "task_id": data.get("user_task_id"),
            "step_id": tool_step,
            "user_task": user_task,

            "history": history,

            "tool_call": tool_call,
            "tool_turn_calls": emitted_tool_calls,
            "tool_turn_size": count_consecutive_tool_results_around(messages, idx),

            "raw_observation_text": raw_text,
            "raw_observation": parsed_observation,
            "observation_parser": parser_name,

            "next_action": assistant_to_action(next_assistant),
            "next_assistant_message_index": next_assistant_index,

            "task_success": utility,
            "security": data.get("security"),
            "duration": data.get("duration"),
            "error": data.get("error"),
        }

        records.append(make_json_serializable(record))
        tool_step += 1

    return records


def discover_logs(root: Path) -> Iterable[Path]:
    """Find JSON logs under a run directory."""
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="AgentDojo run directory, e.g. runs/deepseek_raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clean_traces.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--include-failed-tasks",
        action="store_true",
        help=(
            "Include benign tasks whose final utility is false. "
            "Default: keep only successful clean tasks for MinObs RQ1."
        ),
    )
    args = parser.parse_args()

    input_root: Path = args.input
    output_path: Path = args.output

    if not input_root.exists():
        raise SystemExit(f"Input path does not exist: {input_root}")

    all_records: List[Dict[str, Any]] = []
    scanned_logs = 0

    for log_path in discover_logs(input_root):
        try:
            records = extract_records_from_log(
                log_path,
                include_failed_tasks=args.include_failed_tasks,
            )
        except Exception as exc:
            print(f"[WARN] Failed to parse {log_path}: {exc}")
            continue

        scanned_logs += 1
        all_records.extend(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(
                json.dumps(
                    make_json_serializable(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

    single_tool_records = sum(
        1 for r in all_records
        if r["tool_turn_size"] == 1
    )
    parsed_records = sum(
        1 for r in all_records
        if r["observation_parser"] in {"yaml", "json"}
    )

    print("=" * 72)
    print("Clean trajectory collection complete")
    print(f"Input directory        : {input_root}")
    print(f"Scanned JSON logs      : {scanned_logs}")
    print(f"Extracted tool steps   : {len(all_records)}")
    print(f"Single-tool steps      : {single_tool_records}")
    print(f"Structured observations: {parsed_records}")
    print(f"Output                 : {output_path}")
    print("=" * 72)

    if all_records:
        print("\nFirst extracted record summary:")
        first = all_records[0]
        summary = {
            "task_id": first["task_id"],
            "step_id": first["step_id"],
            "user_task": first["user_task"],
            "tool_call": first["tool_call"],
            "observation_parser": first["observation_parser"],
            "raw_observation": first["raw_observation"],
            "next_action": first["next_action"],
            "task_success": first["task_success"],
        }
        print(
            json.dumps(
                make_json_serializable(summary),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
