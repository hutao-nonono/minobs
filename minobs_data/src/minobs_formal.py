"""
minobs_formal.py
================

Paper-aligned MinObs core.

State
-----
    U           user task, represented separately
    S_t = (T_t, H_t)
        T_t     trusted control state
        H_t     previously admitted external observations (untrusted)
    C_t = A(U, S_t)
    O_t         raw tool observation (untrusted)

Contract boundary (paper Sec. 3.1 / 3.4)
----------------------------------------
An Observation Contract is learned only during trusted clean-side profiling.
Its candidate field universe may come from a trusted tool schema or, when
explicitly enabled, from top-level keys of a clean structured O_t. Once the
contract is frozen, live untrusted O_t or H_t MUST NOT add or restore fields.

Projection
----------
    O_t* = P(U, T_t, C_t, O_t)

P does not rewrite the true tool result. It only decides which already
returned fields may enter the next model context.

Sufficiency is operational reference-decision preservation, not a claim
of task-minimal observation under every valid policy.

Greedy fixed-point search returns a deletion-minimal F* under a fixed
deletion order. It does not prove global minimum cardinality.

Protected rollout matches a frozen contract to the live tool-call template.
Tool/argument mismatches and, when supplied by the runtime, trajectory/schema
mismatches fail-close to an empty payload and never fall back to Raw.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MinObsError(Exception):
    """Base error for MinObs."""


class ExtractionError(MinObsError):
    """Raised when an observation cannot be extracted under a contract."""


class ValidationError(MinObsError):
    """Raised when a value violates a typed field contract."""


class BaselineUnstableError(MinObsError):
    """Raised when the full observation cannot stably reproduce the reference."""


class ContractBoundaryError(MinObsError):
    """Raised when a contract would be expanded from untrusted text."""


class PlanMismatchError(MinObsError):
    """Raised when a live tool call does not match a frozen clean plan."""


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    path: str
    dtype: str = "any"
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_length: Optional[int] = None
    description: str = ""


@dataclass
class TrustedControlState:
    """
    T_t: trusted control state.

    Must not contain free-text from O_t or H_t.
    schema_fields come from the tool JSON schema, not from a live payload.
    """

    system_instruction: str = ""
    tool_name: str = ""
    tool_arguments: Mapping[str, Any] = field(default_factory=dict)
    trajectory_slot: str = ""
    schema_fields: List[FieldSpec] = field(default_factory=list)
    schema_id: str = ""

    def schema_names(self) -> List[str]:
        return [item.path for item in self.schema_fields]


@dataclass
class ObservationContract:
    task_id: str
    step_id: str
    tool_name: str
    fields: List[FieldSpec]
    current_objective: str = ""
    source: str = "trusted_schema"
    schema_id: str = ""
    trajectory_slot: str = ""

    def field_names(self) -> List[str]:
        return [item.path for item in self.fields]

    def subset(
        self,
        keep_fields: Iterable[str],
    ) -> "ObservationContract":
        keep = set(keep_fields)

        return ObservationContract(
            task_id=self.task_id,
            step_id=self.step_id,
            tool_name=self.tool_name,
            current_objective=self.current_objective,
            source=self.source,
            schema_id=self.schema_id,
            trajectory_slot=self.trajectory_slot,
            fields=[
                copy.deepcopy(item)
                for item in self.fields
                if item.path in keep
            ],
        )


@dataclass
class StepContext:
    """
    One fixed trajectory state for step replay.

    user_task:
        U, kept separate from T_t.

    trusted_state:
        T_t. Contract generation may read this object.

    history / admitted_history:
        H_t, previously admitted external observations. Untrusted.
        Must not expand a contract.

    raw_observation:
        Payload O_t. During protected execution it is untrusted and is used
        only by projection/replay. During explicitly trusted clean-side
        profiling, its structured top-level keys may seed a candidate
        contract when schema metadata is unavailable.

    trusted_metadata:
        Non-ablatable execution metadata (tool success/failure).
    """

    task_id: str
    step_id: str
    user_task: str
    history: Sequence[Any]
    current_tool_call: Mapping[str, Any]
    raw_observation: Any
    reference_decision: "DecisionSignature"
    trusted_metadata: Mapping[str, Any] = field(default_factory=dict)
    trusted_state: Optional[TrustedControlState] = None
    admitted_history: Sequence[Any] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionSignature:
    """
    Provider-independent next-action representation.

    kind:
        "tool_call" or "final_answer"

    payload:
        Adapter-defined canonical JSON-like content.
    """

    kind: str
    payload: Any

    def canonical(self) -> str:
        return canonical_json(
            {
                "kind": self.kind,
                "payload": self.payload,
            }
        )


@dataclass
class VerificationAttempt:
    repeat: int
    decision_consistent: bool
    task_constraints_satisfied: bool
    sufficient: bool
    candidate_decision: DecisionSignature
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    sufficient: bool
    attempts: List[VerificationAttempt]

    @property
    def pass_count(self) -> int:
        return sum(
            1
            for attempt in self.attempts
            if attempt.sufficient
        )

    @property
    def repeat_count(self) -> int:
        return len(self.attempts)


@dataclass
class FieldAblationRecord:
    attempted_field: str
    fields_before: List[str]
    proposed_fields: List[str]
    removed: bool
    verification: VerificationResult


@dataclass
class MinimizationResult:
    status: str
    context: StepContext
    candidate_contract: ObservationContract
    minimal_contract: ObservationContract
    candidate_observation: Any
    minimal_observation: Any
    baseline_verification: VerificationResult
    empty_observation_verification: VerificationResult
    ablation_trace: List[FieldAblationRecord]
    minimality_type: str = "greedy_deletion_minimal"
    globally_minimum_proven: bool = False

    @property
    def initial_fields(self) -> List[str]:
        return self.candidate_contract.field_names()

    @property
    def minimal_fields(self) -> List[str]:
        return self.minimal_contract.field_names()

    @property
    def removed_fields(self) -> List[str]:
        remaining = set(self.minimal_fields)
        return [
            field_name
            for field_name in self.initial_fields
            if field_name not in remaining
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "task_id": self.context.task_id,
            "step_id": self.context.step_id,
            "user_task": self.context.user_task,
            "current_tool_call": dict(self.context.current_tool_call),
            "trusted_metadata": dict(self.context.trusted_metadata),
            "reference_decision": {
                "kind": self.context.reference_decision.kind,
                "payload": self.context.reference_decision.payload,
            },
            "initial_fields": self.initial_fields,
            "minimal_fields": self.minimal_fields,
            "removed_fields": self.removed_fields,
            "candidate_observation": self.candidate_observation,
            "minimal_observation": self.minimal_observation,
            "baseline_verification": verification_to_dict(
                self.baseline_verification
            ),
            "empty_observation_verification": verification_to_dict(
                self.empty_observation_verification
            ),
            "minimality_claim": {
                "type": self.minimality_type,
                "globally_minimum_proven": self.globally_minimum_proven,
                "description": (
                    "Fields are greedily deleted while sufficiency is preserved. "
                    "The result is deletion-minimal under the tested search path; "
                    "it is not a proof of global minimum cardinality."
                ),
            },
            "ablation_trace": [
                {
                    "attempted_field": item.attempted_field,
                    "fields_before": item.fields_before,
                    "proposed_fields": item.proposed_fields,
                    "removed": item.removed,
                    "verification": verification_to_dict(
                        item.verification
                    ),
                }
                for item in self.ablation_trace
            ],
        }


@dataclass
class ExposureMetrics:
    raw_tokens_approx: int
    candidate_tokens_approx: int
    minimal_tokens_approx: int
    token_exposure_ratio_approx: float
    initial_field_count: int
    minimal_field_count: int
    field_exposure_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Canonicalization / metrics
# ---------------------------------------------------------------------------

def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            _normalize_json(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _normalize_json(item)
            for item in value
        ]

    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def approximate_token_count(obj: Any) -> int:
    """
    Rough provider-independent estimate.

    The paper should label this as approximate TER unless the actual tested
    model tokenizer/provider usage is used.
    """
    text = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return max(
        1,
        math.ceil(
            len(text) / 4
        ),
    )


# ---------------------------------------------------------------------------
# Contract extraction / typed validation
# ---------------------------------------------------------------------------

def _get_by_path(
    record: Mapping[str, Any],
    path: str,
) -> Any:
    current: Any = record

    for part in path.split("."):
        if (
            not isinstance(current, Mapping)
            or part not in current
        ):
            raise KeyError(path)

        current = current[part]

    return current


def _set_by_path(
    target: Dict[str, Any],
    path: str,
    value: Any,
) -> None:
    parts = path.split(".")
    current = target

    for part in parts[:-1]:
        current = current.setdefault(
            part,
            {},
        )

    current[parts[-1]] = value


def _looks_like_email(
    value: str,
) -> bool:
    if "@" not in value:
        return False

    local, domain = value.rsplit(
        "@",
        1,
    )

    return bool(
        local
        and "." in domain
        and " " not in value
    )


def infer_dtype(
    values: Sequence[Any],
) -> str:
    non_null = [
        value
        for value in values
        if value is not None
    ]

    if not non_null:
        return "any"

    if all(
        isinstance(value, bool)
        for value in non_null
    ):
        return "bool"

    if all(
        isinstance(value, int)
        and not isinstance(value, bool)
        for value in non_null
    ):
        return "int"

    if all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        for value in non_null
    ):
        return "float"

    if all(
        isinstance(value, str)
        for value in non_null
    ):
        if all(
            _looks_like_email(value)
            for value in non_null
        ):
            return "email"

        return "string"

    return "any"


def validate_value(
    value: Any,
    spec: FieldSpec,
) -> Any:
    dtype = spec.dtype.lower()

    if value is None:
        return None

    if dtype == "any":
        normalized = value

    elif dtype in {
        "string",
        "identifier",
        "datetime",
    }:
        if not isinstance(value, str):
            raise ValidationError(
                f"{spec.path}: expected {dtype}, "
                f"got {type(value).__name__}"
            )
        normalized = value

    elif dtype == "email":
        if (
            not isinstance(value, str)
            or not _looks_like_email(value)
        ):
            raise ValidationError(
                f"{spec.path}: invalid email"
            )
        normalized = value

    elif dtype == "bool":
        if not isinstance(value, bool):
            raise ValidationError(
                f"{spec.path}: expected bool"
            )
        normalized = value

    elif dtype == "int":
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise ValidationError(
                f"{spec.path}: expected int"
            )
        normalized = value

    elif dtype in {
        "float",
        "decimal",
    }:
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise ValidationError(
                f"{spec.path}: expected numeric"
            )
        normalized = float(value)

    else:
        raise ValidationError(
            f"{spec.path}: unsupported dtype={spec.dtype}"
        )

    if (
        isinstance(normalized, str)
        and spec.max_length is not None
        and len(normalized) > spec.max_length
    ):
        raise ValidationError(
            f"{spec.path}: string too long"
        )

    if (
        isinstance(normalized, (int, float))
        and not isinstance(normalized, bool)
    ):
        if (
            spec.min_value is not None
            and normalized < spec.min_value
        ):
            raise ValidationError(
                f"{spec.path}: below min_value"
            )

        if (
            spec.max_value is not None
            and normalized > spec.max_value
        ):
            raise ValidationError(
                f"{spec.path}: above max_value"
            )

    return normalized


def _observation_rows(
    raw_observation: Any,
) -> Tuple[List[Mapping[str, Any]], bool]:
    if isinstance(
        raw_observation,
        Mapping,
    ):
        return [
            raw_observation
        ], True

    if (
        isinstance(
            raw_observation,
            list,
        )
        and all(
            isinstance(
                item,
                Mapping,
            )
            for item in raw_observation
        )
    ):
        return list(
            raw_observation
        ), False

    raise ExtractionError(
        "Field-level MinObs currently expects dict or list[dict]."
    )


def extract_observation(
    raw_observation: Any,
    contract: ObservationContract,
) -> Any:
    rows, return_single = _observation_rows(
        raw_observation
    )

    output: List[
        Dict[str, Any]
    ] = []

    for row_index, row in enumerate(
        rows
    ):
        projected: Dict[
            str,
            Any,
        ] = {}

        for spec in contract.fields:
            try:
                value = _get_by_path(
                    row,
                    spec.path,
                )
            except KeyError:
                if spec.required:
                    raise ExtractionError(
                        f"row={row_index}: missing required "
                        f"field {spec.path!r}"
                    )
                continue

            normalized = validate_value(
                value,
                spec,
            )

            _set_by_path(
                projected,
                spec.path,
                normalized,
            )

        output.append(
            projected
        )

    if return_single:
        return output[0]

    return output


def schema_field_specs(
    schema_fields: Sequence[FieldSpec],
) -> List[FieldSpec]:
    return [copy.deepcopy(item) for item in schema_fields]


def discard_unknown_fields(
    fields: Sequence[FieldSpec],
    schema_names: Iterable[str],
) -> List[FieldSpec]:
    allowed = set(schema_names)
    return [
        copy.deepcopy(item)
        for item in fields
        if item.path in allowed
    ]


def restrict_contract_to_clean_presence(
    contract: ObservationContract,
    raw_observation: Any,
) -> ObservationContract:
    """
    Instantiate a trusted-schema contract for one clean observation.

    Raw O_t may only determine whether a schema-defined path is present in
    the clean instance. It may not invent field names or field types.
    """
    rows, _ = _observation_rows(raw_observation)
    present: List[str] = []

    for spec in contract.fields:
        found = False
        for row in rows:
            try:
                _get_by_path(row, spec.path)
                found = True
                break
            except KeyError:
                continue
        if found:
            present.append(spec.path)

    narrowed = contract.subset(present)
    narrowed.source = "trusted_schema_clean_presence"
    return narrowed


def build_trusted_contract(
    *,
    task_id: str,
    step_id: str,
    user_task: str,
    trusted_state: TrustedControlState,
    current_tool_call: Mapping[str, Any],
    current_objective: str = "",
    analyzer_fn: Optional[
        Callable[
            [str, TrustedControlState, Mapping[str, Any]],
            Sequence[FieldSpec],
        ]
    ] = None,
) -> ObservationContract:
    """
    Paper-standard contract constructor.

    Inputs are only U, T_t and C_t. The optional analyzer_fn has the same
    signature and must not receive O_t or H_t.
    """
    if not trusted_state.schema_fields:
        raise ContractBoundaryError(
            "TrustedControlState.schema_fields is empty. "
            "A contract cannot be inferred from raw observation."
        )

    tool_name = str(
        current_tool_call.get("tool")
        or current_tool_call.get("function")
        or trusted_state.tool_name
        or ""
    )

    if analyzer_fn is None:
        selected = schema_field_specs(trusted_state.schema_fields)
    else:
        selected = list(
            analyzer_fn(
                user_task,
                trusted_state,
                current_tool_call,
            )
        )

    selected = discard_unknown_fields(
        selected,
        trusted_state.schema_names(),
    )

    if not selected:
        selected = []

    return ObservationContract(
        task_id=str(task_id),
        step_id=str(step_id),
        tool_name=tool_name,
        fields=selected,
        current_objective=current_objective,
        source="trusted_schema",
        schema_id=trusted_state.schema_id,
        trajectory_slot=trusted_state.trajectory_slot,
    )


def project_observation(
    raw_observation: Any,
    contract: ObservationContract,
) -> Any:
    """P: keep only contracted fields from O_t."""
    return extract_observation(raw_observation, contract)


def empty_payload_for(observation: Any) -> Any:
    if isinstance(observation, Mapping):
        return {}
    if isinstance(observation, list) and all(
        isinstance(item, Mapping) for item in observation
    ):
        return [{} for _ in observation]
    return {}


def fail_closed_project(
    *,
    live_tool_call: Mapping[str, Any],
    live_observation: Any,
    frozen_contract: ObservationContract,
    expected_tool: str,
    expected_arguments: Optional[Mapping[str, Any]] = None,
    live_trajectory_slot: Optional[str] = None,
    expected_trajectory_slot: Optional[str] = None,
    live_schema_id: Optional[str] = None,
    expected_schema_id: Optional[str] = None,
    arguments_equal_fn: Optional[
        Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    ] = None,
) -> Tuple[Any, str]:
    """
    Protected-rollout gate.

    The formal paper path binds a frozen mask to the clean tool-call
    template, trajectory slot, and schema identity. Any supplied mismatch
    fail-closes to an empty payload. Raw O_t is never returned.

    For backward compatibility, trajectory/schema checks are activated when
    the corresponding live or expected values are supplied by the runtime.
    """
    live_tool = str(
        live_tool_call.get("tool")
        or live_tool_call.get("function")
        or ""
    )
    if live_tool != str(expected_tool):
        return empty_payload_for(live_observation), "tool_mismatch"

    if expected_arguments is not None:
        live_args = dict(
            live_tool_call.get("arguments")
            or live_tool_call.get("args")
            or {}
        )
        if arguments_equal_fn is None:
            match = canonical_json(live_args) == canonical_json(
                dict(expected_arguments)
            )
        else:
            match = arguments_equal_fn(live_args, dict(expected_arguments))
        if not match:
            return empty_payload_for(live_observation), "argument_mismatch"

    bound_slot = (
        expected_trajectory_slot
        if expected_trajectory_slot is not None
        else frozen_contract.trajectory_slot
    )
    if (
        bound_slot
        and live_trajectory_slot is not None
        and str(live_trajectory_slot) != str(bound_slot)
    ):
        return empty_payload_for(live_observation), "trajectory_mismatch"

    bound_schema = (
        expected_schema_id
        if expected_schema_id is not None
        else frozen_contract.schema_id
    )
    if (
        bound_schema
        and live_schema_id is not None
        and str(live_schema_id) != str(bound_schema)
    ):
        return empty_payload_for(live_observation), "schema_id_mismatch"

    try:
        projected = project_observation(live_observation, frozen_contract)
    except MinObsError:
        return empty_payload_for(live_observation), "schema_mismatch"

    return projected, "ok"

def is_deletion_minimal(
    *,
    context: "StepContext",
    contract: ObservationContract,
    verifier: "SufficiencyVerifier",
    raw_observation: Any,
) -> bool:
    """
    Check Sufficient(F*) = 1 and every singleton deletion is insufficient.
    """
    full = extract_observation(raw_observation, contract)
    if not verifier.verify(context=context, observation=full).sufficient:
        return False

    for name in contract.field_names():
        reduced_contract = contract.subset(
            [item for item in contract.field_names() if item != name]
        )
        reduced = extract_observation(raw_observation, reduced_contract)
        if verifier.verify(context=context, observation=reduced).sufficient:
            return False
    return True


def build_full_field_contract(
    *,
    task_id: str,
    step_id: str,
    current_tool_call: Mapping[str, Any],
    raw_observation: Any,
    current_objective: str = "",
    trusted_state: Optional[TrustedControlState] = None,
    allow_raw_field_discovery: bool = False,
) -> ObservationContract:
    """
    Candidate-contract helper.

    Paper-standard path: pass trusted_state with schema_fields. Raw O_t
    is used only during trusted clean-side profiling to mark which
    schema-defined paths are present; it cannot invent names or types.

    Trusted clean-profile path: allow_raw_field_discovery=True reconstructs
    the candidate field universe from the top-level keys of a clean structured
    observation. The resulting contract must be frozen before protected
    rollout; live untrusted observations must never use this discovery path.
    """
    if trusted_state is not None and trusted_state.schema_fields:
        trusted_contract = build_trusted_contract(
            task_id=task_id,
            step_id=step_id,
            user_task="",
            trusted_state=trusted_state,
            current_tool_call=current_tool_call,
            current_objective=current_objective,
        )
        return restrict_contract_to_clean_presence(
            trusted_contract,
            raw_observation,
        )

    if not allow_raw_field_discovery:
        raise ContractBoundaryError(
            "Refuse to discover contract fields from raw_observation. "
            "Provide TrustedControlState.schema_fields, or explicitly "
            "set allow_raw_field_discovery=True only in trusted clean-side profiling."
        )

    rows, _ = _observation_rows(
        raw_observation
    )

    ordered_names: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            name = str(key)

            if name not in seen:
                seen.add(name)
                ordered_names.append(
                    name
                )

    fields: List[
        FieldSpec
    ] = []

    for name in ordered_names:
        present_values = [
            row[name]
            for row in rows
            if name in row
        ]

        required = (
            len(present_values)
            == len(rows)
        )

        fields.append(
            FieldSpec(
                path=name,
                dtype=infer_dtype(
                    present_values
                ),
                required=required,
            )
        )

    tool_name = str(
        current_tool_call.get(
            "tool"
        )
        or current_tool_call.get(
            "function"
        )
        or ""
    )

    return ObservationContract(
        task_id=str(task_id),
        step_id=str(step_id),
        tool_name=tool_name,
        fields=fields,
        current_objective=(
            current_objective
        ),
        source="trusted_clean_raw_keys",
        schema_id=(
            trusted_state.schema_id
            if trusted_state is not None
            else ""
        ),
        trajectory_slot=(
            trusted_state.trajectory_slot
            if trusted_state is not None
            else ""
        ),
    )


# ---------------------------------------------------------------------------
# Sufficiency verification
# ---------------------------------------------------------------------------

ReplayDecisionFn = Callable[
    [StepContext, Any],
    DecisionSignature,
]

EvaluateDecisionFn = Callable[
    [
        StepContext,
        DecisionSignature,
        DecisionSignature,
        Any,
    ],
    Tuple[
        bool,
        bool,
        str,
        Mapping[str, Any],
    ],
]


class SufficiencyVerifier:
    """
    Replays the same trajectory state with a reduced observation.

    evaluate_decision_fn must return:

        (
            decision_consistent,
            task_constraints_satisfied,
            reason,
            metadata,
        )
    """

    def __init__(
        self,
        *,
        replay_decision_fn: ReplayDecisionFn,
        evaluate_decision_fn: EvaluateDecisionFn,
        repeats: int = 1,
    ) -> None:
        self.replay_decision_fn = (
            replay_decision_fn
        )
        self.evaluate_decision_fn = (
            evaluate_decision_fn
        )
        self.repeats = max(
            1,
            int(repeats),
        )

    def verify(
        self,
        *,
        context: StepContext,
        observation: Any,
    ) -> VerificationResult:
        attempts: List[
            VerificationAttempt
        ] = []

        for repeat_index in range(
            self.repeats
        ):
            candidate = (
                self.replay_decision_fn(
                    context,
                    observation,
                )
            )

            (
                decision_consistent,
                constraints_ok,
                reason,
                metadata,
            ) = self.evaluate_decision_fn(
                context,
                context.reference_decision,
                candidate,
                observation,
            )

            sufficient = (
                decision_consistent
                and constraints_ok
            )

            attempts.append(
                VerificationAttempt(
                    repeat=(
                        repeat_index + 1
                    ),
                    decision_consistent=(
                        decision_consistent
                    ),
                    task_constraints_satisfied=(
                        constraints_ok
                    ),
                    sufficient=(
                        sufficient
                    ),
                    candidate_decision=(
                        candidate
                    ),
                    reason=reason,
                    metadata=dict(
                        metadata
                        or {}
                    ),
                )
            )

        return VerificationResult(
            sufficient=all(
                attempt.sufficient
                for attempt in attempts
            ),
            attempts=attempts,
        )


# ---------------------------------------------------------------------------
# Greedy sufficiency-guided minimization
# ---------------------------------------------------------------------------

FieldPriorityFn = Callable[
    [FieldSpec],
    float,
]

ProgressFn = Callable[
    [str, Mapping[str, Any]],
    None,
]


class SufficiencyMinimizer:
    """
    Greedy backward elimination with strict baseline stability.

    A field is removed only if ALL repeated verification attempts remain
    sufficient.

    Empty contracts are explicitly tested.
    """

    def __init__(
        self,
        *,
        verifier: SufficiencyVerifier,
        field_priority_fn: Optional[
            FieldPriorityFn
        ] = None,
        repeat_until_fixed_point: bool = True,
        progress_fn: Optional[
            ProgressFn
        ] = None,
    ) -> None:
        self.verifier = verifier
        self.field_priority_fn = (
            field_priority_fn
        )
        self.repeat_until_fixed_point = (
            repeat_until_fixed_point
        )
        self.progress_fn = (
            progress_fn
        )

    def _progress(
        self,
        event: str,
        **data: Any,
    ) -> None:
        if self.progress_fn is not None:
            self.progress_fn(
                event,
                data,
            )

    def minimize(
        self,
        *,
        context: StepContext,
        candidate_contract: ObservationContract,
    ) -> MinimizationResult:
        candidate_observation = (
            extract_observation(
                context.raw_observation,
                candidate_contract,
            )
        )

        self._progress(
            "baseline_start",
            fields=(
                candidate_contract
                .field_names()
            ),
        )

        baseline = (
            self.verifier.verify(
                context=context,
                observation=(
                    candidate_observation
                ),
            )
        )

        self._progress(
            "baseline_result",
            sufficient=(
                baseline.sufficient
            ),
            pass_count=(
                baseline.pass_count
            ),
            repeat_count=(
                baseline.repeat_count
            ),
        )

        if not baseline.sufficient:
            empty_contract = (
                candidate_contract
                .subset([])
            )

            empty_observation = (
                extract_observation(
                    context.raw_observation,
                    empty_contract,
                )
            )

            empty_verification = (
                VerificationResult(
                    sufficient=False,
                    attempts=[],
                )
            )

            return MinimizationResult(
                status="baseline_unstable",
                context=context,
                candidate_contract=(
                    candidate_contract
                ),
                minimal_contract=(
                    candidate_contract
                ),
                candidate_observation=(
                    candidate_observation
                ),
                minimal_observation=(
                    candidate_observation
                ),
                baseline_verification=(
                    baseline
                ),
                empty_observation_verification=(
                    empty_verification
                ),
                ablation_trace=[],
            )

        current_fields = (
            candidate_contract
            .field_names()
        )

        trace: List[
            FieldAblationRecord
        ] = []

        while True:
            changed = False

            specs = [
                spec
                for spec
                in candidate_contract.fields
                if spec.path
                in current_fields
            ]

            if (
                self.field_priority_fn
                is not None
            ):
                specs = sorted(
                    specs,
                    key=(
                        self.field_priority_fn
                    ),
                )

            for spec in specs:
                if (
                    spec.path
                    not in current_fields
                ):
                    continue

                before = list(
                    current_fields
                )

                proposed = [
                    field_name
                    for field_name
                    in current_fields
                    if field_name
                    != spec.path
                ]

                self._progress(
                    "field_test_start",
                    attempted_field=(
                        spec.path
                    ),
                    fields_before=before,
                    proposed_fields=(
                        proposed
                    ),
                )

                proposed_contract = (
                    candidate_contract
                    .subset(
                        proposed
                    )
                )

                try:
                    reduced = (
                        extract_observation(
                            context.raw_observation,
                            proposed_contract,
                        )
                    )

                    verification = (
                        self.verifier
                        .verify(
                            context=context,
                            observation=(
                                reduced
                            ),
                        )
                    )

                except MinObsError as exc:
                    fallback_attempt = (
                        VerificationAttempt(
                            repeat=1,
                            decision_consistent=False,
                            task_constraints_satisfied=False,
                            sufficient=False,
                            candidate_decision=(
                                context
                                .reference_decision
                            ),
                            reason=(
                                "Extraction/validation "
                                f"failed: {exc}"
                            ),
                        )
                    )

                    verification = (
                        VerificationResult(
                            sufficient=False,
                            attempts=[
                                fallback_attempt
                            ],
                        )
                    )

                removed = (
                    verification
                    .sufficient
                )

                if removed:
                    current_fields = (
                        proposed
                    )
                    changed = True

                trace.append(
                    FieldAblationRecord(
                        attempted_field=(
                            spec.path
                        ),
                        fields_before=(
                            before
                        ),
                        proposed_fields=(
                            proposed
                        ),
                        removed=removed,
                        verification=(
                            verification
                        ),
                    )
                )

                self._progress(
                    "field_test_result",
                    attempted_field=(
                        spec.path
                    ),
                    removed=removed,
                    sufficient=(
                        verification
                        .sufficient
                    ),
                    pass_count=(
                        verification
                        .pass_count
                    ),
                    repeat_count=(
                        verification
                        .repeat_count
                    ),
                )

            if (
                not self.repeat_until_fixed_point
                or not changed
            ):
                break

        minimal_contract = (
            candidate_contract
            .subset(
                current_fields
            )
        )

        minimal_observation = (
            extract_observation(
                context.raw_observation,
                minimal_contract,
            )
        )

        empty_contract = (
            candidate_contract
            .subset([])
        )

        empty_observation = (
            extract_observation(
                context.raw_observation,
                empty_contract,
            )
        )

        empty_verification = (
            self.verifier.verify(
                context=context,
                observation=(
                    empty_observation
                ),
            )
        )

        return MinimizationResult(
            status="ok",
            context=context,
            candidate_contract=(
                candidate_contract
            ),
            minimal_contract=(
                minimal_contract
            ),
            candidate_observation=(
                candidate_observation
            ),
            minimal_observation=(
                minimal_observation
            ),
            baseline_verification=(
                baseline
            ),
            empty_observation_verification=(
                empty_verification
            ),
            ablation_trace=trace,
        )


# ---------------------------------------------------------------------------
# End-to-end wrapper
# ---------------------------------------------------------------------------

CandidateContractFn = Callable[
    [StepContext],
    ObservationContract,
]


class MinObs:
    def __init__(
        self,
        *,
        candidate_contract_fn: CandidateContractFn,
        minimizer: SufficiencyMinimizer,
    ) -> None:
        self.candidate_contract_fn = (
            candidate_contract_fn
        )
        self.minimizer = minimizer

    def run(
        self,
        *,
        context: StepContext,
    ) -> Tuple[
        MinimizationResult,
        ExposureMetrics,
    ]:
        contract = (
            self.candidate_contract_fn(
                context
            )
        )

        if (
            context.trusted_state is not None
            and context.trusted_state.schema_fields
        ):
            contract.fields = discard_unknown_fields(
                contract.fields,
                context.trusted_state.schema_names(),
            )
            contract.source = (
                contract.source
                if contract.source.startswith("trusted_schema")
                else "trusted_schema"
            )
            contract.schema_id = context.trusted_state.schema_id
            contract.trajectory_slot = context.trusted_state.trajectory_slot

        result = (
            self.minimizer.minimize(
                context=context,
                candidate_contract=(
                    contract
                ),
            )
        )

        metrics = (
            compute_exposure_metrics(
                context.raw_observation,
                result,
            )
        )

        return (
            result,
            metrics,
        )


def compute_exposure_metrics(
    raw_observation: Any,
    result: MinimizationResult,
) -> ExposureMetrics:
    raw_tokens = (
        approximate_token_count(
            raw_observation
        )
    )

    candidate_tokens = (
        approximate_token_count(
            result
            .candidate_observation
        )
    )

    minimal_tokens = (
        approximate_token_count(
            result
            .minimal_observation
        )
    )

    initial_count = len(
        result.initial_fields
    )

    minimal_count = len(
        result.minimal_fields
    )

    return ExposureMetrics(
        raw_tokens_approx=(
            raw_tokens
        ),
        candidate_tokens_approx=(
            candidate_tokens
        ),
        minimal_tokens_approx=(
            minimal_tokens
        ),
        token_exposure_ratio_approx=(
            minimal_tokens
            / max(
                raw_tokens,
                1,
            )
        ),
        initial_field_count=(
            initial_count
        ),
        minimal_field_count=(
            minimal_count
        ),
        field_exposure_ratio=(
            minimal_count
            / max(
                initial_count,
                1,
            )
        ),
    )


def verification_to_dict(
    verification: VerificationResult,
) -> Dict[str, Any]:
    return {
        "sufficient": (
            verification.sufficient
        ),
        "pass_count": (
            verification.pass_count
        ),
        "repeat_count": (
            verification.repeat_count
        ),
        "attempts": [
            {
                "repeat": (
                    attempt.repeat
                ),
                "decision_consistent": (
                    attempt
                    .decision_consistent
                ),
                "task_constraints_satisfied": (
                    attempt
                    .task_constraints_satisfied
                ),
                "sufficient": (
                    attempt.sufficient
                ),
                "candidate_decision": {
                    "kind": (
                        attempt
                        .candidate_decision
                        .kind
                    ),
                    "payload": (
                        attempt
                        .candidate_decision
                        .payload
                    ),
                },
                "reason": (
                    attempt.reason
                ),
                "metadata": (
                    attempt.metadata
                ),
            }
            for attempt
            in verification.attempts
        ],
    }
