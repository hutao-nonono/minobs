# MinObs: Task-Constrained Minimal Observations against Indirect Prompt Injection in LLM Agents

## Overview

This repository contains the experimental code associated with the manuscript:

**MinObs: Task-Constrained Minimal Observations against Indirect Prompt Injection in LLM Agents**

MinObs is an observation-projection approach for tool-using large language model (LLM) agents. Instead of passing complete external tool results directly into the agent context, MinObs identifies a task-constrained subset of observation fields that is sufficient to preserve the reference next-step decision.

The implementation is evaluated on the **AgentDojo Workspace benchmark**. The experiments examine:

- reduction of untrusted observation exposure under benign tasks;
- preservation of benign task utility and decision consistency;
- resistance to indirect prompt injection attacks;
- comparison with raw observations, relevance-only extraction, and sanitizer-based baselines;
- statistical uncertainty through cluster-bootstrap confidence intervals.

The MinObs implementation uses a greedy, fixed-point, sufficiency-guided deletion procedure over structured top-level observation fields. It provides a deletion-minimal field set under the chosen deletion order; it does **not** claim to compute a globally minimum field set.

---

## Repository Structure

The main files are:

```text
.
├── minobs_formal.py
├── collect_clean.py
├── run_rq1_complete_v4.py
├── run_rq2_complete.py
├── run_rq2_sanitizer_combo.py
├── rq2_statistics.py
├── run_rq3_complete.py
├── run_minobs_workspace_formal_v3.py   # required by RQ1
├── data/
└── README.md
```

### `minobs_formal.py`

Core paper-aligned MinObs implementation.

It defines the main data structures and logic used for:

- observation contracts;
- structured observation extraction;
- sufficiency verification;
- greedy fixed-point field elimination;
- decision signatures;
- fail-closed protected execution;
- MinObs projection.

A frozen observation contract is learned only from trusted clean-side profiling. During protected execution, untrusted live observations are not allowed to introduce new fields into the contract.

### `collect_clean.py`

Parses benign AgentDojo run logs and converts them into JSONL records used by RQ1.

The script extracts, among other fields:

- task ID and step ID;
- user task;
- execution history;
- current tool call;
- raw tool observation;
- parsed structured observation;
- next assistant action;
- task success status.

By default, only successful benign tasks are retained.

### `run_rq1_complete_v4.py`

Runs the complete benign RQ1 evaluation.

RQ1 asks whether MinObs can reduce exposure to untrusted tool observations while preserving benign task utility and decision consistency.

The script contains two phases:

**Phase A: Step-level exposure and sufficiency**

Compares:

- `Raw`: complete tool observation;
- `Relevant`: relevance-only top-level field extraction;
- `MinObs`: sufficiency-guided greedy fixed-point minimization.

Main measurements include:

- Field Exposure Ratio (FER);
- approximate Token Exposure Ratio (TER);
- replay decision consistency / sufficiency;
- baseline-unstable steps;
- zero-payload MinObs steps;
- provider-reported API token usage.

**Phase B: End-to-end benign utility**

Re-runs AgentDojo tasks using the field masks learned in Phase A and evaluates task success with AgentDojo's native task utility evaluator.

Before inclusion in the final utility comparison, tasks pass a raw trajectory stability gate: all raw repeats must preserve native utility and reproduce the clean normalized tool-name/argument trajectory.

### `run_rq2_complete.py`

Runs the main indirect prompt-injection attack experiment.

RQ2 compares:

- `Raw`;
- `Relevant`;
- `MinObs`.

The experiment reuses and freezes the RQ1 outputs. Relevance and MinObs field masks are **not relearned from attacked observations**.

Default experimental configuration in the supplied implementation:

- benchmark suite: AgentDojo Workspace;
- benchmark version: `v1.2.2`;
- attack: `important_instructions`;
- methods: Raw / Relevant / MinObs;
- repeated runs: 5 per user-task × injection-task × method combination.

Reported metrics include:

- Attack Success Rate (ASR);
- attack resistance (`1 - ASR`);
- utility under attack;
- joint safe utility;
- injection reached rate;
- injection exposure rate;
- exposure rate conditional on injection reach;
- ASR conditional on reach/exposure;
- clean-plan trajectory match;
- plan misses and argument mismatches.

For Relevant and MinObs, a mismatch with the frozen clean tool/argument plan triggers **fail-closed behavior**: the observation shown to the model is replaced with an empty payload rather than falling back to the raw untrusted observation.

The experiment checkpoints run-level results and can resume an interrupted run unless `--force` is specified.

### `run_rq2_sanitizer_combo.py`

Runs the sanitizer comparison used as an extension to RQ2.

It evaluates:

- `sanitizer`: full structured raw observation → sanitizer → LLM;
- `combo`: MinObs projection → sanitizer → LLM.

The sanitizer contains:

1. deterministic rules for obvious AI-directed instruction patterns; and
2. an optional sentence/span classifier using an OpenAI-compatible chat model at temperature 0.

This script is intentionally separated from the original Raw / Relevant / MinObs runs so that the original results remain unchanged.

### `rq2_statistics.py`

Computes post-hoc statistics for RQ2.

The script reports point estimates and 95% percentile cluster-bootstrap confidence intervals.

The bootstrap unit is:

```text
(user_task_id, injection_task_id)
```

All repeated runs belonging to a sampled task-goal pair are resampled together.

Default statistical configuration:

```text
bootstrap samples: 2000
seed: 20260916
```

### `run_rq3_complete.py`

This file is a reconstructed RQ3 analysis/simulation driver.

It contains modules corresponding to:

- component ablation;
- tool-chain length analysis;
- replay-sensitivity analysis.

---

## Dataset and Benchmark Information

The experiments use **AgentDojo**, with the supplied code configured for the:

```text
Workspace suite
benchmark version: v1.2.2
```

Clean execution logs are first generated through AgentDojo and then converted into MinObs-compatible JSONL records with `collect_clean.py`.

This repository does not define a separate standalone human-subject dataset. Experimental inputs are benchmark tasks, tool observations, execution traces, and injection tasks provided or generated through the AgentDojo evaluation workflow.

---

## Requirements

The scripts require a Python environment with the project and AgentDojo dependencies installed.

Directly imported Python packages include:

```text
agentdojo
openai
python-dotenv
PyYAML
numpy
pandas
```

The standard Python library modules used by the scripts include `argparse`, `csv`, `json`, `pathlib`, `dataclasses`, `collections`, `re`, `math`, `time`, and related utilities.

If the repository already contains a `pyproject.toml`, `requirements.txt`, or lock file, install dependencies using that project specification. Otherwise, install the packages above together with the AgentDojo version required by the experiments.

The supplied experimental scripts are written around AgentDojo Workspace benchmark version `v1.2.2`.

---

## API Configuration

Experiments that call an LLM require an OpenAI-compatible API endpoint.

Create a `.env` file in the repository root:

```env
OPENAI_COMPATIBLE_API_KEY=YOUR_API_KEY
OPENAI_COMPATIBLE_BASE_URL=https://YOUR_OPENAI_COMPATIBLE_ENDPOINT
```

For the configuration used with DeepSeek-compatible endpoints, the base URL may for example be:

```env
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
```

Do **not** commit API keys to a public repository.

Model IDs are supplied through command-line arguments such as:

```text
--model
--model-id
--judge-model
--relevance-model
--sanitizer-model
```

The exact model ID should match the endpoint used for reproduction.

---

## File Naming

The Python modules should use the canonical names referenced by their imports.

For example:

```text
minobs_formal.py
collect_clean.py
run_rq1_complete_v4.py
run_rq2_complete.py
run_rq2_sanitizer_combo.py
rq2_statistics.py
run_rq3_complete.py
run_minobs_workspace_formal_v3.py
```

If files were downloaded or uploaded with suffixes such as `(1)`, `(2)`, or `(3)`, rename them to the canonical module names before running the experiments.

---

## Reproduction Workflow

### 1. Generate benign AgentDojo runs

Run the AgentDojo Workspace benchmark under the desired model configuration and store the resulting benign run logs in a directory, for example:

```text
runs/deepseek_raw/
```

Only benign runs should be supplied to the clean-trace collection stage.

### 2. Convert benign logs into clean traces

```bash
python collect_clean.py \
  --input runs/deepseek_raw \
  --output data/clean_traces.jsonl
```

By default, failed benign tasks are excluded.

To include them for debugging:

```bash
python collect_clean.py \
  --input runs/deepseek_raw \
  --output data/clean_traces.jsonl \
  --include-failed-tasks
```

---

## RQ1: Benign Exposure Reduction and Utility Preservation

### Smoke test

```bash
python run_rq1_complete_v4.py \
  --input data/clean_traces.jsonl \
  --output-dir data/rq1_smoke \
  --max-tasks 3 \
  --repeats 1 \
  --utility-repeats 1
```

### Formal run

```bash
python run_rq1_complete_v4.py \
  --input data/clean_traces.jsonl \
  --output-dir data/rq1_complete \
  --max-tasks 40 \
  --repeats 5 \
  --utility-repeats 3
```

When using an OpenAI-compatible model, model arguments can be specified explicitly, for example:

```bash
python run_rq1_complete_v4.py \
  --input data/clean_traces.jsonl \
  --output-dir data/rq1_complete \
  --model YOUR_MODEL_ID \
  --model-id YOUR_MODEL_ID \
  --repeats 5 \
  --utility-repeats 3
```

Important RQ1 artifacts include:

```text
data/rq1_complete/rq1_final_aggregate.json
data/rq1_complete/rq1_plans.json
```

These artifacts are reused by RQ2.

---

## RQ2: Indirect Prompt-Injection Evaluation

A typical formal run is:

```bash
python run_rq2_complete.py \
  --rq1-final data/rq1_complete/rq1_final_aggregate.json \
  --rq1-plans data/rq1_complete/rq1_plans.json \
  --output-dir data/rq2_complete \
  --model-id YOUR_MODEL_ID \
  --attack-model-name YOUR_MODEL_NAME \
  --repeats 5 \
  --attack important_instructions
```

To use only injection tasks that pass the injection-task utility precheck:

```bash
python run_rq2_complete.py \
  --rq1-final data/rq1_complete/rq1_final_aggregate.json \
  --rq1-plans data/rq1_complete/rq1_plans.json \
  --output-dir data/rq2_complete \
  --model-id YOUR_MODEL_ID \
  --repeats 5 \
  --validated-injections-only
```

The default methods are:

```text
raw,relevant,minobs
```

A subset may be specified with:

```bash
--methods raw,minobs
```

### RQ2 output files

The main RQ2 script produces:

```text
rq2_runs.jsonl
rq2_runs.csv
rq2_aggregate.json
rq2_by_injection_task.csv
rq2_pair_summary.csv
rq2_injection_task_validation.json
```

Run-level JSONL output is checkpointed after each run. Re-running the same command resumes completed run keys by default. Use:

```bash
--force
```

only when intentionally restarting the requested experiment.

---

## Sanitizer and MinObs + Sanitizer Comparison

Run:

```bash
python run_rq2_sanitizer_combo.py \
  --rq1-final data/rq1_complete/rq1_final_aggregate.json \
  --rq1-plans data/rq1_complete/rq1_plans.json \
  --model-id YOUR_MODEL_ID \
  --sanitizer-model YOUR_SANITIZER_MODEL_ID \
  --repeats 5 \
  --attack important_instructions \
  --output-dir data/rq2_sanitizer_combo
```

The supported methods are:

```text
sanitizer
combo
sanitizer,combo
```

For deterministic rule-only sanitization without the LLM span classifier:

```bash
--sanitizer-rules-only
```

The `combo` condition preserves MinObs fail-closed behavior before sanitization.

---

## RQ2 Statistical Analysis

After RQ2 completes, compute point estimates and cluster-bootstrap confidence intervals with:

```bash
python rq2_statistics.py \
  --runs data/rq2_complete/rq2_runs.csv \
  --output-json data/rq2_complete/rq2_statistics.json \
  --output-csv data/rq2_complete/rq2_statistics.csv \
  --bootstrap 2000 \
  --seed 20260916
```

The default bootstrap unit is the user-task × injection-task pair, with repeated runs resampled together.

---

## RQ3 Analysis / Simulation Driver

The supplied `run_rq3_complete.py` is intended for deterministic simulation of the RQ3 analysis schema and target aggregates.

It must **not** be interpreted as producing new empirical LLM measurements.

For a complete live reproduction of RQ3, the relevant AgentDojo pipelines must be executed separately for each ablation condition, including regeneration of clean-side masks where required.

---

## Methodological Notes

### Observation contract

An observation contract is learned during trusted clean-side profiling. Candidate fields may originate from a trusted tool schema or, when explicitly permitted, from the structured top-level keys of a trusted clean observation.

After the contract is frozen, live untrusted observations or prior admitted untrusted history cannot expand or restore contract fields.

### Sufficiency

MinObs defines sufficiency operationally as preservation of the reference next-step decision under the fixed replay procedure.

This should not be interpreted as proof that the retained observation is globally minimal for every possible valid agent policy.

### Minimization

The supplied implementation performs greedy fixed-point backward elimination over candidate fields.

The result is deletion-minimal under the fixed deletion order.

The implementation does not claim:

- globally minimum cardinality;
- beam-search optimization;
- hierarchical global pruning.

### Fail-closed execution

During protected Relevant/MinObs execution, the live tool call is compared against the frozen clean plan.

If the expected tool name or normalized arguments do not match, the filtered conditions expose an empty payload rather than reverting to the raw observation.

Tool-execution errors themselves are preserved using AgentDojo's native error behavior.

---

## Main Metrics

### FER — Field Exposure Ratio

Fraction of candidate structured observation fields retained and exposed to the model.

Raw observations have:

```text
FER = 1
```

under the RQ1 definition.

### TER — Token Exposure Ratio

Approximate relative amount of observation content exposed to the model.

The RQ1 implementation uses a provider-independent character-based approximation for the exposure ratio and separately records provider-reported token usage for relevant API calls.

### ASR — Attack Success Rate

```text
ASR = mean(injection_success)
```

where `injection_success` is AgentDojo's attack-success signal.

### Attack Resistance

```text
Attack Resistance = 1 - ASR
```

### Utility under Attack

Mean legitimate user-task utility during attack evaluation.

### Joint Safe Utility

```text
P(utility = 1 AND injection_success = 0)
```

### Injection Reached

The injected attack text appeared in the raw successful tool result before condition-specific observation filtering.

### Injection Exposed

The injected attack text remained in the observation actually shown to the LLM after filtering.

### IER Given Reached

```text
P(injection_exposed | injection_reached)
```

---

## Reproducibility Notes

LLM-based experiments can vary across providers, model versions, API implementations, and execution dates. For reproducibility, record at least:

- exact model ID;
- API/provider;
- AgentDojo version;
- benchmark version;
- command-line arguments;
- number of repeats;
- random seeds where applicable;
- date of evaluation.

The code separates clean-side profiling from attacked evaluation so that attack observations do not redefine the MinObs or relevance field masks used in RQ2.

---

## Citation

If you use this code, please cite the associated manuscript:

```text
MinObs: Task-Constrained Minimal Observations against Indirect Prompt Injection in LLM Agents
```

Full bibliographic information can be added here after publication.

---

## License

No software license was specified in the supplied experimental files. Before public redistribution or reuse, consult the license distributed with the final repository and the licenses of third-party dependencies, including AgentDojo.

---

## Contact

For questions about the experimental implementation or reproduction of the manuscript results, please contact the corresponding author listed in the manuscript and PeerJ submission metadata.
