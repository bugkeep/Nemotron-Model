# Classify-Divide + Stable Trace + SFT

This is the working strategy for the repository.

## Why this path

The public Kaggle discussions suggest that the strongest open solutions do not rely on generic free-form chain-of-thought. They rely on:

- classifying prompts into a small number of task families
- using one stable trace template per family
- training with SFT on those deterministic traces

This repository now treats the competition as a mixture of task-specific programs instead of one broad reasoning task.

## Strategy by task type

### symbolic_transform

- Infer one local rewrite rule from the examples.
- Apply it left-to-right to the query string.
- Keep the trace mechanical and short.

### numeral

- Extract the numeral or integer first.
- Use one canonical conversion routine.
- Avoid multiple explanation styles for the same conversion.

### unit_conversion

- Read source quantity and target unit.
- Choose the factor explicitly.
- Perform one clean factor chain and stop.

### gravity

- Read the given mass, weight, constant, or planet multiplier.
- Apply one fixed formula path.
- Keep units consistent and avoid extra physics commentary.

### cipher

- Build the mapping from example pairs.
- Decode left-to-right.
- Preserve spaces and punctuation exactly.

### bit_manipulation

- Inspect example bit patterns first.
- Infer the rule at the output-bit level.
- Apply the same deterministic bit rule to the query.

This is the highest-value bucket and should get the most trace iteration effort.

### equation

- Read examples before solving the query.
- Infer the repeated arithmetic pattern.
- Reuse one canonical explanation style.

### cryptarithm

- Keep the explanation conservative and column-oriented.
- Prefer consistency over ambitious free-form reasoning.
- Treat this as a hard bucket and avoid noisy traces.

### other

- Use a short fallback procedure only when classification fails.
- Keep fallback examples limited so the model stays anchored to the main buckets.

## Repository outputs that support this strategy

When `python -m nemotron_model.data_bridge prepare ...` runs, it now writes:

- `train_sft.jsonl`
- `val_sft.jsonl`
- `dataset_summary.json`
- `task_strategy_report.json`
- `trace_preview.md`

The two new files are useful for iteration:

- `task_strategy_report.json`
  - task counts by split
  - strategy notes by task
- `trace_preview.md`
  - one prompt and one generated stable trace per task

## Iteration loop

1. Run `prepare` on the latest `train.csv`.
2. Inspect `task_strategy_report.json` to make sure bucket counts look sane.
3. Inspect `trace_preview.md` to see whether each trace is stable and task-specific.
4. Only then run SFT.
5. After evaluation, refine the weakest task bucket first instead of changing everything at once.
