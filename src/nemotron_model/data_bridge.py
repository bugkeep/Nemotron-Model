from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypeVar, TypedDict

LOGGER = logging.getLogger("tinker_nemotron_bridge")

SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. "
    "Use short deterministic numbered steps. "
    "Keep wording stable. "
    'End with a line that starts with "Final answer:". '
    "Use \\boxed{} only when the final answer does not contain braces or backslashes."
)

USER_SUFFIX = (
    "\nUse short deterministic numbered steps. "
    "End with Final answer:. "
    "Use \\boxed{} only when the final answer does not contain braces or backslashes."
)

T = TypeVar("T")


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class CompetitionExample:
    prompt: str
    answer: str
    task_type: str


@dataclass(frozen=True, slots=True)
class CotExample:
    prompt: str
    answer: str
    task_type: str
    assistant_text: str


@dataclass(frozen=True, slots=True)
class TaskStrategy:
    goal: str
    stable_trace_focus: list[str]
    sft_note: str


def clean_answer_text(answer: str) -> str:
    """Normalize an answer string from CSV inputs."""

    return str(answer).strip()


def needs_plain_final_line(answer: str) -> bool:
    """Return whether the answer should avoid ``\\boxed{}`` wrapping."""

    normalized = clean_answer_text(answer)
    return any(ch in normalized for ch in ["{", "}", "\\"]) or "\n" in normalized or "\r" in normalized


def format_final_answer(answer: str) -> str:
    """Format the metric-aligned final answer line used in the notebook."""

    normalized = clean_answer_text(answer)
    if needs_plain_final_line(normalized):
        return f"Final answer: {normalized}"
    return f"Final answer: \\boxed{{{normalized}}}"


def join_trace(lines: list[str]) -> str:
    """Join reasoning steps into a stable trace."""

    return "\n".join(lines)


def extract_roman_candidate(prompt: str) -> str | None:
    """Extract the strongest Roman numeral candidate from a prompt."""

    candidates = re.findall(r"\b[IVXLCDM]+\b", prompt)
    if not candidates:
        return None
    return max(candidates, key=len)


def extract_arabic_candidate(prompt: str) -> str | None:
    """Extract the final integer-looking token from a prompt."""

    numbers = re.findall(r"\b\d+\b", prompt)
    if not numbers:
        return None
    return numbers[-1]


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    """Match keywords using word boundaries to avoid ciphertext false positives."""

    for keyword in keywords:
        normalized = keyword.strip()
        if not normalized:
            continue
        pattern = r"\b" + re.escape(normalized) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def classify_task(prompt: str) -> str:
    """Heuristically bucket competition prompts into coarse task types."""

    lowered = prompt.lower()
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    arrow_or_equal_lines = [line for line in lines if "->" in line or "=" in line]

    special_count = sum(ch in r"[]{}()<>/\|&*#%^!@+-=\\" for ch in prompt)
    alpha_num_count = sum(ch.isalnum() for ch in prompt)

    if arrow_or_equal_lines:
        symbolic_matches = 0
        for line in arrow_or_equal_lines:
            parts = re.split(r"->|=", line)
            if len(parts) < 2:
                continue
            left = parts[0].strip()
            right = parts[1].strip()
            left_symbolic = sum(not ch.isalnum() and not ch.isspace() for ch in left)
            right_symbolic = sum(not ch.isalnum() and not ch.isspace() for ch in right)
            if left_symbolic + right_symbolic >= 3:
                symbolic_matches += 1
        if symbolic_matches >= 2:
            return "symbolic_transform"

    if (
        ("determine the result for" in lowered or "determine the output for" in lowered)
        and ("example" in lowered or "input -> output" in lowered or "below are a few examples" in lowered)
        and special_count > alpha_num_count * 0.15
    ):
        return "symbolic_transform"

    equation_like_lines = [line for line in lines if "=" in line]
    numeric_equation_lines = [
        line
        for line in equation_like_lines
        if any(ch.isdigit() for ch in line.split("=", maxsplit=1)[0])
    ]
    if (
        ("transformation rules" in lowered or "determine the result for" in lowered)
        and len(numeric_equation_lines) >= 2
    ):
        return "equation"

    if (
        "roman numeral" in lowered
        or "wonderland numeral system" in lowered
        or "numeral system" in lowered
        or "write the number" in lowered
        or "convert roman numeral" in lowered
    ):
        return "numeral"

    if contains_any_keyword(
        lowered,
        [
            "convert",
            "conversion",
            "unit",
            "units",
            "meter",
            "meters",
            "kilometer",
            "kilometers",
            "centimeter",
            "centimeters",
            "millimeter",
            "millimeters",
            "inch",
            "inches",
            "foot",
            "feet",
            "yard",
            "yards",
            "mile",
            "miles",
            "gram",
            "grams",
            "kilogram",
            "kilograms",
            "pound",
            "pounds",
            "ounce",
            "ounces",
            "liter",
            "liters",
            "milliliter",
            "milliliters",
            "celsius",
            "fahrenheit",
            "kelvin",
        ],
    ):
        return "unit_conversion"

    if contains_any_keyword(
        lowered,
        [
            "gravity",
            "gravitational",
            "planet",
            "weight on",
            "mass on",
            "mars",
            "jupiter",
            "venus",
            "saturn",
            "neptune",
            "uranus",
            "mercury",
            "moon",
        ],
    ):
        return "gravity"

    if contains_any_keyword(
        lowered,
        [
            "cipher",
            "decode",
            "encoded",
            "decoded",
            "encrypt",
            "decrypt",
            "substitution",
            "mapping",
            "code word",
        ],
    ):
        return "cipher"

    if contains_any_keyword(
        lowered,
        [
            "xor",
            "and ",
            " or ",
            "shift",
            "bit",
            "bits",
            "binary",
            "bitwise",
            "left shift",
            "right shift",
            "rotation",
            "rotate",
            "majority",
            "choice function",
        ],
    ):
        return "bit_manipulation"

    if contains_any_keyword(
        lowered,
        [
            "equation",
            "solve",
            "variable",
            "algebra",
            "unknown value",
            "find x",
            "find y",
        ],
    ):
        return "equation"

    if contains_any_keyword(
        lowered,
        [
            "cryptarithm",
            "alphametic",
            "letter-digit",
            "each letter",
            "assign a digit",
            "letters represent digits",
        ],
    ):
        return "cryptarithm"

    return "other"


TASK_STRATEGIES: dict[str, TaskStrategy] = {
    "symbolic_transform": TaskStrategy(
        goal="Infer a deterministic local symbol rewrite rule from the examples.",
        stable_trace_focus=[
            "Read input-output examples first.",
            "Describe the symbol rewrite rule in fixed wording.",
            "Apply the rule left-to-right to the query string.",
        ],
        sft_note="Keep the trace short and mechanical so the model learns the rule, not stylistic variation.",
    ),
    "numeral": TaskStrategy(
        goal="Convert between Roman-like numerals and integers with a fixed scan procedure.",
        stable_trace_focus=[
            "Extract the numeral or integer from the prompt.",
            "Use the same add/subtract scan logic every time.",
            "State only the final converted value.",
        ],
        sft_note="Prefer one canonical conversion explanation to avoid format drift.",
    ),
    "unit_conversion": TaskStrategy(
        goal="Apply unit factors step by step while keeping units explicit.",
        stable_trace_focus=[
            "Identify source quantity and target unit.",
            "Select the conversion factor explicitly.",
            "Apply one arithmetic chain and finish.",
        ],
        sft_note="This category benefits from highly repeatable arithmetic phrasing.",
    ),
    "gravity": TaskStrategy(
        goal="Use the requested physical relation or planet multiplier in a fixed formula flow.",
        stable_trace_focus=[
            "Read the given mass/weight or constant.",
            "Substitute into the required formula.",
            "Compute once and report the result with units when needed.",
        ],
        sft_note="Avoid extra physics commentary; keep the derivation formula-first.",
    ),
    "cipher": TaskStrategy(
        goal="Recover a consistent character or token mapping from examples, then decode left-to-right.",
        stable_trace_focus=[
            "Build the mapping from examples.",
            "Preserve spaces and punctuation.",
            "Decode in one deterministic pass.",
        ],
        sft_note="The trace should emphasize exact mapping reconstruction, not creative interpretation.",
    ),
    "bit_manipulation": TaskStrategy(
        goal="Infer a deterministic bit rule from examples and apply it per output position.",
        stable_trace_focus=[
            "Inspect example input-output bit patterns.",
            "Reason about output bits position by position.",
            "Apply the same bit rule to the query input.",
        ],
        sft_note="This is the main score differentiator, so trace wording should stay extremely stable.",
    ),
    "equation": TaskStrategy(
        goal="Infer the algebraic or arithmetic transformation pattern used by the examples.",
        stable_trace_focus=[
            "Read examples before touching the query.",
            "Identify the repeated operation order.",
            "Apply the same operation chain to the query.",
        ],
        sft_note="Use a single canonical explanation for the inferred arithmetic pattern.",
    ),
    "cryptarithm": TaskStrategy(
        goal="Use a fixed assignment or column-consistency story instead of open-ended reasoning.",
        stable_trace_focus=[
            "Read the symbolic arithmetic examples carefully.",
            "Infer the consistent assignment pattern.",
            "Apply that pattern once to the query expression.",
        ],
        sft_note="Coverage is usually lower here, so keep traces conservative and deterministic.",
    ),
    "other": TaskStrategy(
        goal="Fallback deterministic reasoning when the prompt does not match a known bucket.",
        stable_trace_focus=[
            "State the detected category.",
            "Use a short deterministic procedure.",
            "Return the final value without extra variation.",
        ],
        sft_note="Fallback traces should be rare and stable.",
    ),
}


def build_symbolic_transform_trace(example: CompetitionExample) -> str:
    """Build a short trace for symbolic transformation tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Read the example input-output pairs carefully.",
            "Step 2: Track how each symbol or local symbol pattern changes from input to output.",
            "Step 3: Identify the deterministic transformation rule that is consistent with the examples.",
            "Step 4: Apply the same symbol transformation to the query string from left to right.",
            f"Step 5: The transformed string is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_numeral_trace(example: CompetitionExample) -> str:
    """Build a trace for numeral conversion tasks."""

    answer = clean_answer_text(example.answer)
    roman = extract_roman_candidate(example.prompt)
    arabic = extract_arabic_candidate(example.prompt)
    lowered = example.prompt.lower()

    if (
        "write the number" in lowered
        or "numeral system" in lowered
        or "convert the number" in lowered
        or re.search(r"now,\s*write", lowered)
    ):
        return join_trace(
            [
                "Step 1: Identify the target integer in the prompt.",
                f"Integer: {arabic}" if arabic else "Integer: [read from prompt]",
                "Step 2: Convert the integer into the target numeral system using the observed examples.",
                "Step 3: Build the numeral from largest place value to smallest place value.",
                f"Step 4: The numeral representation is {answer}.",
                format_final_answer(answer),
            ]
        )

    return join_trace(
        [
            "Step 1: Identify the Roman numeral in the prompt.",
            f"Roman numeral: {roman}" if roman else "Roman numeral: [read from prompt]",
            "Step 2: Use the values I=1, V=5, X=10, L=50, C=100, D=500, M=1000.",
            "Step 3: Scan from left to right. If a symbol is smaller than the next symbol, subtract it. Otherwise, add it.",
            f"Step 4: The computed integer is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_unit_conversion_trace(example: CompetitionExample) -> str:
    """Build a trace for unit-conversion tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Read the given quantity and the target unit.",
            "Step 2: Choose the conversion factor that matches the source unit and the target unit.",
            "Step 3: Apply multiplication or division one step at a time, while keeping the units consistent.",
            "Step 4: Simplify the expression until only the requested unit remains.",
            f"Step 5: The converted value is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_gravity_trace(example: CompetitionExample) -> str:
    """Build a trace for gravity or planet-weight tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Identify the physical quantity being asked for.",
            "Step 2: Read the gravity constant or the planet-specific multiplier from the prompt.",
            "Step 3: Substitute the known value into the required formula.",
            "Step 4: Perform the arithmetic carefully and keep the units consistent.",
            f"Step 5: The resulting value is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_cipher_trace(example: CompetitionExample) -> str:
    """Build a trace for cipher or decoding tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Read the example input-output pairs.",
            "Step 2: Build the current character mapping from the examples.",
            "Step 3: Infer any remaining unmapped characters consistently.",
            "Step 4: Decode the target text from left to right, preserving spaces and punctuation.",
            f"Step 5: The decoded text is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_bit_manipulation_trace(example: CompetitionExample) -> str:
    """Build a trace for binary or bitwise tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Read all example binary input-output pairs.",
            "Step 2: Compare each output bit with candidate operations on the input bits.",
            "Step 3: Select the single deterministic bit rule that matches all examples.",
            "Step 4: Apply the same rule to the query input to produce the final binary output.",
            f"Step 5: The resulting binary output is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_equation_trace(example: CompetitionExample) -> str:
    """Build a trace for algebra-like equation tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Read the example equations carefully.",
            "Step 2: Check whether the operands or the result are reversed before applying the rule.",
            "Step 3: Test the candidate arithmetic transformation that is consistent with the examples.",
            "Step 4: Apply the same transformation to the query equation.",
            f"Step 5: The resulting value is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_cryptarithm_trace(example: CompetitionExample) -> str:
    """Build a trace for symbolic arithmetic tasks."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            "Step 1: Read the symbolic examples carefully.",
            "Step 2: Infer the operation pattern that is consistent with the examples.",
            "Step 3: Apply the same pattern to the query expression.",
            f"Step 4: The resulting value is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_default_trace(example: CompetitionExample) -> str:
    """Build a generic deterministic trace."""

    answer = clean_answer_text(example.answer)
    return join_trace(
        [
            f"Step 1: Task category is {example.task_type}.",
            "Step 2: Follow a deterministic step-by-step procedure.",
            "Step 3: Keep the wording stable and avoid unnecessary variation.",
            f"Step 4: The result is {answer}.",
            format_final_answer(answer),
        ]
    )


def build_trace(example: CompetitionExample) -> str:
    """Route each competition example to a stable reasoning template."""

    builders: dict[str, Callable[[CompetitionExample], str]] = {
        "numeral": build_numeral_trace,
        "unit_conversion": build_unit_conversion_trace,
        "gravity": build_gravity_trace,
        "cipher": build_cipher_trace,
        "symbolic_transform": build_symbolic_transform_trace,
        "bit_manipulation": build_bit_manipulation_trace,
        "equation": build_equation_trace,
        "cryptarithm": build_cryptarithm_trace,
    }
    return builders.get(example.task_type, build_default_trace)(example)


def build_messages(prompt: str, assistant_text: str) -> list[ChatMessage]:
    """Convert a prompt and answer trace into Tinker-ready chat messages."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + USER_SUFFIX},
        {"role": "assistant", "content": assistant_text},
    ]


def normalize_external_cot(text: str) -> str:
    """Strip final-answer wrappers from external CoT rows."""

    normalized = str(text).strip()
    normalized = re.sub(r"Final answer\s*[:：].*$", "", normalized, flags=re.IGNORECASE | re.DOTALL).rstrip()
    normalized = re.sub(r"\\boxed\{[^}]*\}", "", normalized).rstrip()
    return normalized


def build_assistant_text_from_cot(cot: str, answer: str) -> str:
    """Append the metric-aligned answer line to an external chain of thought."""

    return normalize_external_cot(cot) + "\n" + format_final_answer(answer)


def read_competition_rows(path: Path) -> list[CompetitionExample]:
    """Load Kaggle competition CSV rows from disk."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"prompt", "answer"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required_columns)}")

        rows: list[CompetitionExample] = []
        for row in reader:
            prompt = row["prompt"]
            answer = row["answer"]
            rows.append(
                CompetitionExample(
                    prompt=prompt,
                    answer=answer,
                    task_type=classify_task(prompt),
                )
            )
    if not rows:
        raise ValueError(f"{path} did not contain any usable rows")
    return rows


def read_optional_cot_rows(paths: list[Path]) -> list[CotExample]:
    """Load optional matched CoT files in a competition-compatible format."""

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"prompt", "answer"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                continue

            cot_column = next(
                (name for name in ("generated_cot", "cot", "reasoning") if name in reader.fieldnames),
                None,
            )
            if cot_column is None:
                continue

            rows: list[CotExample] = []
            for row in reader:
                cot_value = str(row.get(cot_column, "")).strip()
                if not cot_value:
                    continue
                prompt = row["prompt"]
                answer = row["answer"]
                task_type = row.get("task_type") or classify_task(prompt)
                rows.append(
                    CotExample(
                        prompt=prompt,
                        answer=answer,
                        task_type=task_type,
                        assistant_text=build_assistant_text_from_cot(cot_value, answer),
                    )
                )
        if rows:
            LOGGER.info("Loaded %s external CoT rows from %s", len(rows), path)
            return rows

    LOGGER.info("No optional external CoT rows found")
    return []


def stratified_split_by_task(
    rows: list[CompetitionExample],
    *,
    seed: int,
    val_fraction: float,
    val_min_size_per_task: int,
) -> tuple[list[CompetitionExample], list[CompetitionExample]]:
    """Split rows into train and validation sets while preserving task buckets."""

    grouped: dict[str, list[CompetitionExample]] = {}
    for row in rows:
        grouped.setdefault(row.task_type, []).append(row)

    train_rows: list[CompetitionExample] = []
    val_rows: list[CompetitionExample] = []

    for group_rows in grouped.values():
        shuffled = list(group_rows)
        random.Random(seed).shuffle(shuffled)

        if len(shuffled) <= 4:
            train_rows.extend(shuffled)
            continue

        val_count = max(val_min_size_per_task, int(round(len(shuffled) * val_fraction)))
        val_count = min(val_count, max(1, len(shuffled) // 5))
        val_count = min(val_count, len(shuffled) - 2)

        train_rows.extend(shuffled[:-val_count])
        val_rows.extend(shuffled[-val_count:])

    return train_rows, val_rows


def balance_by_task(
    rows: list[T],
    *,
    target_per_task: int,
    seed: int,
    task_getter: Callable[[T], str],
) -> list[T]:
    """Up-sample or down-sample each task bucket to a common target size."""

    grouped: dict[str, list[T]] = {}
    for row in rows:
        grouped.setdefault(task_getter(row), []).append(row)

    balanced: list[T] = []
    rng = random.Random(seed)
    for group_rows in grouped.values():
        if len(group_rows) >= target_per_task:
            balanced.extend(rng.sample(group_rows, target_per_task))
            continue

        expanded = list(group_rows)
        while len(expanded) < target_per_task:
            expanded.append(rng.choice(group_rows))
        balanced.extend(expanded)

    rng.shuffle(balanced)
    return balanced


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write JSONL conversation records to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_rows_by_task(rows: list[CompetitionExample]) -> dict[str, int]:
    """Count examples by task type in a stable key order."""

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.task_type] = counts.get(row.task_type, 0) + 1
    return dict(sorted(counts.items()))


def count_records_by_task(records: list[dict[str, object]]) -> dict[str, int]:
    """Count prepared records by task type."""

    counts: dict[str, int] = {}
    for record in records:
        task_type = str(record["task_type"])
        counts[task_type] = counts.get(task_type, 0) + 1
    return dict(sorted(counts.items()))


def write_trace_preview(path: Path, rows: list[CompetitionExample]) -> None:
    """Write one sample trace preview per task bucket."""

    samples: dict[str, CompetitionExample] = {}
    for row in rows:
        samples.setdefault(row.task_type, row)

    lines = [
        "# Trace Preview",
        "",
        "This file shows one deterministic trace template per detected task type.",
        "",
    ]

    for task_type in sorted(samples):
        sample = samples[task_type]
        strategy = TASK_STRATEGIES.get(task_type, TASK_STRATEGIES["other"])
        trace = build_trace(sample)
        lines.extend(
            [
                f"## {task_type}",
                "",
                f"- Goal: {strategy.goal}",
                f"- SFT note: {strategy.sft_note}",
                "- Stable trace focus:",
            ]
        )
        lines.extend(f"  - {item}" for item in strategy.stable_trace_focus)
        lines.extend(
            [
                "",
                "### Prompt",
                "",
                "```text",
                sample.prompt,
                "```",
                "",
                "### Answer",
                "",
                "```text",
                clean_answer_text(sample.answer),
                "```",
                "",
                "### Trace",
                "",
                "```text",
                trace,
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_datasets(args: argparse.Namespace) -> tuple[Path, Path]:
    """Prepare train and validation JSONL files for Tinker SFT."""

    train_csv = Path(args.train_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    optional_cot_paths = [Path(item).expanduser().resolve() for item in args.optional_cot]

    competition_rows = read_competition_rows(train_csv)
    train_part, val_part = stratified_split_by_task(
        competition_rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_min_size_per_task=args.val_min_size_per_task,
    )
    train_balanced = balance_by_task(
        train_part,
        target_per_task=args.target_samples_per_task,
        seed=args.seed,
        task_getter=lambda row: row.task_type,
    )

    train_records = [
        {
            "messages": build_messages(row.prompt, build_trace(row)),
            "task_type": row.task_type,
            "source": "templated",
        }
        for row in train_balanced
    ]
    val_records = [
        {
            "messages": build_messages(row.prompt, build_trace(row)),
            "task_type": row.task_type,
            "source": "templated_val",
        }
        for row in val_part
    ]

    optional_cot_rows = read_optional_cot_rows(optional_cot_paths)
    if optional_cot_rows:
        cot_target = min(1200, args.target_samples_per_task)
        cot_balanced = balance_by_task(
            optional_cot_rows,
            target_per_task=cot_target,
            seed=args.seed + 7,
            task_getter=lambda row: row.task_type,
        )
        train_records.extend(
            {
                "messages": build_messages(row.prompt, row.assistant_text),
                "task_type": row.task_type,
                "source": "external_cot",
            }
            for row in cot_balanced
        )
        random.Random(args.seed).shuffle(train_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_dir / "train_sft.jsonl"
    val_jsonl = output_dir / "val_sft.jsonl"
    summary_json = output_dir / "dataset_summary.json"
    strategy_json = output_dir / "task_strategy_report.json"
    trace_preview = output_dir / "trace_preview.md"

    write_jsonl(train_jsonl, train_records)
    write_jsonl(val_jsonl, val_records)

    strategy_report = {
        "strategy": "classify_divide_trace_sft",
        "task_counts": {
            "raw": count_rows_by_task(competition_rows),
            "train_split": count_rows_by_task(train_part),
            "validation": count_rows_by_task(val_part),
            "balanced_train": count_rows_by_task(train_balanced),
            "final_train": count_records_by_task(train_records),
        },
        "task_strategies": {
            task_type: {
                "goal": strategy.goal,
                "stable_trace_focus": strategy.stable_trace_focus,
                "sft_note": strategy.sft_note,
            }
            for task_type, strategy in sorted(TASK_STRATEGIES.items())
        },
    }

    summary_json.write_text(
        json.dumps(
            {
                "raw_train_count": len(competition_rows),
                "train_split_count": len(train_part),
                "balanced_train_count": len(train_balanced),
                "validation_count": len(val_part),
                "optional_cot_count": len(optional_cot_rows),
                "final_train_count": len(train_records),
                "task_counts": strategy_report["task_counts"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    strategy_json.write_text(json.dumps(strategy_report, indent=2) + "\n", encoding="utf-8")
    write_trace_preview(trace_preview, competition_rows)

    LOGGER.info("Prepared %s", train_jsonl)
    LOGGER.info("Prepared %s", val_jsonl)
    LOGGER.info("Prepared %s", summary_json)
    LOGGER.info("Prepared %s", strategy_json)
    LOGGER.info("Prepared %s", trace_preview)
    return train_jsonl, val_jsonl


def compute_learning_rate(base_lr: float, *, schedule: str, global_step: int, total_steps: int) -> float:
    """Compute a simple scalar learning-rate schedule."""

    if schedule == "constant" or total_steps <= 0:
        return base_lr
    if schedule == "linear":
        return base_lr * max(0.0, 1.0 - (global_step / total_steps))
    raise ValueError(f"Unsupported schedule: {schedule}")


async def train_with_tinker(args: argparse.Namespace) -> str:
    """Launch a simple Tinker LoRA SFT run and return the final tinker path."""

    import tinker
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    if not args.tinker_api_key and "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("TINKER_API_KEY is not set. Export it or pass --tinker-api-key.")

    if args.tinker_api_key:
        os.environ["TINKER_API_KEY"] = args.tinker_api_key

    train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    builder = FromConversationFileBuilder(
        file_path=str(train_jsonl),
        common_config=common_config,
        test_size=0,
        shuffle_seed=args.seed,
    )
    train_dataset, _ = builder()

    service_client = tinker.ServiceClient(base_url=args.base_url or None)
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.model_name,
        rank=args.lora_rank,
        train_unembed=args.train_unembed,
    )

    total_steps = len(train_dataset) * args.epochs
    global_step = 0

    for epoch in range(args.epochs):
        train_dataset.set_epoch(seed=args.seed + epoch)
        for batch_index in range(len(train_dataset)):
            batch = train_dataset.get_batch(batch_index)
            learning_rate = compute_learning_rate(
                args.learning_rate,
                schedule=args.lr_schedule,
                global_step=global_step,
                total_steps=total_steps,
            )

            forward_backward = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
            await forward_backward.result_async()
            optimizer_step = await training_client.optim_step_async({"learning_rate": learning_rate})
            await optimizer_step.result_async()

            global_step += 1
            if global_step % args.log_every == 0:
                LOGGER.info(
                    "epoch=%s step=%s/%s lr=%.7f",
                    epoch + 1,
                    global_step,
                    total_steps,
                    learning_rate,
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                checkpoint_name = f"{args.checkpoint_prefix}-step{global_step:05d}"
                checkpoint = await training_client.save_weights_for_sampler_async(checkpoint_name)
                checkpoint_result = await checkpoint.result_async()
                LOGGER.info("Saved interim sampler weights: %s", checkpoint_result.path)

    final_weights = await training_client.save_weights_for_sampler_async(args.checkpoint_name)
    final_result = await final_weights.result_async()
    LOGGER.info("Final sampler weights: %s", final_result.path)

    if args.tinker_path_out:
        output_path = Path(args.tinker_path_out).expanduser().resolve()
        output_path.write_text(final_result.path + "\n", encoding="utf-8")
        LOGGER.info("Wrote final tinker path to %s", output_path)

    return final_result.path


def export_adapter(args: argparse.Namespace) -> Path:
    """Download Tinker weights and export a Kaggle-friendly PEFT adapter."""

    from tinker_cookbook.weights import build_lora_adapter, download, publish_to_hf_hub

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    download_dir = Path(args.download_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(download(tinker_path=args.tinker_path, output_dir=str(download_dir)))
    build_lora_adapter(
        base_model=args.model_name,
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
        trust_remote_code=args.trust_remote_code,
    )

    if args.hf_repo_id:
        publish_to_hf_hub(model_path=str(output_dir), repo_id=args.hf_repo_id)
        LOGGER.info("Published adapter to %s", args.hf_repo_id)

    LOGGER.info("Exported PEFT adapter to %s", output_dir)
    return output_dir


def main() -> int:
    """Parse arguments and run the selected subcommand."""

    parser = argparse.ArgumentParser(
        description="Bridge Kaggle Nemotron reasoning data into a Tinker training and export workflow."
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build Tinker JSONL datasets from train.csv")
    prepare_parser.add_argument("--train-csv", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--optional-cot", nargs="*", default=[])
    prepare_parser.add_argument("--target-samples-per-task", type=int, default=1200)
    prepare_parser.add_argument("--val-fraction", type=float, default=0.05)
    prepare_parser.add_argument("--val-min-size-per-task", type=int, default=2)
    prepare_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Run a simple Tinker LoRA SFT job")
    train_parser.add_argument("--train-jsonl", required=True)
    train_parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    train_parser.add_argument("--renderer-name", default="nemotron3")
    train_parser.add_argument("--max-length", type=int, default=4096)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--lora-rank", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear")
    train_parser.add_argument("--train-unembed", action="store_true")
    train_parser.add_argument("--save-every", type=int, default=0)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--checkpoint-prefix", default="nemotron-kaggle")
    train_parser.add_argument("--checkpoint-name", default="nemotron-kaggle-final")
    train_parser.add_argument("--base-url", default="")
    train_parser.add_argument("--tinker-api-key", default="")
    train_parser.add_argument("--tinker-path-out", default="")
    train_parser.add_argument("--seed", type=int, default=42)

    export_parser = subparsers.add_parser("export", help="Download Tinker weights and export a PEFT adapter")
    export_parser.add_argument("--tinker-path", required=True)
    export_parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    export_parser.add_argument("--download-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--hf-repo-id", default="")
    export_parser.add_argument("--hf-token", default="")
    export_parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    export_parser.set_defaults(trust_remote_code=True)

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "prepare":
        prepare_datasets(args)
        return 0
    if args.command == "train":
        asyncio.run(train_with_tinker(args))
        return 0
    if args.command == "export":
        export_adapter(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
