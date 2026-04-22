from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nemotron_model.data_bridge import (
    CompetitionExample,
    balance_by_task,
    build_messages,
    build_trace,
    classify_task,
    format_final_answer,
    needs_plain_final_line,
    prepare_datasets,
    stratified_split_by_task,
)


def test_final_answer_formatting_boxes_simple_answers() -> None:
    assert needs_plain_final_line("42") is False
    assert format_final_answer("42") == "Final answer: \\boxed{42}"


def test_final_answer_formatting_avoids_box_for_special_tokens() -> None:
    assert needs_plain_final_line(r"a\{b") is True
    assert format_final_answer(r"a\{b") == r"Final answer: a\{b"


def test_classify_task_detects_common_buckets() -> None:
    assert classify_task("Convert 15 kilometers to meters.") == "unit_conversion"
    assert classify_task("Apply XOR to the binary bits and return the output.") == "bit_manipulation"
    assert classify_task("Convert Roman numeral XIV into an integer.") == "numeral"


def test_classify_task_detects_equation_bucket_from_numeric_operator_examples() -> None:
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        "34/44 = 1\n"
        "41/32 = 9\n"
        "34|25 = 69\n"
        "87\\64 = 8853\n"
        "Now, determine the result for: 69/52"
    )
    assert classify_task(prompt) == "equation"


def test_classify_task_avoids_cipher_false_positive_from_embedded_substrings() -> None:
    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        "fdcfiu qpgjqeuy tulnev zgigxu -> turtle imagines beyond palace\n"
        "oqbgcv xagyuy fau xiuwuc fcugydcu -> wizard chases the clever treasure\n"
        "pndyu yuuy gcndev wgiiul -> mouse sees around valley\n"
        "fau xninchdi fugxauc qpgjqeuy -> the colorful teacher imagines\n"
        "fau gexquef kqej ogfxauy -> the ancient king watches\n"
        "Now, decrypt the following text: kqej qpgjqeuy gtnwu pndefgqe"
    )
    assert classify_task(prompt) == "cipher"


def test_classify_task_avoids_false_roman_detection_inside_ciphertext() -> None:
    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        "hmxad apdhvdq vid ohexahm apwqvhm -> alice creates the magical crystal\n"
        "zxuhpl zhvaidq xyqxld txmmhed -> wizard watches inside village\n"
        "nfddy xohexydq xy ehpldy -> queen imagines in garden\n"
        "osfqd qddq lssp -> mouse sees door\n"
        "vid amdtdp zxuhpl dgjmspdq -> the clever wizard explores\n"
        "Now, decrypt the following text: bxye aihqdq ahqvmd"
    )
    assert classify_task(prompt) == "cipher"


def test_build_trace_keeps_metric_aligned_final_line() -> None:
    example = CompetitionExample(
        prompt="Apply XOR to the binary bits and return the output.",
        answer="0101",
        task_type="bit_manipulation",
    )
    trace = build_trace(example)
    assert "Step 1:" in trace
    assert trace.endswith("Final answer: \\boxed{0101}")


def test_build_messages_wraps_system_user_and_assistant() -> None:
    messages = build_messages("prompt body", "assistant body")
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"].startswith("prompt body")


def test_stratified_split_and_balance_keep_task_coverage() -> None:
    rows = [
        CompetitionExample(prompt=f"prompt-{index}", answer=str(index), task_type="task_a")
        for index in range(6)
    ] + [
        CompetitionExample(prompt=f"other-{index}", answer=str(index), task_type="task_b")
        for index in range(6)
    ]

    train_rows, val_rows = stratified_split_by_task(
        rows,
        seed=42,
        val_fraction=0.2,
        val_min_size_per_task=1,
    )
    assert {row.task_type for row in train_rows} == {"task_a", "task_b"}
    assert {row.task_type for row in val_rows} == {"task_a", "task_b"}

    balanced = balance_by_task(
        train_rows,
        target_per_task=4,
        seed=42,
        task_getter=lambda row: row.task_type,
    )
    counts = {
        task_type: sum(1 for row in balanced if row.task_type == task_type)
        for task_type in {"task_a", "task_b"}
    }
    assert counts == {"task_a": 4, "task_b": 4}


def test_prepare_datasets_writes_jsonl_and_summary(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "\n".join(
            [
                "id,prompt,answer",
                '1,"Convert 15 kilometers to meters.","15000"',
                '2,"Convert 20 kilometers to meters.","20000"',
                '3,"Convert Roman numeral XIV into an integer.","14"',
                '4,"Convert Roman numeral IX into an integer.","9"',
                '5,"Apply XOR to the binary bits and return the output.","0101"',
                '6,"Apply XOR to the binary bits and return the output.","1100"',
                '7,"Decode the cipher word using the mapping.","cat"',
                '8,"Decode the cipher word using the mapping.","dog"',
                '9,"Solve the equation x + 1 = 3.","2"',
                '10,"Solve the equation x + 2 = 5.","3"',
            ]
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "prepared"),
        optional_cot=[],
        target_samples_per_task=2,
        val_fraction=0.2,
        val_min_size_per_task=1,
        seed=42,
    )

    train_jsonl, val_jsonl = prepare_datasets(args)
    summary_path = Path(args.output_dir) / "dataset_summary.json"
    strategy_path = Path(args.output_dir) / "task_strategy_report.json"
    trace_preview_path = Path(args.output_dir) / "trace_preview.md"

    train_lines = train_jsonl.read_text(encoding="utf-8").strip().splitlines()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    trace_preview = trace_preview_path.read_text(encoding="utf-8")

    assert train_lines
    assert val_jsonl.exists()
    assert strategy_path.exists()
    assert trace_preview_path.exists()
    first_record = json.loads(train_lines[0])
    assert "messages" in first_record
    assert first_record["messages"][0]["role"] == "system"
    assert summary["raw_train_count"] == 10
    assert summary["task_counts"]["raw"]["bit_manipulation"] == 2
    assert strategy["strategy"] == "classify_divide_trace_sft"
    assert "bit_manipulation" in strategy["task_strategies"]
    assert "## bit_manipulation" in trace_preview
