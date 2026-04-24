from __future__ import annotations

import argparse
import json
import pathlib
import textwrap
from pprint import pformat


VARIANT_PRESETS: dict[str, dict[str, object]] = {
    "baseline_hc": {
        "title": "Tong Baseline HC",
        "note": "Baseline Tong continuation with the current small high-confidence solved subset.",
        "sft_max_len": 4096,
        "sft_epochs": 2,
        "sft_grad_accum": 16,
        "sft_lr": "2e-5",
        "small_subtype_overrides": {},
        "tong_style_overrides": {},
        "priority_duplication_factor_small": 1,
    },
    "bit_eq_boost": {
        "title": "Tong BitEq Boost",
        "note": "Boost high-confidence bit-manipulation and equation subtypes while keeping Tong continuation.",
        "sft_max_len": 4096,
        "sft_epochs": 2,
        "sft_grad_accum": 16,
        "sft_lr": "2e-5",
        "small_subtype_overrides": {
            ("bit_manipulation", "pairwise_parity_conjunctive_mix"): 96,
            ("bit_manipulation", "pairwise_conjunctive_family"): 64,
            ("bit_manipulation", "pairwise_disjunctive_family"): 64,
            ("equation", "digitwise_compose"): 64,
            ("equation", "query_filtered_whole_number_evaluate"): 32,
            ("equation", "query_filtered_scalar_reduce"): 32,
            ("equation", "query_filtered_digitwise_compose"): 32,
        },
        "tong_style_overrides": {
            "bit_manipulation_including_wrong.csv": 192,
            "bit_manipulation_synth_including_wrong_v2.csv": 96,
            "equation_numeric.csv": 144,
        },
        "priority_duplication_factor_small": 1,
    },
    "ultra_low_drift": {
        "title": "Tong Ultra Low Drift",
        "note": "More conservative continuation with lower learning rate and fewer epochs to preserve Tong behavior.",
        "sft_max_len": 4096,
        "sft_epochs": 1,
        "sft_grad_accum": 16,
        "sft_lr": "1e-5",
        "small_subtype_overrides": {
            ("bit_manipulation", "pairwise_parity_conjunctive_mix"): 48,
            ("equation", "digitwise_compose"): 32,
        },
        "tong_style_overrides": {
            "bit_manipulation_including_wrong.csv": 96,
            "equation_numeric.csv": 80,
        },
        "priority_duplication_factor_small": 1,
    },
    "priority_weighted": {
        "title": "Tong Priority Weighted",
        "note": "Keep the solved subset small, but duplicate Tong priority rows even in small mode.",
        "sft_max_len": 4096,
        "sft_epochs": 2,
        "sft_grad_accum": 16,
        "sft_lr": "2e-5",
        "small_subtype_overrides": {},
        "tong_style_overrides": {
            "bit_manipulation_including_wrong.csv": 160,
            "equation_numeric.csv": 128,
        },
        "priority_duplication_factor_small": 2,
    },
    "query_focus": {
        "title": "Tong Query Focus",
        "note": "Bias continuation toward query-operator-filtered equation rules and local boolean mixes.",
        "sft_max_len": 4096,
        "sft_epochs": 2,
        "sft_grad_accum": 16,
        "sft_lr": "2e-5",
        "small_subtype_overrides": {
            ("bit_manipulation", "pairwise_parity_conjunctive_mix"): 96,
            ("equation", "query_filtered_whole_number_evaluate"): 48,
            ("equation", "query_filtered_scalar_reduce"): 48,
            ("equation", "query_filtered_digitwise_compose"): 48,
        },
        "tong_style_overrides": {
            "bit_manipulation_including_wrong.csv": 176,
            "equation_numeric.csv": 160,
        },
        "priority_duplication_factor_small": 1,
    },
    "cryptarithm_probe": {
        "title": "Tong Cryptarithm Probe",
        "note": "Add a tiny cryptarithm slice on top of the low-drift Tong continuation to test headroom.",
        "sft_max_len": 4096,
        "sft_epochs": 1,
        "sft_grad_accum": 16,
        "sft_lr": "1.5e-5",
        "small_subtype_overrides": {},
        "tong_style_overrides": {
            "equation_numeric.csv": 112,
            "cryptarithm.csv": 24,
        },
        "priority_duplication_factor_small": 1,
    },
}


def to_source(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in text.splitlines()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite the original Kaggle notebook into the improved version.")
    parser.add_argument("--src", required=True, help="Source ipynb path.")
    parser.add_argument("--dst", required=True, help="Destination ipynb path.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_PRESETS),
        default="baseline_hc",
        help="Experiment preset to bake into the generated notebook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = pathlib.Path(args.src).expanduser().resolve()
    dst = pathlib.Path(args.dst).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    variant = VARIANT_PRESETS[args.variant]

    nb = json.loads(src.read_text(encoding="utf-8"))

    def set_source(idx: int, text: str) -> None:
        nb["cells"][idx]["source"] = to_source(text)

    set_source(
        0,
        f"""
        ## 1. Offline Package Installation

        > Variant: {variant["title"]}. {variant["note"]}
        """,
    )

    set_source(
        1,
        r"""
        # ─── GPU preflight + install ONLY what is missing from the Kaggle Docker image ─────────
        # torch, transformers, peft, triton, bitsandbytes, kagglehub are PRE-INSTALLED
        # in dockerImageVersionId 31287. DO NOT reinstall them — they contain
        # Blackwell-specific patches from Kaggle/NVIDIA.
        import torch

        print(f'preflight torch : {torch.__version__}')
        if not torch.cuda.is_available():
            raise RuntimeError(
                'No CUDA GPU detected. In Kaggle Settings, set Accelerator to RTX PRO 6000 '
                '(or another GPU), then save a new version and rerun.'
            )
        print(f'preflight GPU   : {torch.cuda.get_device_name(0)}')

        !pip install -q --no-index \
            --find-links /kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages \
            datasets trl \
            --ignore-installed


        import datasets, trl
        print(f'datasets : {datasets.__version__}')   # expect 4.8.4
        print(f'trl      : {trl.__version__}')         # expect 0.29.1
        """,
    )

    set_source(
        3,
        r"""
        import os, sys, stat, shutil, gc, re, json, types, zipfile, math, time
        from collections import Counter

        import torch
        import torch.nn.functional as F
        import pandas as pd
        import matplotlib.pyplot as plt
        import kagglehub

        from datasets import Dataset, concatenate_datasets
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

        import triton.backends.nvidia.compiler as nv_compiler

        print(f'torch        : {torch.__version__}')
        print(f'GPU          : {torch.cuda.get_device_name(0)}')
        print(f'VRAM total   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        COMPUTE_DTYPE = torch.bfloat16

        TONG_ADAPTER_MODEL_URL = 'https://www.kaggle.com/models/huikang/nemotron-adapter/Transformers/default/20'
        TONG_ADAPTER_MODEL_HANDLE = 'huikang/nemotron-adapter/Transformers/default/20'
        WARM_START_ADAPTER_PATH = os.environ.get(
            'NEMOTRON_WARM_START_ADAPTER',
            '/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20',
        )
        if not WARM_START_ADAPTER_PATH or not os.path.exists(WARM_START_ADAPTER_PATH):
            WARM_START_ADAPTER_PATH = None
        if WARM_START_ADAPTER_PATH is None:
            try:
                print('Tong adapter is not mounted under /kaggle/input; downloading via kagglehub ...')
                WARM_START_ADAPTER_PATH = kagglehub.model_download(TONG_ADAPTER_MODEL_HANDLE)
                print(f'kagglehub adapter path: {WARM_START_ADAPTER_PATH}')
            except Exception as kagglehub_exc:
                print(f'kagglehub model_download failed: {kagglehub_exc!r}')
                WARM_START_ADAPTER_PATH = None
        REQUIRE_WARM_START = True
        if REQUIRE_WARM_START and WARM_START_ADAPTER_PATH is None:
            raise FileNotFoundError(
                'Tong warm-start adapter is required. Either attach it in Kaggle editor with '
                + TONG_ADAPTER_MODEL_URL
                + ', or allow kagglehub.model_download('
                + TONG_ADAPTER_MODEL_HANDLE
                + ') to resolve it, or set NEMOTRON_WARM_START_ADAPTER to the mounted path.'
            )

        VARIANT_NAME = """
        + repr(args.variant)
        + r"""
        VARIANT_TITLE = """
        + repr(str(variant["title"]))
        + r"""
        VARIANT_NOTE = """
        + repr(str(variant["note"]))
        + r"""
        VARIANT_SMALL_SUBTYPE_OVERRIDES = """
        + pformat(variant["small_subtype_overrides"], width=100)
        + r"""
        VARIANT_TONG_STYLE_OVERRIDES = """
        + pformat(variant["tong_style_overrides"], width=100)
        + r"""
        VARIANT_PRIORITY_DUPLICATION_FACTOR_SMALL = """
        + repr(int(variant["priority_duplication_factor_small"]))
        + r"""

        LORA_RANK = 8
        SFT_MAX_LEN = int(os.environ.get('NEMOTRON_SFT_MAX_LEN', '"""
        + str(int(variant["sft_max_len"]))
        + r"""'))
        SFT_EPOCHS = int(os.environ.get('NEMOTRON_SFT_EPOCHS', '"""
        + str(int(variant["sft_epochs"]))
        + r"""'))
        SFT_PER_DEVICE_BATCH = int(os.environ.get('NEMOTRON_SFT_PER_DEVICE_BATCH', '1'))
        SFT_GRAD_ACCUM = int(os.environ.get('NEMOTRON_SFT_GRAD_ACCUM', '"""
        + str(int(variant["sft_grad_accum"]))
        + r"""'))
        SMALL_SFT_MODE = os.environ.get('NEMOTRON_SMALL_SFT_MODE', '1') == '1'
        TARGET_SAMPLES_PER_TASK = int(os.environ.get('NEMOTRON_TARGET_SAMPLES_PER_TASK', '128'))
        PREFER_TONG_STYLE_DATA = os.environ.get('NEMOTRON_PREFER_TONG_STYLE_DATA', '1') == '1'
        SMALL_PROFILE_TASKS = {
            'numeral',
            'unit_conversion',
            'gravity',
            'cipher',
            'symbolic_transform',
            'bit_manipulation',
            'equation',
        }
        SMALL_TASK_LIMITS = {
            'numeral': 96,
            'unit_conversion': 128,
            'gravity': 128,
            'cipher': 128,
            'symbolic_transform': 128,
        }
        SMALL_SUBTYPE_LIMITS = {
            ('numeral', 'roman_int'): 96,
            ('unit_conversion', 'linear_ratio'): 128,
            ('gravity', 'fall_distance_from_time'): 128,
            ('cipher', 'monoalphabetic_substitution'): 128,
            ('symbolic_transform', 'symbol_string_rewrite'): 128,
            ('bit_manipulation', 'shift'): 48,
            ('bit_manipulation', 'rotate'): 48,
            ('bit_manipulation', 'bit_pick_or_negate'): 24,
            ('bit_manipulation', 'pairwise_parity_family'): 24,
            ('bit_manipulation', 'pairwise_conjunctive_family'): 48,
            ('bit_manipulation', 'pairwise_disjunctive_family'): 48,
            ('bit_manipulation', 'pairwise_parity_conjunctive_mix'): 64,
            ('equation', 'whole_number_evaluate'): 24,
            ('equation', 'scalar_reduce'): 24,
            ('equation', 'digitwise_compose'): 48,
            ('equation', 'query_filtered_whole_number_evaluate'): 24,
            ('equation', 'query_filtered_scalar_reduce'): 24,
            ('equation', 'query_filtered_digitwise_compose'): 24,
        }
        SMALL_SUBTYPE_LIMITS.update(VARIANT_SMALL_SUBTYPE_OVERRIDES)

        SFT_LR = float(os.environ.get(
            'NEMOTRON_SFT_LR',
            '"""
        + str(variant["sft_lr"])
        + r"""',
        ))
        SFT_MAX_GRAD_NORM = float(os.environ.get('NEMOTRON_SFT_MAX_GRAD_NORM', '1000000000.0'))

        ENABLE_OPTIONAL_GRPO = False
        GRPO_SAMPLE_SIZE = 1024
        GRPO_EPOCHS = 1
        GRPO_LR = 5e-6
        GRPO_NUM_GEN = 4
        GRPO_MAX_COMP_LEN = 256

        LOCAL_EVAL_SAMPLES = 48
        LOCAL_EVAL_MIN_ACCURACY = float(os.environ.get('NEMOTRON_MIN_LOCAL_SANITY', '0.20'))
        VAL_FRACTION = 0.05
        VAL_MIN_SIZE_PER_TASK = 2

        OUTPUT_DIR = '/kaggle/working'
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print('\nHyperparameters loaded:')
        print({
            'LORA_RANK': LORA_RANK,
            'SFT_MAX_LEN': SFT_MAX_LEN,
            'SFT_EPOCHS': SFT_EPOCHS,
            'SFT_LR': SFT_LR,
            'SFT_GRAD_ACCUM': SFT_GRAD_ACCUM,
            'TARGET_SAMPLES_PER_TASK': TARGET_SAMPLES_PER_TASK,
            'WARM_START_ADAPTER_PATH': WARM_START_ADAPTER_PATH,
            'REQUIRE_WARM_START': REQUIRE_WARM_START,
            'VARIANT_NAME': VARIANT_NAME,
            'VARIANT_TITLE': VARIANT_TITLE,
            'VARIANT_NOTE': VARIANT_NOTE,
            'SMALL_SFT_MODE': SMALL_SFT_MODE,
            'SMALL_PROFILE_TASKS': sorted(SMALL_PROFILE_TASKS),
            'SMALL_SUBTYPE_LIMITS': {str(k): v for k, v in SMALL_SUBTYPE_LIMITS.items()},
            'PREFER_TONG_STYLE_DATA': PREFER_TONG_STYLE_DATA,
            'ENABLE_OPTIONAL_GRPO': ENABLE_OPTIONAL_GRPO,
            'LOCAL_EVAL_MIN_ACCURACY': LOCAL_EVAL_MIN_ACCURACY,
        })
        """,
    )

    set_source(
        10,
        """
        ## 5. Tokenizer, Base Model, and Optional Warm Start

        > Load the Nemotron base model, apply the Blackwell fast-path fix, and optionally resume from the strong public Tinker adapter before any additional tuning.
        """,
    )

    set_source(
        11,
        r"""
        MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
        print(f'Model path: {MODEL_PATH}')

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'right'

        print('Loading model ...')
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map={'': 0},
            trust_remote_code=True,
            dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
        )
        print(f'VRAM used after load: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB')

        patched_mods = []
        for name, mod in sys.modules.items():
            if 'modeling_nemotron_h' in name:
                mod.is_fast_path_available = False
                patched_mods.append(name)
        print(f'Fix 4 OK  Fast path disabled in: {patched_mods}')
        """,
    )

    set_source(
        12,
        """
        ## 6. Adapter Strategy

        > Required path: continue training from Tong's public Tinker adapter. This notebook does not fall back to a fresh LoRA, because the current route is adapter continuation rather than cold-start fine-tuning.
        """,
    )

    set_source(
        13,
        r"""
        print(f'Warm-starting from Tong adapter: {WARM_START_ADAPTER_PATH}')
        model = PeftModel.from_pretrained(
            model,
            WARM_START_ADAPTER_PATH,
            is_trainable=True,
        )

        model.print_trainable_parameters()
        print('Adapter is ready for continued training')
        """,
    )

    set_source(
        14,
        """
        ## 7. Metric-Aligned Trace Formatting, Small High-Confidence SFT Data, and Optional External CoT

        > Key fixes here:
        > 1. Do not force `\\boxed{}` when the gold answer contains `{`, `}`, or `\\`, because the official metric parser truncates those cases.
        > 2. Keep templated SFT on a small solved subset instead of broad full-data balancing.
        > 3. If an external CoT dataset is available, keep it small and avoid aggressive duplication.
        """,
    )

    set_source(
        15,
        r"""
        def clean_answer_text(answer) -> str:
            return str(answer).strip()


        def needs_plain_final_line(answer: str) -> bool:
            answer = clean_answer_text(answer)
            return any(ch in answer for ch in ['{', '}', '\\']) or '\n' in answer or '\r' in answer


        def format_final_answer(answer: str) -> str:
            answer = clean_answer_text(answer)
            if needs_plain_final_line(answer):
                return f'Final answer: {answer}'
            return f'Final answer: \\boxed{{{answer}}}'


        def join_trace(lines):
            return "\n".join(lines)


        def extract_roman_candidate(prompt: str):
            cands = re.findall(r'\b[IVXLCDM]+\b', prompt)
            if not cands:
                return None
            return max(cands, key=len)


        def extract_arabic_candidate(prompt: str):
            nums = re.findall(r'\b\d+\b', prompt)
            if not nums:
                return None
            return nums[-1]


        BIT_PAIR_RE = re.compile(r'([01]+)\s*->\s*([01]+)')
        EQUATION_PAIR_RE = re.compile(r'(\d{2})([^\w\s])(\d{2})\s*=\s*([^\n\r]+)')
        QUERY_EQUATION_RE = re.compile(
            r'Now,\s*determine the result for:\s*(\d{2})([^\w\s])(\d{2})',
            re.IGNORECASE,
        )


        def parse_bit_pairs(prompt: str):
            return BIT_PAIR_RE.findall(prompt)


        def parse_equation_examples(prompt: str):
            return [
                (m.group(1), m.group(2), m.group(3), m.group(4).strip())
                for m in EQUATION_PAIR_RE.finditer(prompt)
            ]


        def extract_query_equation(prompt: str):
            match = QUERY_EQUATION_RE.search(prompt)
            if match is None:
                return None
            return match.group(1), match.group(2), match.group(3)


        def bit_not(bits: str) -> str:
            return ''.join('1' if ch == '0' else '0' for ch in bits)


        def bit_shift_left(bits: str, amount: int) -> str:
            return bits[amount:] + ('0' * amount)


        def bit_shift_right(bits: str, amount: int) -> str:
            return ('0' * amount) + bits[:-amount]


        def bit_rotate_left(bits: str, amount: int) -> str:
            amount %= len(bits)
            return bits[amount:] + bits[:amount]


        def bit_rotate_right(bits: str, amount: int) -> str:
            amount %= len(bits)
            if amount == 0:
                return bits
            return bits[-amount:] + bits[:-amount]


        def equation_examples_match_whole_number_rule(pairs) -> bool:
            def all_match(transform):
                try:
                    return all(transform(left, right) == output for left, _, right, output in pairs)
                except Exception:
                    return False

            return any([
                all_match(lambda left, right: str(int(left) + int(right))),
                all_match(lambda left, right: str(int(left) - int(right))),
                all_match(lambda left, right: str(int(right) - int(left))),
                all_match(lambda left, right: str(abs(int(left) - int(right)))),
                all_match(lambda left, right: str(int(left) * int(right))),
                all_match(lambda left, right: left + right),
                all_match(lambda left, right: right + left),
            ])


        PAIRWISE_OPERATION_FAMILIES = {
            'pairwise_parity_family': (lambda a, b: a ^ b,),
            'pairwise_conjunctive_family': (lambda a, b: a & b,),
            'pairwise_disjunctive_family': (lambda a, b: a | b,),
        }
        PAIRWISE_MIX_SUBTYPE_BY_FAMILIES = {
            ('pairwise_parity_family', 'pairwise_conjunctive_family'): 'pairwise_parity_conjunctive_mix',
            ('pairwise_parity_family', 'pairwise_disjunctive_family'): 'pairwise_parity_disjunctive_mix',
            ('pairwise_conjunctive_family', 'pairwise_disjunctive_family'): 'pairwise_conjunctive_disjunctive_mix',
            (
                'pairwise_parity_family',
                'pairwise_conjunctive_family',
                'pairwise_disjunctive_family',
            ): 'pairwise_three_family_mix',
        }


        def pairwise_family_matches_target(inputs, target, width, operations) -> bool:
            for operation in operations:
                for left_index in range(width):
                    for right_index in range(width):
                        for negate_left in (False, True):
                            for negate_right in (False, True):
                                values = []
                                for row in inputs:
                                    left_value = 1 - row[left_index] if negate_left else row[left_index]
                                    right_value = 1 - row[right_index] if negate_right else row[right_index]
                                    values.append(operation(left_value, right_value))
                                if values == target:
                                    return True
            return False


        def infer_pairwise_family_cover(pairs):
            if not pairs:
                return None
            widths = {len(bits) for pair in pairs for bits in pair}
            if len(widths) != 1:
                return None
            width = widths.pop()
            inputs = [[int(ch) for ch in left] for left, _ in pairs]
            outputs = [[int(ch) for ch in right] for _, right in pairs]
            allowed_families_per_position = []

            for position in range(width):
                target = [row[position] for row in outputs]
                allowed = {
                    family_name
                    for family_name, operations in PAIRWISE_OPERATION_FAMILIES.items()
                    if pairwise_family_matches_target(inputs, target, width, operations)
                }
                if not allowed:
                    return None
                allowed_families_per_position.append(allowed)

            family_names = tuple(PAIRWISE_OPERATION_FAMILIES)
            for cover_size in range(1, len(family_names) + 1):
                for family_cover in __import__('itertools').combinations(family_names, cover_size):
                    family_cover_set = set(family_cover)
                    if all(set(allowed) & family_cover_set for allowed in allowed_families_per_position):
                        return family_cover
            return None


        def classify_bit_small_profile_subtype(prompt: str) -> str:
            pairs = parse_bit_pairs(prompt)
            if not pairs:
                return 'hybrid_boolean_program'

            lengths = {len(bits) for pair in pairs for bits in pair}
            if len(lengths) != 1:
                return 'pad_truncate'

            width = lengths.pop()
            if all(bit_not(left) == right for left, right in pairs):
                return 'NOT'

            for amount in range(1, width):
                if all(bit_shift_left(left, amount) == right for left, right in pairs):
                    return 'shift'
                if all(bit_shift_right(left, amount) == right for left, right in pairs):
                    return 'shift'
                if all(bit_rotate_left(left, amount) == right for left, right in pairs):
                    return 'rotate'
                if all(bit_rotate_right(left, amount) == right for left, right in pairs):
                    return 'rotate'

            inputs = [[int(ch) for ch in left] for left, _ in pairs]
            outputs = [[int(ch) for ch in right] for _, right in pairs]
            bit_pick_ok = True
            for position in range(width):
                target = [row[position] for row in outputs]
                matched = False
                for source_index in range(width):
                    values = [row[source_index] for row in inputs]
                    if values == target or [1 - value for value in values] == target:
                        matched = True
                        break
                if not matched:
                    bit_pick_ok = False
                    break
            if bit_pick_ok:
                return 'bit_pick_or_negate'

            family_cover = infer_pairwise_family_cover(pairs)
            if family_cover is None:
                return 'hybrid_boolean_program'
            if len(family_cover) == 1:
                return family_cover[0]
            return PAIRWISE_MIX_SUBTYPE_BY_FAMILIES.get(family_cover, 'pairwise_mixed_family')


        def classify_equation_pairs(pairs, allow_multi_operator: bool = True) -> str:
            if len(pairs) < 2:
                return 'single_operator_hidden_numeric'
            outputs = [output for _, _, _, output in pairs]
            operators = {operator for _, operator, _, _ in pairs}
            if any(output.startswith('-') for output in outputs):
                return 'signed_result_rule'
            if any(output.startswith('0') and len(output) > 1 for output in outputs):
                return 'leading_zero_rule'
            if any(not re.fullmatch(r'-?\d+', output) for output in outputs):
                return 'literal_symbol_rule'
            if equation_examples_match_whole_number_rule(pairs):
                return 'whole_number_evaluate'
            lengths = {len(output.lstrip('-')) for output in outputs}
            if max(lengths) <= 2:
                return 'scalar_reduce'
            if min(lengths) >= 3:
                return 'digitwise_compose'
            if allow_multi_operator and len(operators) > 1:
                return 'multi_operator_examples'
            return 'single_operator_hidden_numeric'


        def classify_equation_small_profile_subtype(prompt: str) -> str:
            pairs = parse_equation_examples(prompt)
            if len(pairs) < 2:
                return 'single_operator_hidden_numeric'

            operators = {operator for _, operator, _, _ in pairs}
            query_equation = extract_query_equation(prompt)
            if len(operators) > 1 and query_equation is not None:
                _, query_operator, _ = query_equation
                filtered_pairs = [pair for pair in pairs if pair[1] == query_operator]
                if len(filtered_pairs) >= 2:
                    filtered_subtype = classify_equation_pairs(filtered_pairs, allow_multi_operator=False)
                    query_filtered_map = {
                        'whole_number_evaluate': 'query_filtered_whole_number_evaluate',
                        'scalar_reduce': 'query_filtered_scalar_reduce',
                        'digitwise_compose': 'query_filtered_digitwise_compose',
                    }
                    if filtered_subtype in query_filtered_map:
                        return query_filtered_map[filtered_subtype]

            return classify_equation_pairs(pairs, allow_multi_operator=True)


        def infer_small_profile_subtype(example: dict) -> str | None:
            task = example.get('task_type', 'other')
            prompt = example['prompt']
            if task == 'numeral':
                return 'roman_int'
            if task == 'unit_conversion':
                return 'linear_ratio'
            if task == 'gravity':
                return 'fall_distance_from_time'
            if task == 'cipher':
                return 'monoalphabetic_substitution'
            if task == 'symbolic_transform':
                return 'symbol_string_rewrite'
            if task == 'bit_manipulation':
                return classify_bit_small_profile_subtype(prompt)
            if task == 'equation':
                return classify_equation_small_profile_subtype(prompt)
            return None


        def build_symbolic_transform_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            return join_trace([
                'Step 1: Read the example input-output pairs carefully.',
                'Step 2: Track how each symbol or local symbol pattern changes from input to output.',
                'Step 3: Identify the deterministic transformation rule that is consistent with the examples.',
                'Step 4: Apply the same symbol transformation to the query string from left to right.',
                f'Step 5: The transformed string is {answer}.',
                format_final_answer(answer),
            ])


        def build_numeral_trace(example: dict) -> str:
            prompt = example['prompt']
            answer = clean_answer_text(example['answer'])
            roman = extract_roman_candidate(prompt)
            arabic = extract_arabic_candidate(prompt)
            p = prompt.lower()

            if (
                'write the number' in p
                or 'numeral system' in p
                or 'convert the number' in p
                or re.search(r'now,\s*write', p)
            ):
                return join_trace([
                    'Step 1: Identify the target integer in the prompt.',
                    f'Integer: {arabic}' if arabic else 'Integer: [read from prompt]',
                    'Step 2: Convert the integer into the target numeral system using the observed examples.',
                    'Step 3: Build the numeral from largest place value to smallest place value.',
                    f'Step 4: The numeral representation is {answer}.',
                    format_final_answer(answer),
                ])

            return join_trace([
                'Step 1: Identify the Roman numeral in the prompt.',
                f'Roman numeral: {roman}' if roman else 'Roman numeral: [read from prompt]',
                'Step 2: Use the values I=1, V=5, X=10, L=50, C=100, D=500, M=1000.',
                'Step 3: Scan from left to right. If a symbol is smaller than the next symbol, subtract it. Otherwise, add it.',
                f'Step 4: The computed integer is {answer}.',
                format_final_answer(answer),
            ])


        def build_unit_conversion_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            return join_trace([
                'Step 1: Read the given quantity and the target unit.',
                'Step 2: Choose the conversion factor that matches the source unit and the target unit.',
                'Step 3: Apply multiplication or division one step at a time, while keeping the units consistent.',
                'Step 4: Simplify the expression until only the requested unit remains.',
                f'Step 5: The converted value is {answer}.',
                format_final_answer(answer),
            ])


        def build_gravity_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            return join_trace([
                'Step 1: Identify the physical quantity being asked for.',
                'Step 2: Read the gravity constant or the planet-specific multiplier from the prompt.',
                'Step 3: Substitute the known value into the required formula.',
                'Step 4: Perform the arithmetic carefully and keep the units consistent.',
                f'Step 5: The resulting value is {answer}.',
                format_final_answer(answer),
            ])


        def build_cipher_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            return join_trace([
                'Step 1: Read the example input-output pairs.',
                'Step 2: Build the current character mapping from the examples.',
                'Step 3: Infer any remaining unmapped characters consistently.',
                'Step 4: Decode the target text from left to right, preserving spaces and punctuation.',
                f'Step 5: The decoded text is {answer}.',
                format_final_answer(answer),
            ])


        def build_bit_manipulation_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            subtype = example.get('task_subtype') or infer_small_profile_subtype(example)

            if subtype == 'shift':
                return join_trace([
                    'Step 1: Compare the examples and identify the single shift direction and amount.',
                    'Step 2: Confirm that dropped bits are replaced by zeros, not wrapped around.',
                    'Step 3: Apply the same zero-filled shift to the query input.',
                    f'Step 4: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'rotate':
                return join_trace([
                    'Step 1: Compare the examples and identify the single rotation direction and amount.',
                    'Step 2: Confirm that bits wrap around at the ends instead of disappearing.',
                    'Step 3: Rotate the query input with the same wraparound rule.',
                    f'Step 4: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'bit_pick_or_negate':
                return join_trace([
                    'Step 1: Compare each output position against the input positions directly.',
                    'Step 2: Notice that every output bit copies or flips one fixed input bit.',
                    'Step 3: Apply the same position mapping to the query input.',
                    f'Step 4: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'pairwise_parity_family':
                return join_trace([
                    'Step 1: Read all example binary input-output pairs.',
                    'Step 2: Compare each output position with XOR-style parity between two source bits.',
                    'Step 3: Keep the single parity-style local program that fits all examples.',
                    'Step 4: Apply the same parity rule to the query input.',
                    f'Step 5: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'pairwise_conjunctive_family':
                return join_trace([
                    'Step 1: Read all example binary input-output pairs.',
                    'Step 2: Compare each output position with AND-style gating between two source bits.',
                    'Step 3: Keep the single conjunctive local rule family that fits all examples.',
                    'Step 4: Apply the same conjunctive rule to the query input.',
                    f'Step 5: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'pairwise_disjunctive_family':
                return join_trace([
                    'Step 1: Read all example binary input-output pairs.',
                    'Step 2: Compare each output position with OR-style gating between two source bits.',
                    'Step 3: Keep the single disjunctive local rule family that fits all examples.',
                    'Step 4: Apply the same disjunctive rule to the query input.',
                    f'Step 5: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'pairwise_parity_conjunctive_mix':
                return join_trace([
                    'Step 1: Read all example binary input-output pairs.',
                    'Step 2: Confirm that every output position still depends on only two source bits.',
                    'Step 3: Separate the parity-style positions from the conjunctive-style positions.',
                    'Step 4: Apply the same parity/conjunctive position map to the query input.',
                    f'Step 5: The resulting binary output is {answer}.',
                    format_final_answer(answer),
                ])

            return join_trace([
                'Step 1: Read all example binary input-output pairs.',
                'Step 2: Compare each output bit with candidate operations on the input bits.',
                'Step 3: Select the single deterministic bit rule that matches all examples.',
                'Step 4: Apply the same rule to the query input to produce the final binary output.',
                f'Step 5: The resulting binary output is {answer}.',
                format_final_answer(answer),
            ])


        def build_equation_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            subtype = example.get('task_subtype') or infer_small_profile_subtype(example)

            if subtype == 'whole_number_evaluate':
                return join_trace([
                    'Step 1: Read the example equations and test direct whole-number operations on the two operands.',
                    'Step 2: Keep the simplest operation or concatenation rule that fits all examples.',
                    'Step 3: Apply the same whole-number rule to the query operands.',
                    f'Step 4: The resulting value is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'scalar_reduce':
                return join_trace([
                    'Step 1: Read the examples and notice that each equation reduces to one short scalar output.',
                    'Step 2: Infer the repeated reduction pattern that maps the operands to that scalar.',
                    'Step 3: Apply the same reduction to the query operands.',
                    f'Step 4: The resulting value is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'digitwise_compose':
                return join_trace([
                    'Step 1: Split each two-digit operand into tens and ones digits.',
                    'Step 2: Infer the repeated digit-level transformation used by the examples.',
                    'Step 3: Compose the query result by joining the digit-level partial outputs in the same order.',
                    f'Step 4: The resulting value is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'query_filtered_whole_number_evaluate':
                return join_trace([
                    'Step 1: Note the query operator and ignore examples with different symbols.',
                    'Step 2: Test direct whole-number rules on the filtered examples only.',
                    'Step 3: Apply the same whole-number rule to the query operands.',
                    f'Step 4: The resulting value is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'query_filtered_scalar_reduce':
                return join_trace([
                    'Step 1: Note the query operator and ignore examples with different symbols.',
                    'Step 2: Infer the scalar reduction rule from the filtered examples only.',
                    'Step 3: Apply the same scalar reduction to the query operands.',
                    f'Step 4: The resulting value is {answer}.',
                    format_final_answer(answer),
                ])

            if subtype == 'query_filtered_digitwise_compose':
                return join_trace([
                    'Step 1: Note the query operator and ignore examples with different symbols.',
                    'Step 2: Split the filtered examples into digit-level pieces.',
                    'Step 3: Apply the same digitwise composition rule to the query operands.',
                    f'Step 4: The resulting value is {answer}.',
                    format_final_answer(answer),
                ])

            return join_trace([
                'Step 1: Read the example equations carefully.',
                'Step 2: Check whether the operands or the result are reversed before applying the rule.',
                'Step 3: Test the candidate arithmetic transformation that is consistent with the examples.',
                'Step 4: Apply the same transformation to the query equation.',
                f'Step 5: The resulting value is {answer}.',
                format_final_answer(answer),
            ])


        def build_cryptarithm_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            return join_trace([
                'Step 1: Read the symbolic examples carefully.',
                'Step 2: Infer the operation pattern that is consistent with the examples.',
                'Step 3: Apply the same pattern to the query expression.',
                f'Step 4: The resulting value is {answer}.',
                format_final_answer(answer),
            ])


        def build_default_trace(example: dict) -> str:
            answer = clean_answer_text(example['answer'])
            task = example.get('task_type', 'other')
            return join_trace([
                f'Step 1: Task category is {task}.',
                'Step 2: Follow a deterministic step-by-step procedure.',
                'Step 3: Keep the wording stable and avoid unnecessary variation.',
                f'Step 4: The result is {answer}.',
                format_final_answer(answer),
            ])


        def build_trace(example: dict) -> str:
            task = example.get('task_type', 'other')
            if task == 'numeral':
                return build_numeral_trace(example)
            if task == 'unit_conversion':
                return build_unit_conversion_trace(example)
            if task == 'gravity':
                return build_gravity_trace(example)
            if task == 'cipher':
                return build_cipher_trace(example)
            if task == 'symbolic_transform':
                return build_symbolic_transform_trace(example)
            if task == 'bit_manipulation':
                return build_bit_manipulation_trace(example)
            if task == 'equation':
                return build_equation_trace(example)
            if task == 'cryptarithm':
                return build_cryptarithm_trace(example)
            return build_default_trace(example)


        SYSTEM_PROMPT = (
            'You are a precise reasoning assistant. '
            'Use short deterministic numbered steps. '
            'Keep wording stable. '
            'End with a line that starts with "Final answer:". '
            'Use \\boxed{} only when the final answer does not contain braces or backslashes.'
        )

        USER_SUFFIX = (
            '\nUse short deterministic numbered steps. '
            'End with Final answer:. '
            'Use \\boxed{} only when the final answer does not contain braces or backslashes.'
        )


        def build_messages(prompt: str, assistant_text: str | None = None, add_generation_prompt: bool = False):
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt + USER_SUFFIX},
            ]
            if assistant_text is not None:
                messages.append({'role': 'assistant', 'content': assistant_text})
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                text = (
                    f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
                    f'<|im_start|>user\n{prompt + USER_SUFFIX}<|im_end|>\n'
                )
                if assistant_text is None:
                    return text + '<|im_start|>assistant\n'
                return text + f'<|im_start|>assistant\n{assistant_text}<|im_end|>'


        def format_for_sft(example: dict) -> dict:
            assistant_msg = build_trace(example)
            text = build_messages(example['prompt'], assistant_msg, add_generation_prompt=False)
            text = re.sub(r"<think>\s*</think>", "", text)
            return {'text': text}


        def normalize_external_cot(text: str) -> str:
            text = str(text).strip()
            text = re.sub(r'Final answer\s*[:：].*$', '', text, flags=re.I | re.S).rstrip()
            text = re.sub(r'\\boxed\{[^}]*\}', '', text).rstrip()
            return text


        OPTIONAL_COT_PATHS = [
            '/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv',
        ]


        def load_optional_cot_rows() -> pd.DataFrame | None:
            for path in OPTIONAL_COT_PATHS:
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                required = {'prompt', 'answer'}
                if not required.issubset(df.columns):
                    continue
                cot_col = None
                for candidate in ['generated_cot', 'cot', 'reasoning']:
                    if candidate in df.columns:
                        cot_col = candidate
                        break
                if cot_col is None:
                    continue
                df = df.copy()
                df = df[df[cot_col].notna()].reset_index(drop=True)
                if 'task_type' not in df.columns:
                    df['task_type'] = df['prompt'].apply(classify_task)
                df['assistant_text'] = (
                    df[cot_col].astype(str).map(normalize_external_cot)
                    + '\n'
                    + df['answer'].map(format_final_answer)
                )
                print(f'Loaded optional external CoT rows from {path}: {len(df)}')
                return df[['prompt', 'answer', 'task_type', 'assistant_text']]
            print('No optional external CoT dataset found. Using metric-aligned templated traces only.')
            return None


        def stratified_split_by_task(df: pd.DataFrame, seed: int = 42):
            train_parts, val_parts = [], []
            for _, group in df.groupby('task_type', dropna=False):
                group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
                if len(group) <= 4:
                    train_parts.append(group)
                    continue
                val_n = max(VAL_MIN_SIZE_PER_TASK, int(round(len(group) * VAL_FRACTION)))
                val_n = min(val_n, max(1, len(group) // 5))
                val_n = min(val_n, len(group) - 2)
                train_parts.append(group.iloc[:-val_n].reset_index(drop=True))
                val_parts.append(group.iloc[-val_n:].reset_index(drop=True))
            train_df_local = pd.concat(train_parts, ignore_index=True)
            val_df_local = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[:0].copy()
            return train_df_local, val_df_local


        def balance_by_task(df: pd.DataFrame, task_col: str, target_per_task: int, seed: int = 42):
            chunks = []
            for _, group in df.groupby(task_col, dropna=False):
                if len(group) >= target_per_task:
                    chunk = group.sample(n=target_per_task, random_state=seed)
                else:
                    extra = group.sample(
                        n=target_per_task - len(group),
                        replace=True,
                        random_state=seed,
                    )
                    chunk = pd.concat([group, extra], ignore_index=True)
                chunks.append(chunk)
            return (
                pd.concat(chunks, ignore_index=True)
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True)
            )


        def build_small_profile_candidates(df: pd.DataFrame):
            if not SMALL_SFT_MODE:
                return df.copy()

            df = df[df['task_type'].isin(SMALL_PROFILE_TASKS)].copy()
            if len(df) == 0:
                return df

            df['task_subtype'] = [
                infer_small_profile_subtype({
                    'prompt': prompt,
                    'task_type': task_type,
                })
                for prompt, task_type in zip(df['prompt'], df['task_type'])
            ]
            allowed_pairs = set(SMALL_SUBTYPE_LIMITS)
            keep_mask = [
                (task_type, task_subtype) in allowed_pairs
                for task_type, task_subtype in zip(df['task_type'], df['task_subtype'])
            ]
            return df.loc[keep_mask].reset_index(drop=True)


        def build_small_high_confidence_subset(df: pd.DataFrame, seed: int = 42):
            if not SMALL_SFT_MODE:
                return balance_by_task(
                    df,
                    task_col='task_type',
                    target_per_task=TARGET_SAMPLES_PER_TASK,
                    seed=seed,
                )

            df = build_small_profile_candidates(df)
            chunks = []
            for (task_type, task_subtype), group in df.groupby(['task_type', 'task_subtype'], dropna=False):
                limit = SMALL_SUBTYPE_LIMITS.get((task_type, task_subtype), SMALL_TASK_LIMITS.get(task_type, TARGET_SAMPLES_PER_TASK))
                take = min(limit, len(group))
                chunks.append(group.sample(n=take, random_state=seed))
            if not chunks:
                return df.iloc[:0].copy()
            return (
                pd.concat(chunks, ignore_index=True)
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True)
            )

        train_core = train_df[['prompt', 'answer', 'task_type']].copy()
        templated_core = build_small_profile_candidates(train_core) if SMALL_SFT_MODE else train_core
        train_part, val_part = stratified_split_by_task(templated_core, seed=42)
        train_balanced = build_small_high_confidence_subset(train_part, seed=42)

        hf_train_sft = Dataset.from_pandas(train_balanced).map(
            format_for_sft,
            remove_columns=train_balanced.columns.tolist(),
        )
        hf_val_sft = Dataset.from_pandas(val_part).map(
            format_for_sft,
            remove_columns=val_part.columns.tolist(),
        )

        extra_cot_rows = load_optional_cot_rows()
        if extra_cot_rows is not None and len(extra_cot_rows):
            def format_external_cot_for_sft(example: dict) -> dict:
                text = build_messages(example['prompt'], example['assistant_text'], add_generation_prompt=False)
                text = re.sub(r"<think>\s*</think>", "", text)
                return {'text': text}

            extra_train = balance_by_task(
                extra_cot_rows[['prompt', 'answer', 'task_type', 'assistant_text']],
                task_col='task_type',
                target_per_task=min(1200, TARGET_SAMPLES_PER_TASK),
                seed=7,
            )
            hf_extra_sft = Dataset.from_pandas(extra_train).map(
                format_external_cot_for_sft,
                remove_columns=extra_train.columns.tolist(),
            )
            hf_train_sft = concatenate_datasets([hf_train_sft, hf_extra_sft]).shuffle(seed=42)
            print(f'After mixing in optional external CoT rows: {len(hf_train_sft)} train samples')

        print(f'Raw train split   : {len(train_part)}')
        print(f'Balanced train    : {len(train_balanced)}')
        print(f'Validation split  : {len(val_part)}')
        print(f'SFT train dataset : {len(hf_train_sft)} samples')
        print(f'SFT val dataset   : {len(hf_val_sft)} samples')
        print('\nSample training text:')
        print(hf_train_sft[0]['text'][:1800])

        # Rebuild the SFT dataset with competition-aligned Tong-style weighting when available.
        def normalize_external_cot(text: str) -> str:
            text = str(text).strip()
            text = re.sub(r'Final answer\s*[:：].*$', '', text, flags=re.I | re.S).rstrip()
            text = re.sub(r'\\boxed\{[^}]*\}', '', text).rstrip()
            return text


        def build_assistant_text_from_cot(cot: str, answer: str) -> str:
            return normalize_external_cot(cot) + '\n' + format_final_answer(answer)


        def format_external_cot_for_sft(example: dict) -> dict:
            text = build_messages(example['prompt'], example['assistant_text'], add_generation_prompt=False)
            text = re.sub(r"<think>\s*</think>", "", text)
            return {'text': text}


        TONG_STYLE_TYPE_DIRS = [
            '/kaggle/input/datasets/konbu17/exp024-tong-style-cot-data/type_tong',
            '/kaggle/input/datasets/dgxchen/nemotron-cot-tong/type_tong',
        ]
        TONG_STYLE_PRIORITY_FILES = [
            '/kaggle/input/datasets/konbu17/exp024-tong-style-cot-data/priority/exp026_s011_priority.txt',
            '/kaggle/input/datasets/dgxchen/nemotron-cot-tong/priority/exp026_s011_priority.txt',
        ]
        TONG_STYLE_BASE_SAMPLES = {
            'numeral_system.csv': 96 if SMALL_SFT_MODE else 600,
            'gravity_physics.csv': 128 if SMALL_SFT_MODE else 1200,
            'unit_conversion.csv': 128 if SMALL_SFT_MODE else 1150,
            'text_decryption.csv': 128 if SMALL_SFT_MODE else 1492,
            'bit_manipulation_including_wrong.csv': 128 if SMALL_SFT_MODE else 1508,
            'bit_manipulation_synth_including_wrong_v2.csv': 64 if SMALL_SFT_MODE else 500,
            'equation_numeric.csv': 96 if SMALL_SFT_MODE else 535,
            'cryptarithm.csv': 0 if SMALL_SFT_MODE else 69,
        }
        TONG_STYLE_BASE_SAMPLES.update(VARIANT_TONG_STYLE_OVERRIDES)


        def load_priority_ids(paths: list[str]) -> set[str]:
            for path in paths:
                if not os.path.exists(path):
                    continue
                with open(path, encoding='utf-8') as f:
                    ids = {line.strip() for line in f if line.strip()}
                print(f'Loaded priority IDs from {path}: {len(ids)}')
                return ids
            print('No Tong-style priority file found.')
            return set()


        def load_tong_style_rows() -> pd.DataFrame | None:
            if not PREFER_TONG_STYLE_DATA:
                print('Tong-style data loading disabled by environment.')
                return None

            base_dir = None
            for path in TONG_STYLE_TYPE_DIRS:
                if os.path.exists(path):
                    base_dir = path
                    break
            if base_dir is None:
                print('No Tong-style per-category dataset found.')
                return None

            priority_ids = load_priority_ids(TONG_STYLE_PRIORITY_FILES)
            rows = []
            for fname, n in TONG_STYLE_BASE_SAMPLES.items():
                path = os.path.join(base_dir, fname)
                if not os.path.exists(path):
                    print(f'Missing Tong-style file: {path}')
                    continue
                df = pd.read_csv(path)
                if 'generated_cot' not in df.columns:
                    print(f'Skipping {fname}: generated_cot column missing')
                    continue
                df = df[df['generated_cot'].notna() & (df['generated_cot'].astype(str).str.len() > 5)].copy()
                if len(df) == 0:
                    continue
                if n <= 0:
                    continue
                sampled = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
                print(f'Tong-style sample {fname}: {len(sampled)}/{n}')
                for row in sampled.to_dict('records'):
                    rows.append({
                        'id': str(row.get('id', '')),
                        'prompt': str(row['prompt']),
                        'answer': clean_answer_text(row['answer']),
                        'task_type': classify_task(str(row['prompt'])),
                        'assistant_text': build_assistant_text_from_cot(row['generated_cot'], row['answer']),
                    })

            if not rows:
                print('Tong-style dataset path exists, but no usable rows were found.')
                return None

            if priority_ids:
                duplication_factor = 1 if not SMALL_SFT_MODE else VARIANT_PRIORITY_DUPLICATION_FACTOR_SMALL
                if not SMALL_SFT_MODE:
                    duplication_factor = max(duplication_factor, 2)
                if duplication_factor > 1:
                    dup_rows = []
                    for _ in range(duplication_factor - 1):
                        dup_rows.extend(dict(row) for row in rows if row['id'] in priority_ids)
                    rows.extend(dup_rows)
                    print(f'Priority duplicated Tong-style rows: {len(dup_rows)}')
            elif not SMALL_SFT_MODE:
                hard_rows = [
                    dict(row)
                    for row in rows
                    if row['task_type'] in {'bit_manipulation', 'equation', 'cryptarithm'}
                ]
                rows.extend(hard_rows)
                print(f'Heuristic hard-task duplicated Tong-style rows: {len(hard_rows)}')

            result = pd.DataFrame(rows)
            print(f'Tong-style training rows after duplication: {len(result)}')
            return result[['prompt', 'answer', 'task_type', 'assistant_text']]


        def load_optional_cot_rows() -> pd.DataFrame | None:
            for path in OPTIONAL_COT_PATHS:
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                required = {'prompt', 'answer'}
                if not required.issubset(df.columns):
                    continue
                cot_col = None
                for candidate in ['generated_cot', 'cot', 'reasoning']:
                    if candidate in df.columns:
                        cot_col = candidate
                        break
                if cot_col is None:
                    continue
                df = df.copy()
                df = df[df[cot_col].notna()].reset_index(drop=True)
                if 'task_type' not in df.columns:
                    df['task_type'] = df['prompt'].apply(classify_task)
                df['assistant_text'] = [
                    build_assistant_text_from_cot(cot, ans)
                    for cot, ans in zip(df[cot_col], df['answer'])
                ]
                print(f'Loaded optional external CoT rows from {path}: {len(df)}')
                return df[['prompt', 'answer', 'task_type', 'assistant_text']]
            print('No optional external CoT dataset found. Using metric-aligned templated traces only.')
            return None


        hf_train_core = Dataset.from_pandas(train_balanced).map(
            format_for_sft,
            remove_columns=train_balanced.columns.tolist(),
        )
        hf_train_sft = hf_train_core
        tong_style_rows = load_tong_style_rows()
        extra_cot_rows = None

        if tong_style_rows is not None and len(tong_style_rows):
            hf_tong_sft = Dataset.from_pandas(tong_style_rows).map(
                format_external_cot_for_sft,
                remove_columns=tong_style_rows.columns.tolist(),
            )
            hf_train_sft = concatenate_datasets([hf_tong_sft, hf_train_core]).shuffle(seed=42)
        else:
            extra_cot_rows = load_optional_cot_rows()
            if extra_cot_rows is not None and len(extra_cot_rows):
                extra_train = balance_by_task(
                    extra_cot_rows[['prompt', 'answer', 'task_type', 'assistant_text']],
                    task_col='task_type',
                    target_per_task=min(128 if SMALL_SFT_MODE else 1200, TARGET_SAMPLES_PER_TASK),
                    seed=7,
                )
                hf_extra_sft = Dataset.from_pandas(extra_train).map(
                    format_external_cot_for_sft,
                    remove_columns=extra_train.columns.tolist(),
                )
                hf_train_sft = concatenate_datasets([hf_train_core, hf_extra_sft]).shuffle(seed=42)

        print('\nCompetition-aligned dataset rebuild:')
        print(f'Tong-style rows      : {0 if tong_style_rows is None else len(tong_style_rows)}')
        print(f'Optional matched CoT : {0 if extra_cot_rows is None else len(extra_cot_rows)}')
        print(f'Final SFT train set  : {len(hf_train_sft)} samples')
        print(hf_train_sft[0]['text'][:1800])
        """,
    )

    set_source(
        17,
        """
        ## 8. Continued SFT, Local Sanity Eval, Optional Targeted GRPO, and Packaging

        > This cell does the heavy lifting:
        > 1. Continued SFT from either a warm-start adapter or a fresh LoRA.
        > 2. A small greedy validation sanity-check using the official metric logic.
        > 3. Optional GRPO on a curated balanced subset.
        > 4. Save and validate `submission.zip`.
        """,
    )

    set_source(
        18,
        r"""
        def extract_final_answer_metric(text: str | None) -> str:
            if text is None:
                return 'NOT_FOUND'
            matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
            if matches:
                non_empty = [m.strip() for m in matches if m.strip()]
                if non_empty:
                    return non_empty[-1]
                return matches[-1].strip()
            patterns = [
                r'The final answer is:\s*([^\n]+)',
                r'Final answer is:\s*([^\n]+)',
                r'Final answer\s*[:：]\s*([^\n]+)',
                r'final answer\s*[:：]\s*([^\n]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[-1].strip()
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)
            if nums:
                return nums[-1]
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return lines[-1] if lines else 'NOT_FOUND'


        def verify_metric(stored_answer: str, predicted: str) -> bool:
            stored_answer = str(stored_answer).strip()
            predicted = str(predicted).strip()
            if re.fullmatch(r'[01]+', stored_answer):
                return predicted.lower() == stored_answer.lower()
            try:
                return math.isclose(
                    float(stored_answer),
                    float(predicted),
                    rel_tol=1e-2,
                    abs_tol=1e-5,
                )
            except Exception:
                return predicted.lower() == stored_answer.lower()


        def run_local_eval(model_to_eval, eval_df: pd.DataFrame, max_new_tokens: int = 256):
            if len(eval_df) == 0:
                print('Validation set is empty; skipping local eval.')
                return None

            eval_df = eval_df.sample(
                n=min(LOCAL_EVAL_SAMPLES, len(eval_df)),
                random_state=42,
            ).reset_index(drop=True)

            model_to_eval.eval()
            rows = []
            for row in eval_df.itertuples(index=False):
                prompt_text = build_messages(row.prompt, assistant_text=None, add_generation_prompt=True)
                inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    outputs = model_to_eval.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
                pred = extract_final_answer_metric(raw_text)
                rows.append({
                    'task_type': row.task_type,
                    'answer': row.answer,
                    'prediction': pred,
                    'correct': verify_metric(row.answer, pred),
                })

            eval_result = pd.DataFrame(rows)
            print(f"\nLocal sanity accuracy on {len(eval_result)} sampled validation rows: {eval_result['correct'].mean():.4f}")
            print(eval_result.groupby('task_type')['correct'].mean().sort_values(ascending=False))
            return eval_result


        os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = '/tmp/ptxas-blackwell'
        nv_compiler.get_ptxas_version = lambda arch: '12.0'
        print('Fix 5 OK  Triton env vars set')

        sft_config = SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=SFT_PER_DEVICE_BATCH,
            gradient_accumulation_steps=SFT_GRAD_ACCUM,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            learning_rate=SFT_LR,
            num_train_epochs=SFT_EPOCHS,
            logging_steps=10,
            save_strategy='no',
            fp16=False,
            bf16=True,
            optim='adamw_torch',
            lr_scheduler_type='linear',
            warmup_ratio=0.0,
            weight_decay=0.0,
            max_grad_norm=SFT_MAX_GRAD_NORM,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            report_to='none',
            dataset_text_field='text',
            max_length=SFT_MAX_LEN,
            packing=False,
        )

        model.config.use_cache = False
        sft_trainer = SFTTrainer(
            model=model,
            train_dataset=hf_train_sft,
            eval_dataset=hf_val_sft,
            args=sft_config,
            processing_class=tokenizer,
        )

        print('Starting continued SFT...')
        sft_trainer.train()
        print('SFT complete')

        local_eval_result = run_local_eval(sft_trainer.model, val_part, max_new_tokens=256)
        if local_eval_result is not None and len(local_eval_result):
            local_eval_accuracy = float(local_eval_result['correct'].mean())
            print(f'Local sanity threshold: {LOCAL_EVAL_MIN_ACCURACY:.4f}')
            if local_eval_accuracy < LOCAL_EVAL_MIN_ACCURACY:
                raise RuntimeError(
                    f'Local sanity accuracy {local_eval_accuracy:.4f} is below the threshold '
                    f'{LOCAL_EVAL_MIN_ACCURACY:.4f}. Aborting packaging to avoid a bad submission.'
                )
        final_model = sft_trainer.model
        """,
    )

    set_source(
        19,
        """
        ## 9. Optional Targeted GRPO and Final Packaging

        > Keep `ENABLE_OPTIONAL_GRPO = False` until the continued SFT run looks healthy. Once SFT is stable, flip it on for a small metric-aligned GRPO pass over a balanced subset.
        """,
    )

    set_source(
        20,
        r"""
        if ENABLE_OPTIONAL_GRPO:
            def reward_correct_answer(prompts, completions, **kwargs):
                answers = kwargs.get('answer', [None] * len(completions))
                rewards = []
                for comp, ans in zip(completions, answers):
                    pred = extract_final_answer_metric(comp)
                    rewards.append(1.0 if ans is not None and verify_metric(ans, pred) else 0.0)
                return rewards


            def reward_final_line(prompts, completions, **kwargs):
                return [
                    0.15 if re.search(r'Final answer\s*[:：]', c, flags=re.I) else -0.05
                    for c in completions
                ]


            def reward_safe_format(prompts, completions, **kwargs):
                allow_boxed = kwargs.get('allow_boxed', [True] * len(completions))
                rewards = []
                for comp, safe_to_box in zip(completions, allow_boxed):
                    has_box = bool(re.search(r'\\boxed\{[^}]*\}', comp))
                    if safe_to_box and has_box:
                        rewards.append(0.10)
                    elif (not safe_to_box) and ('Final answer:' in comp or 'Final answer：' in comp) and not has_box:
                        rewards.append(0.10)
                    else:
                        rewards.append(-0.02)
                return rewards

            def reward_final_line(prompts, completions, **kwargs):
                return [
                    0.15 if re.search(r'Final answer\s*[:：]', c, flags=re.I) else -0.05
                    for c in completions
                ]


            def reward_safe_format(prompts, completions, **kwargs):
                allow_boxed = kwargs.get('allow_boxed', [True] * len(completions))
                rewards = []
                for comp, safe_to_box in zip(completions, allow_boxed):
                    has_box = bool(re.search(r'\\boxed\{[^}]*\}', comp))
                    has_final_line = 'Final answer:' in comp or 'Final answer：' in comp
                    if safe_to_box and has_box:
                        rewards.append(0.10)
                    elif (not safe_to_box) and has_final_line and not has_box:
                        rewards.append(0.10)
                    else:
                        rewards.append(-0.02)
                return rewards


            def reward_length(prompts, completions, **kwargs):
                rewards = []
                for c in completions:
                    n = len(c.split())
                    if n < 15:
                        rewards.append(-0.10)
                    elif n > 600:
                        rewards.append(-0.05)
                    else:
                        rewards.append(0.02)
                return rewards


            def make_grpo_row(row: dict) -> dict:
                return {
                    'prompt': build_messages(row['prompt'], assistant_text=None, add_generation_prompt=True),
                    'answer': clean_answer_text(row['answer']),
                    'allow_boxed': not needs_plain_final_line(row['answer']),
                }


            task_count = train_part['task_type'].nunique()
            per_task_target = max(64, GRPO_SAMPLE_SIZE // max(1, task_count))
            grpo_seed = balance_by_task(
                train_part[['prompt', 'answer', 'task_type']].copy(),
                task_col='task_type',
                target_per_task=per_task_target,
                seed=123,
            )
            grpo_seed = grpo_seed.sample(
                n=min(GRPO_SAMPLE_SIZE, len(grpo_seed)),
                random_state=123,
            ).reset_index(drop=True)
            grpo_dataset = Dataset.from_list(
                [make_grpo_row(row) for row in grpo_seed.to_dict('records')]
            )
            print(f'GRPO dataset size: {len(grpo_dataset)}')

            grpo_config = GRPOConfig(
                output_dir=OUTPUT_DIR,
                num_train_epochs=GRPO_EPOCHS,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_generations=GRPO_NUM_GEN,
                max_completion_length=GRPO_MAX_COMP_LEN,
                max_prompt_length=min(4096, SFT_MAX_LEN),
                beta=0.001,
                learning_rate=GRPO_LR,
                lr_scheduler_type='cosine',
                warmup_ratio=0.05,
                bf16=True,
                logging_steps=5,
                save_strategy='no',
                optim='adamw_torch',
                max_grad_norm=0.1,
                report_to='none',
                use_vllm=False,
            )

            grpo_trainer = GRPOTrainer(
                model=final_model,
                processing_class=tokenizer,
                args=grpo_config,
                train_dataset=grpo_dataset,
                reward_funcs=[
                    reward_correct_answer,
                    reward_final_line,
                    reward_safe_format,
                    reward_length,
                ],
            )

            print('Starting optional GRPO...')
            grpo_trainer.train()
            print('GRPO complete')
            final_model = grpo_trainer.model
            _ = run_local_eval(final_model, val_part, max_new_tokens=256)

        print(f'Saving adapter to {OUTPUT_DIR}...')
        final_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print('Adapter and tokenizer saved')

        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                shutil.rmtree(item_path)
                print(f'Removed checkpoint: {item}')

        print(f'\nFiles in {OUTPUT_DIR}:')
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                print(f'  {f:55s}  {os.path.getsize(fpath) / 1024 / 1024:.1f} MB')

        config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
        weights_path = os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')
        readme_path = os.path.join(OUTPUT_DIR, 'README.md')

        assert os.path.exists(config_path), 'adapter_config.json missing'
        assert os.path.exists(weights_path), 'adapter_model.safetensors missing'

        with open(config_path) as f:
            cfg = json.load(f)

        rank = cfg.get('r', 999)
        assert rank <= 32, f'LoRA rank {rank} exceeds competition limit of 32'

        print(json.dumps(cfg, indent=2))
        print(f'\nLoRA rank = {rank} (<= 32)')

        if not os.path.exists(readme_path):
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(
                    '# Submission\n\n'
                    'Continued Nemotron LoRA adapter for the NVIDIA Nemotron Model Reasoning Challenge.\n'
                )
            print('README.md created')

        zip_path = '/kaggle/working/submission.zip'
        files_to_zip = ['adapter_model.safetensors', 'adapter_config.json', 'README.md']

        print(f'\nCreating {zip_path} ...')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in files_to_zip:
                fpath = os.path.join(OUTPUT_DIR, fname)
                if os.path.exists(fpath):
                    zf.write(fpath, fname)
                    print(f'  Added: {fname}')
                else:
                    print(f'  Missing: {fname}')

        zip_size = os.path.getsize(zip_path) / 1024 / 1024
        print(f'\nArchive size: {zip_size:.1f} MB')

        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            print(f'Contents: {names}')
            assert 'adapter_config.json' in names, 'adapter_config.json missing from zip'
            assert 'adapter_model.safetensors' in names, 'adapter_model.safetensors missing from zip'

        print('\nsubmission.zip is valid and ready for submission')
        print(f'Path: {zip_path}')

        from IPython.display import FileLink, display
        display(FileLink(zip_path))
        """,
    )

    nb.setdefault("metadata", {})
    nb["metadata"].setdefault("kaggle", {})
    nb["metadata"]["kaggle"]["accelerator"] = "nvidiaRtxPro6000"
    nb["metadata"]["kaggle"]["isGpuEnabled"] = True
    nb["metadata"]["kaggle"]["isInternetEnabled"] = False

    nb["metadata"].setdefault("codex_note", {})
    nb["metadata"]["codex_note"]["generated_from"] = str(src)
    nb["metadata"]["codex_note"]["strategy"] = (
        "Tong adapter continuation + small solved-subset SFT variants + metric-safe answer formatting"
    )
    nb["metadata"]["codex_note"]["variant"] = args.variant
    nb["metadata"]["codex_note"]["variant_title"] = str(variant["title"])
    nb["metadata"]["codex_note"]["variant_note"] = str(variant["note"])
    nb["metadata"]["codex_note"]["required_model_input"] = (
        "https://www.kaggle.com/models/huikang/nemotron-adapter/Transformers/default/20"
    )

    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote improved notebook to: {dst}")


if __name__ == "__main__":
    main()
