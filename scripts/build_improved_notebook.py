from __future__ import annotations

import argparse
import json
import pathlib
import textwrap


def to_source(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in text.splitlines()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite the original Kaggle notebook into the improved version.")
    parser.add_argument("--src", required=True, help="Source ipynb path.")
    parser.add_argument("--dst", required=True, help="Destination ipynb path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = pathlib.Path(args.src).expanduser().resolve()
    dst = pathlib.Path(args.dst).expanduser().resolve()

    nb = json.loads(src.read_text(encoding="utf-8"))

    def set_source(idx: int, text: str) -> None:
        nb["cells"][idx]["source"] = to_source(text)

    set_source(
        0,
        """
        ## 1. Offline Package Installation

        > Improved continuation notebook: longer context, safer metric-aligned answer formatting, optional warm-start from the strong Tinker adapter, balanced SFT, and optional GRPO on a curated subset.
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

        WARM_START_ADAPTER_PATH = os.environ.get(
            'NEMOTRON_WARM_START_ADAPTER',
            '/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20',
        )
        if not WARM_START_ADAPTER_PATH or not os.path.exists(WARM_START_ADAPTER_PATH):
            WARM_START_ADAPTER_PATH = None

        LORA_RANK = 32
        SFT_MAX_LEN = int(os.environ.get('NEMOTRON_SFT_MAX_LEN', '4096'))
        SFT_EPOCHS = int(os.environ.get('NEMOTRON_SFT_EPOCHS', '2' if WARM_START_ADAPTER_PATH else '1'))
        SFT_PER_DEVICE_BATCH = int(os.environ.get('NEMOTRON_SFT_PER_DEVICE_BATCH', '1'))
        SFT_GRAD_ACCUM = int(os.environ.get('NEMOTRON_SFT_GRAD_ACCUM', '16' if WARM_START_ADAPTER_PATH else '32'))
        TARGET_SAMPLES_PER_TASK = int(os.environ.get('NEMOTRON_TARGET_SAMPLES_PER_TASK', '1200'))
        PREFER_TONG_STYLE_DATA = os.environ.get('NEMOTRON_PREFER_TONG_STYLE_DATA', '1') == '1'

        SFT_LR = float(os.environ.get(
            'NEMOTRON_SFT_LR',
            '5e-5' if WARM_START_ADAPTER_PATH else '2e-4',
        ))
        SFT_MAX_GRAD_NORM = float(os.environ.get('NEMOTRON_SFT_MAX_GRAD_NORM', '1000000000.0'))

        ENABLE_OPTIONAL_GRPO = False
        GRPO_SAMPLE_SIZE = 1024
        GRPO_EPOCHS = 1
        GRPO_LR = 5e-6
        GRPO_NUM_GEN = 4
        GRPO_MAX_COMP_LEN = 256

        LOCAL_EVAL_SAMPLES = 48
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
            'PREFER_TONG_STYLE_DATA': PREFER_TONG_STYLE_DATA,
            'ENABLE_OPTIONAL_GRPO': ENABLE_OPTIONAL_GRPO,
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

        > Preferred path: continue training from the strong public Tinker adapter if it is available. Fallback path: attach a fresh LoRA with broader module coverage than the original notebook.
        """,
    )

    set_source(
        13,
        r"""
        if WARM_START_ADAPTER_PATH:
            print(f'Warm-starting from adapter: {WARM_START_ADAPTER_PATH}')
            model = PeftModel.from_pretrained(
                model,
                WARM_START_ADAPTER_PATH,
                is_trainable=True,
            )
        else:
            print('No warm-start adapter found. Attaching a fresh LoRA.')
            lora_config = LoraConfig(
                r=LORA_RANK,
                lora_alpha=32,
                target_modules=[
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'in_proj', 'out_proj', 'up_proj', 'down_proj', 'lm_head',
                ],
                lora_dropout=0.05,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()
        print('Adapter is ready for continued training')
        """,
    )

    set_source(
        14,
        """
        ## 7. Metric-Aligned Trace Formatting, Balanced SFT Data, and Optional External CoT

        > Key fixes here:
        > 1. Do not force `\\boxed{}` when the gold answer contains `{`, `}`, or `\\`, because the official metric parser truncates those cases.
        > 2. Balance task types so the model does not overfit the dominant bit-manipulation bucket.
        > 3. If an external CoT dataset is available, mix it in on top of the balanced gold-answer templated data.
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


        train_core = train_df[['prompt', 'answer', 'task_type']].copy()
        train_part, val_part = stratified_split_by_task(train_core, seed=42)
        train_balanced = balance_by_task(
            train_part,
            task_col='task_type',
            target_per_task=TARGET_SAMPLES_PER_TASK,
            seed=42,
        )

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
            'numeral_system.csv': 600,
            'gravity_physics.csv': 1200,
            'unit_conversion.csv': 1150,
            'text_decryption.csv': 1492,
            'bit_manipulation_including_wrong.csv': 1508,
            'bit_manipulation_synth_including_wrong_v2.csv': 500,
            'equation_numeric.csv': 535,
            'cryptarithm.csv': 69,
        }


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
                dup_rows = [dict(row) for row in rows if row['id'] in priority_ids]
                rows.extend(dup_rows)
                print(f'Priority duplicated Tong-style rows: {len(dup_rows)}')
            else:
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
                    target_per_task=min(1200, TARGET_SAMPLES_PER_TASK),
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

    nb["metadata"].setdefault("codex_note", {})
    nb["metadata"]["codex_note"]["generated_from"] = str(src)
    nb["metadata"]["codex_note"]["strategy"] = (
        "Warm-start Tinker adapter + Tong-style priority-weighted CoT mix + metric-safe SFT + optional targeted GRPO"
    )

    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote improved notebook to: {dst}")


if __name__ == "__main__":
    main()
