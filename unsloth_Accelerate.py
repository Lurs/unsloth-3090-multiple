#!/usr/bin/env python
# Modified for non-Docker environment
# Fine-tuning script using Unsloth, Accelerate, and transformers

import os
import sys
import gc
import glob
from pathlib import Path
from huggingface_hub.hf_file_system import HfFileSystem
from accelerate import Accelerator
# Use FastLanguageModel for standard Unsloth usage
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from transformers import TextStreamer
import torch

# --- Patch HfFileSystem.glob ---
_orig_glob = HfFileSystem.glob
def _glob_override(self, pattern, *args, **kwargs):
    if pattern.startswith("file://"):
        return glob.glob(pattern[len("file://"):])
    if os.path.isabs(pattern) or pattern.startswith('./') or pattern.startswith('../') or '/' in pattern:
         if os.path.exists(pattern) or '*' in pattern or '?' in pattern or '[' in pattern:
             return glob.glob(pattern)
    return _orig_glob(self, pattern, *args, **kwargs)
HfFileSystem.glob = _glob_override
print("[PID {}] Patched HfFileSystem.glob for local paths.".format(os.getpid()), flush=True)

# --- Critical Environment Variables ---
os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# Use current working directory + outputs instead of /app/output
OUTPUT_ROOT = Path.cwd() / "training_output"
OUTPUT_ROOT.mkdir(exist_ok=True)

# --- Early debug prints ---
print(f"[PID {os.getpid()}] Script start. Python version: {sys.version}", flush=True)
print(f"[PID {os.getpid()}] Current PWD: {os.getcwd()}", flush=True)
print(f"[PID {os.getpid()}] Output directory: {OUTPUT_ROOT}", flush=True)
print(f"[PID {os.getpid()}] TORCH_DISTRIBUTED_USE_DTENSOR: {os.environ.get('TORCH_DISTRIBUTED_USE_DTENSOR')}", flush=True)
print(f"[PID {os.getpid()}] CUDA_VISIBLE_DEVICES (from env): {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"[PID {os.getpid()}] ACCELERATE_USE_TP: {os.environ.get('ACCELERATE_USE_TP')}", flush=True)
LAUNCHER_RANK = os.environ.get('RANK', 'N/A_LAUNCHER_RANK')
LAUNCHER_LOCAL_RANK = os.environ.get('LOCAL_RANK', 'N/A_LOCAL_RANK')
LAUNCHER_WORLD_SIZE = os.environ.get('WORLD_SIZE', 'N/A_WORLD_SIZE')
print(f"[PID {os.getpid()}] Launcher Env: RANK={LAUNCHER_RANK}, LOCAL_RANK={LAUNCHER_LOCAL_RANK}, WORLD_SIZE={LAUNCHER_WORLD_SIZE}", flush=True)

# --- Import torch and apply aggressive DTensor patch ---
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported torch. Version: {torch.__version__}. CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] CUDA device count: {torch.cuda.device_count()}", flush=True)
    try:
        if LAUNCHER_LOCAL_RANK != 'N/A_LOCAL_RANK' and int(LAUNCHER_LOCAL_RANK) < torch.cuda.device_count():
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Current CUDA device (by torch.cuda.current_device()): {torch.cuda.current_device()}", flush=True)
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Name of current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
        else:
             print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] LOCAL_RANK check skipped.", flush=True)
    except Exception as e_cuda_print:
        print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Error printing CUDA device info early: {e_cuda_print}", flush=True)

# AGGRESSIVE DTENSOR PATCH
try:
    from torch.distributed.tensor import DTensor
    if hasattr(DTensor, "_op_dispatcher") and hasattr(DTensor._op_dispatcher, "sharding_propagator") and hasattr(DTensor._op_dispatcher.sharding_propagator, "propagate"):
        original_propagate = DTensor._op_dispatcher.sharding_propagator.propagate
        def _no_op_propagate(self_sharding_prop, op_info, *args, **kwargs):
            return op_info.output_sharding
        DTensor._op_dispatcher.sharding_propagator.propagate = _no_op_propagate
        print(f"✅ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Successfully patched DTensor propagate.", flush=True)
    else:
        print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Could not find DTensor attributes for patching.", flush=True)
except ImportError:
    print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] torch.distributed.tensor.DTensor not found. Patch skipped.", flush=True)
except Exception as e:
    print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Error during DTensor patching: {e}", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing accelerate...", flush=True)
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported accelerate.", flush=True)
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing Unsloth...", flush=True)
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported Unsloth.", flush=True)
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing Transformers & Datasets...", flush=True)
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported Transformers & Datasets.", flush=True)

# --- Hugging Face Login ---

login()

# --- Configuration matching your script ---
MODEL_NAME = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
MAX_SEQ_LENGTH = 40960
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
FULL_FINETUNING = True

LORA_R = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_BIAS = "none"
USE_GRADIENT_CHECKPOINTING = "unsloth"
LORA_RANDOM_STATE = 3407
USE_RSLORA = False
LOFTQ_CONFIG = None

DATASET_NAME = "LarsVI/botw_finetune1_with_val"

PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
MAX_STEPS = 30
LEARNING_RATE = 2e-4
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
SEED = 3407
REPORT_TO = "wandb"  # Change to "none" if you don't want W&B logging

# --- Chat Template ---
QWEN_CHAT_TEMPLATE = r"""{% macro render_item_list(item_list, tag_name='required') %}\n    {%- if item_list is defined and item_list is iterable and item_list | length > 0 %}\n        {%- if tag_name %}{{- '\\n<' ~ tag_name ~ '>' -}}{% endif %}\n            {{- '[' }}\n                {%- for item in item_list -%}\n                    {%- if loop.index > 1 %}{{- \", \"}}{% endif -%}\n                    {%- if item is string -%}\n                        {{ \"`\" ~ item ~ \"`\" }}\n                    {%- else -%}\n                        {{ item }}\n                    {%- endif -%}\n                {%- endfor -%}\n            {{- ']' }}\n        {%- if tag_name %}{{- '</' ~ tag_name ~ '>' -}}{% endif %}\n    {%- endif %}\n{% endmacro %}\n\n{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{%- if not tools is defined %}\n    {%- set tools = [] %}\n{%- endif %}\n\n{%- if system_message is defined %}\n    {{- \"<|im_start|>system\\n\" + system_message }}\n{%- else %}\n    {%- if tools is iterable and tools | length > 0 %}\n        {{- \"<|im_start|>system\\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.\" }}\n    {%- endif %}\n{%- endif %}\n{%- if tools is iterable and tools | length > 0 %}\n    {{- \"\\n\\nYou have access to the following functions:\\n\\n\" }}\n    {{- \"<tools>\" }}\n    {%- for tool in tools %}\n        {%- if tool.function is defined %}\n            {%- set tool = tool.function %}\n        {%- endif %}\n        {{- \"\\n<function>\\n<name>\" ~ tool.name ~ \"</name>\" }}\n        {{- '\\n<description>' ~ (tool.description | trim) ~ '</description>' }}\n        {{- '\\n<parameters>' }}\n        {%- for param_name, param_fields in tool.parameters.properties|items %}\n            {{- '\\n<parameter>' }}\n            {{- '\\n<name>' ~ param_name ~ '</name>' }}\n            {%- if param_fields.type is defined %}\n                {{- '\\n<type>' ~ (param_fields.type | string) ~ '</type>' }}\n            {%- endif %}\n            {%- if param_fields.description is defined %}\n                {{- '\\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}\n            {%- endif %}\n            {{- render_item_list(param_fields.enum, 'enum') }}\n            {%- set handled_keys = ['type', 'description', 'enum', 'required'] %}\n            {%- for json_key in param_fields.keys() | reject(\"in\", handled_keys) %}\n                {%- set normed_json_key = json_key | replace(\"-\", \"_\") | replace(\" \", \"_\") | replace(\"$\", \"\") %}\n                {%- if param_fields[json_key] is mapping %}\n                    {{- '\\n<' ~ normed_json_key ~ '>' ~ (param_fields[json_key] | tojson | safe) ~ '</' ~ normed_json_key ~ '>' }}\n                {%- else %}\n                    {{-'\\n<' ~ normed_json_key ~ '>' ~ (param_fields[json_key] | string) ~ '</' ~ normed_json_key ~ '>' }}\n                {%- endif %}\n            {%- endfor %}\n            {{- render_item_list(param_fields.required, 'required') }}\n            {{- '\\n</parameter>' }}\n        {%- endfor %}\n        {{- render_item_list(tool.parameters.required, 'required') }}\n        {{- '\\n</parameters>' }}\n        {%- if tool.return is defined %}\n            {%- if tool.return is mapping %}\n                {{- '\\n<return>' ~ (tool.return | tojson | safe) ~ '</return>' }}\n            {%- else %}\n                {{- '\\n<return>' ~ (tool.return | string) ~ '</return>' }}\n            {%- endif %}\n        {%- endif %}\n        {{- '\\n</function>' }}\n    {%- endfor %}\n    {{- \"\\n</tools>\" }}\n    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n{%- endif %}\n{%- if system_message is defined %}\n    {{- '<|im_end|>\\n' }}\n{%- else %}\n    {%- if tools is iterable and tools | length > 0 %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in loop_messages %}\n    {%- if message.role == \"assistant\" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}\n            {{- '\\n' + message.content | trim + '\\n' }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n            {%- if tool_call.arguments is defined %}\n                {%- for args_name, args_value in tool_call.arguments|items %}\n                    {{- '<parameter=' + args_name + '>\\n' }}\n                    {%- set args_value = args_value if args_value is string else args_value | string %}\n                    {{- args_value }}\n                    {{- '\\n</parameter>\\n' }}\n                {%- endfor %}\n            {%- endif %}\n            {{- '</function>\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"user\" or message.role == \"system\" or message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.previtem and loop.previtem.role != \"tool\" %}\n            {{- '<|im_start|>user\\n' }}\n        {%- endif %}\n        {{- '<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>\\n' }}\n        {%- if not loop.last and loop.nextitem.role != \"tool\" %}\n            {{- '<|im_end|>\\n' }}\n        {%- elif loop.last %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- else %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""

# --- Model Loading ---
def load_model(current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In load_model()...", flush=True)

    device_map_config = {"": current_accelerator.device}
    print(f"[PID {pid}, Rank {rank_idx}] Using device_map: {device_map_config}", flush=True)

    model_kwargs = {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": LOAD_IN_4BIT,
        "attn_implementation": "flash_attention_2",
        "device_map": device_map_config,
        # Let Unsloth handle attn_implementation and dtype with 4bit loading
    }

    print(f"[PID {pid}, Rank {rank_idx}] model_kwargs: {model_kwargs}", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Calling FastLanguageModel.from_pretrained...", flush=True)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        print(f"[PID {pid}, Rank {rank_idx}] FastLanguageModel.from_pretrained successful.", flush=True)
        print(f"[PID {pid}, Rank {rank_idx}] Model device after load: {model.device}", flush=True)
    except Exception as e_load:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during FastLanguageModel.from_pretrained: {e_load}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
    return model, tokenizer

# --- LoRA Application ---
def apply_lora(base_model, current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In apply_lora()...", flush=True)
    try:
        lora_model = FastLanguageModel.get_peft_model(
            base_model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
            random_state=LORA_RANDOM_STATE,
            use_rslora=USE_RSLORA,
            loftq_config=LOFTQ_CONFIG,
        )
        print(f"[PID {pid}, Rank {rank_idx}] apply_lora successful.", flush=True)
        return lora_model
    except Exception as e_lora:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during apply_lora: {e_lora}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

# --- Dataset Handling ---
def load_and_split_dataset(current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In load_and_split_dataset()...", flush=True)
    try:
        with current_accelerator.main_process_first():
             ds = load_dataset(DATASET_NAME, trust_remote_code=True)["train"]

        print(f"[PID {pid}, Rank {rank_idx}] Dataset loaded successfully.", flush=True)
        # Returning the same dataset for train/val as example. Adjust if needed.
        return ds, ds
    except Exception as e_dsload:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during load_and_split_dataset: {e_dsload}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

# --- Main Function ---
def main():
    pid = os.getpid()
    print(f"[PID {pid}, Pre-Accelerator-Rank] In main(). Initializing Accelerator...", flush=True)
    accelerator = Accelerator()
    rank_idx = accelerator.process_index
    print(f"[PID {pid}, Rank {rank_idx}] Accelerator initialized.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Loading model and tokenizer...", flush=True)
    model, tokenizer = load_model(accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] Model and tokenizer loaded.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Applying LoRA...", flush=True)
    model = apply_lora(model, accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] LoRA applied.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Loading dataset...", flush=True)
    train_ds_raw, val_ds_raw = load_and_split_dataset(accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] Dataset loaded.", flush=True)

    # --- Apply Chat Template ---
    print(f"[PID {pid}, Rank {rank_idx}] Applying custom QWEN_CHAT_TEMPLATE...", flush=True)
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    print(f"[PID {pid}, Rank {rank_idx}] Custom QWEN_CHAT_TEMPLATE applied.", flush=True)

    if tokenizer.pad_token is None:
        print(f"[PID {pid}, Rank {rank_idx}] Setting pad_token to eos_token.", flush=True)
        tokenizer.pad_token = tokenizer.eos_token

    def format_with_official_template(examples):
        texts = []
        for messages in examples['messages']:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False # Important for training data formatting
            )
            texts.append(formatted)
        return {"text": texts}

    # --- Format Datasets using the Template ---
    print(f"[PID {pid}, Rank {rank_idx}] Formatting datasets with custom chat template...", flush=True)
    with accelerator.main_process_first():
        train_ds = train_ds_raw.map(format_with_official_template, batched=True, remove_columns=train_ds_raw.column_names)
        val_ds = val_ds_raw.map(format_with_official_template, batched=True, remove_columns=val_ds_raw.column_names)
    print(f"[PID {pid}, Rank {rank_idx}] Datasets formatted.", flush=True)

    # --- Training Arguments (SFTConfig) ---
    print(f"[PID {pid}, Rank {rank_idx}] Initializing SFTConfig...", flush=True)
    try:
        sft_config = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim=OPTIM,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=SEED,
            output_dir=str(OUTPUT_ROOT / "training_outputs"),
            report_to=REPORT_TO,
        )
        print(f"[PID {pid}, Rank {rank_idx}] SFTConfig initialized.", flush=True)
    except Exception as e_config:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during SFTConfig initialization: {e_config}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    # --- Trainer Setup ---
    print(f"[PID {pid}, Rank {rank_idx}] Initializing SFTTrainer...", flush=True)
    SFT_DATASET_NUM_PROC = 1
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_num_proc=SFT_DATASET_NUM_PROC,
        )
        print(f"[PID {pid}, Rank {rank_idx}] SFTTrainer initialized successfully.", flush=True)
    except Exception as e_trainer:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during SFTTrainer initialization: {e_trainer}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    # --- Model Configuration Check (Main Process Only) ---
    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Pre-train config check.", flush=True)
        unwrapped_model_for_config = accelerator.unwrap_model(trainer.model)
        if (hasattr(unwrapped_model_for_config, "config") and
            hasattr(unwrapped_model_for_config.config, "use_cache") and
            getattr(unwrapped_model_for_config.config, "use_cache", False)):
            print(f"✅ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Forcing model.config.use_cache = False.", flush=True)
            unwrapped_model_for_config.config.use_cache = False
    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] All processes synchronized before training.", flush=True)

    # --- Training ---
    print(f"[PID {pid}, Rank {rank_idx}] Starting training...", flush=True)
    try:
        trainer_stats = trainer.train()
        print(f"[PID {pid}, Rank {rank_idx}] Training completed.", flush=True)
        print(f"[PID {pid}, Rank {rank_idx}] Training Stats: {trainer_stats}", flush=True)
    except Exception as e_train:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during training: {e_train}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    # --- Critical: Wait for ALL processes to finish training BEFORE main process starts saving ---
    print(f"[PID {pid}, Rank {rank_idx}] Waiting for all processes to finish training...", flush=True)
    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] All processes finished training and are synchronized.", flush=True)

    # --- Saving the model (Main Process Only) ---
    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving training artifacts...", flush=True)
        lora_save_path = OUTPUT_ROOT / "lora_model"

        try:
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving LoRA adapters and tokenizer...", flush=True)
            trainer.model.save_pretrained(str(lora_save_path))
            tokenizer.save_pretrained(str(lora_save_path))
            #print(f"[PID {pid}, Rank {rank_idx}] MAIN