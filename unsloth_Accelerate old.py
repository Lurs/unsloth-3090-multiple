#!/usr/bin/env python
import os
import sys # For flushing output
import gc

# --- Critical Environment Variables (set BEFORE torch import) ---
os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL" # Already set
os.environ["NCCL_DEBUG"] = "INFO" # Set to WARN or INFO
# For more verbose NCCL, uncomment next line. Can be very noisy.
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
# For CUDA errors, makes them synchronous. Slows things down significantly. Use if suspecting CUDA misbehavior.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# --- Early debug prints ---
print(f"[PID {os.getpid()}] Script start. Python version: {sys.version}", flush=True)
print(f"[PID {os.getpid()}] Current PWD: {os.getcwd()}", flush=True)
print(f"[PID {os.getpid()}] TORCH_DISTRIBUTED_USE_DTENSOR: {os.environ.get('TORCH_DISTRIBUTED_USE_DTENSOR')}", flush=True)
print(f"[PID {os.getpid()}] CUDA_VISIBLE_DEVICES (from env): {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"[PID {os.getpid()}] ACCELERATE_USE_TP: {os.environ.get('ACCELERATE_USE_TP')}", flush=True)
# RANK, LOCAL_RANK etc are set by accelerate launcher
LAUNCHER_RANK = os.environ.get('RANK', 'N/A_LAUNCHER_RANK')
LAUNCHER_LOCAL_RANK = os.environ.get('LOCAL_RANK', 'N/A_LOCAL_RANK')
LAUNCHER_WORLD_SIZE = os.environ.get('WORLD_SIZE', 'N/A_WORLD_SIZE')
print(f"[PID {os.getpid()}] Launcher Env: RANK={LAUNCHER_RANK}, LOCAL_RANK={LAUNCHER_LOCAL_RANK}, WORLD_SIZE={LAUNCHER_WORLD_SIZE}", flush=True)

# --- Import torch and apply aggressive DTensor patch ---
import torch
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported torch. Version: {torch.__version__}. CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] CUDA device count: {torch.cuda.device_count()}", flush=True)
    # Note: LOCAL_RANK might not be the final device index if CUDA_VISIBLE_DEVICES is used to remap.
    # torch.cuda.current_device() will give the actual device index relative to the process.
    try:
        # This will be the actual GPU index for this process (e.g., 0, 1, 2 if CUDA_VISIBLE_DEVICES was "0,1,2")
        # Or just 0 if CUDA_VISIBLE_DEVICES was e.g. "2" for a specific process.
        # This depends on how accelerate sets CUDA_VISIBLE_DEVICES for each process.
        # Typically, each process sees its assigned GPU as device 0.
        if LAUNCHER_LOCAL_RANK != 'N/A_LOCAL_RANK' and int(LAUNCHER_LOCAL_RANK) < torch.cuda.device_count():
            # This assumes accelerate makes each process see its assigned GPU as cuda:0
            # Let's verify this understanding
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Current CUDA device (by torch.cuda.current_device()): {torch.cuda.current_device()}", flush=True)
            print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Name of current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
        else:
             print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] LOCAL_RANK ({LAUNCHER_LOCAL_RANK}) not valid for device_count check, or CUDA not fully initialized by accelerate yet for this print.", flush=True)

    except Exception as e_cuda_print:
        print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Error printing CUDA device info early: {e_cuda_print}", flush=True)


# AGGRESSIVE DTENSOR PATCH
try:
    from torch.distributed.tensor import DTensor
    if hasattr(DTensor, "_op_dispatcher") and \
       hasattr(DTensor._op_dispatcher, "sharding_propagator") and \
       hasattr(DTensor._op_dispatcher.sharding_propagator, "propagate"):

        original_propagate = DTensor._op_dispatcher.sharding_propagator.propagate
        def _no_op_propagate(self_sharding_prop, op_info, *args, **kwargs): # op_info is the OpInfo object
            # print(f"[PID {os.getpid()}, Rank {os.environ.get('RANK', 'N/A')}] DTensor _no_op_propagate called for op: {op_info.schema.name}", flush=True)
            return op_info.output_sharding # Pass through, returning the default/expected output_sharding

        DTensor._op_dispatcher.sharding_propagator.propagate = _no_op_propagate
        print(f"✅ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Successfully patched DTensor._op_dispatcher.sharding_propagator.propagate.", flush=True)
    else:
        print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Could not find DTensor attributes for patching.", flush=True)
except ImportError:
    print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] torch.distributed.tensor.DTensor not found. Patch skipped.", flush=True)
except Exception as e:
    print(f"⚠️ [PID {os.getpid()}, Rank {LAUNCHER_RANK}] Error during DTensor patching: {e}", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing accelerate...", flush=True)
from accelerate import Accelerator
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported accelerate.", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing Unsloth...", flush=True)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported Unsloth.", flush=True)

print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Importing Transformers & Datasets...", flush=True)
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
print(f"[PID {os.getpid()}, Rank {LAUNCHER_RANK}] Imported Transformers & Datasets.", flush=True)

# --- Configuration ---
MODEL_PATH = "/media/user/Qwen3-32B"
MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ", 8192))
LORA_R = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LOCAL_JSONL = "/media/user/qwen_combined_valid_prompts.jsonl"
TEST_SPLIT_RATIO = 0.1238
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MAX_STEPS = 30
LEARNING_RATE = 2e-4

# --- Model Loading ---
def load_model(current_accelerator):
    rank_idx = current_accelerator.process_index
    pid = os.getpid()
    print(f"[PID {pid}, Rank {rank_idx}] In load_model()...", flush=True)
    LOAD_IN_4BIT = True

    device_map_config = {"": current_accelerator.device}
    print(f"[PID {pid}, Rank {rank_idx}] Using device_map: {device_map_config}", flush=True)

    model_kwargs = {
        "model_name": MODEL_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": LOAD_IN_4BIT,
        "attn_implementation": "flash_attention_2",
        "device_map": device_map_config,
        "dtype" : torch.bfloat16, # <--- ADD THIS
    }

    # REMOVED the specific bnb args block as it's handled by dtype
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
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
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
        # Ensure dataset loading happens on main process only if it's not safe for multiple processes
        # or if it involves downloads. load_dataset is generally safe.
        # with current_accelerator.main_process_first(): # Example if download/caching is an issue
        #     ds = load_dataset("json", data_files={"train": LOCAL_JSONL})["train"]
        ds = load_dataset("json", data_files={"train": LOCAL_JSONL}, trust_remote_code=True)["train"] # Added trust_remote_code
        splits = ds.train_test_split(test_size=TEST_SPLIT_RATIO, seed=42)
        print(f"[PID {pid}, Rank {rank_idx}] load_and_split_dataset successful.", flush=True)
        return splits["train"], splits["test"]
    except Exception as e_dsload:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during load_and_split_dataset: {e_dsload}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

# --- Main Function ---
def main():
    pid = os.getpid() # PID for prints before accelerator is fully initialized
    print(f"[PID {pid}, Pre-Accelerator-Rank] In main(). Initializing Accelerator...", flush=True)
    # Initialize accelerator here
    accelerator = Accelerator()
    rank_idx = accelerator.process_index # Now use accelerator's rank
    print(f"[PID {pid}, Rank {rank_idx}] Accelerator initialized. Distributed: {accelerator.distributed_type}, Device: {accelerator.device}, Num_processes: {accelerator.num_processes}", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Loading model and tokenizer...", flush=True)
    model, tokenizer = load_model(accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] Model and tokenizer loaded.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Applying LoRA...", flush=True)
    model = apply_lora(model, accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] LoRA applied.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Loading and splitting dataset...", flush=True)
    train_ds_raw, val_ds_raw = load_and_split_dataset(accelerator)
    print(f"[PID {pid}, Rank {rank_idx}] Dataset loaded and split.", flush=True)

    if tokenizer.pad_token is None:
        print(f"[PID {pid}, Rank {rank_idx}] Setting pad_token to eos_token.", flush=True)
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None: # Should not happen for Qwen3
        print(f"⚠️ [PID {pid}, Rank {rank_idx}] EOS token not set. Using default.", flush=True)
        tokenizer.eos_token = "<|endoftext|>"

    train_ds = train_ds_raw
    val_ds = val_ds_raw

    print(f"[PID {pid}, Rank {rank_idx}] Getting chat template...", flush=True)
    #tokenizer = get_chat_template(
    #    tokenizer,
    #    chat_template="qwen3", # Qwen3 should be correct
    #    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}, # Check Unsloth docs for Qwen3 exact mapping
    #)
    if tokenizer.pad_token is None: # Re-check after get_chat_template
        print(f"[PID {pid}, Rank {rank_idx}] Re-setting pad_token to eos_token post chat_template.", flush=True)
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[PID {pid}, Rank {rank_idx}] Chat template applied.", flush=True)

    # Dataset mapping: Using fewer num_proc for debugging.
    # Can be slow without num_proc, but safer for finding errors.
    # Consider dataset_num_proc = 0 or 1 initially.
    DATASET_MAP_NUM_PROC = 1 # Reduced for debugging
    print(f"[PID {pid}, Rank {rank_idx}] Passing through train_ds text as-is (num_proc={DATASET_MAP_NUM_PROC})...", flush=True)
    # Ensure the 'text' column exists after these maps
    train_ds = train_ds.map(lambda ex: {"text": ex["text"]}, num_proc=DATASET_MAP_NUM_PROC, remove_columns=[col for col in train_ds.features if col != 'text'])

    print(f"[PID {pid}, Rank {rank_idx}] Passing through val_ds text as-is (batched=True)...", flush=True)
    val_ds = val_ds.map(lambda batch: {"text": batch["text"]}, batched=True, num_proc=DATASET_MAP_NUM_PROC, remove_columns=[col for col in val_ds.features if col != 'text'])

    print(f"[PID {pid}, Rank {rank_idx}] Datasets processed.", flush=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False, # Usually False for LoRA with Unsloth
        bf16=True,
        fp16=False,
    )
    print(f"[PID {pid}, Rank {rank_idx}] TrainingArguments initialized.", flush=True)

    print(f"[PID {pid}, Rank {rank_idx}] Initializing SFTTrainer...", flush=True)
    # SFTTrainer's dataset_num_proc is for tokenization by the trainer
    SFT_DATASET_NUM_PROC = 1 # Reduced for debugging
    trainer = SFTTrainer(
        model=model, # Accelerate will handle DDP wrapping internally via prepare()
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=SFT_DATASET_NUM_PROC,
        packing=False,
        args=training_args,
    )
    print(f"[PID {pid}, Rank {rank_idx}] SFTTrainer initialized. Model is on: {trainer.model.device}", flush=True)

    # Model might be DDP wrapped by SFTTrainer's __init__ (via accelerator.prepare)
    # Access config on unwrapped model only for main process check
    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Pre-train check for model.config.use_cache.", flush=True)
        unwrapped_model_for_config = accelerator.unwrap_model(trainer.model)
        if hasattr(unwrapped_model_for_config, "config") and getattr(unwrapped_model_for_config.config, "use_cache", False):
            print(f"✅ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Forcing model.config.use_cache = False on unwrapped model.", flush=True)
            unwrapped_model_for_config.config.use_cache = False
            # Apply to the wrapped model as well if needed, although trainer should handle this
            # if hasattr(trainer.model, "config"):
            #     trainer.model.config.use_cache = False # This might not work on DDP object

    # Wait for all processes to complete initialization and configuration before starting training
    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] All processes ready. Calling trainer.train()...", flush=True)

    try:
        metrics = trainer.train()
        print(f"[PID {pid}, Rank {rank_idx}] trainer.train() completed.", flush=True)
    except Exception as e_train:
        print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] ERROR during trainer.train(): {e_train}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1) # Exit the failing process

    # --- Critical: Wait for all processes to finish training BEFORE main process starts saving ---
    # The error occurred here previously because Rank 0 started saving while others were waiting.
    accelerator.wait_for_everyone()
    print(f"[PID {pid}, Rank {rank_idx}] All processes finished training and synchronized.", flush=True)

    # --- Saving logic moved AFTER the final barrier ---
    if accelerator.is_main_process:
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Training finished. Saving artifacts...", flush=True)
        lora_adapter_path = "lora_adapters_final"
        merged_model_16bit_path = "merged_model_16bit"
        full_merged_model_path = "full_merged_model" # Using directory name as before

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving LoRA adapters to {lora_adapter_path}...", flush=True)
        # Save LoRA adapters from the trainer's model (which is the wrapped PEFT model)
        try:
            # PeftModel's save_pretrained can handle DDP wrapped models on the main process
            trainer.model.save_pretrained(lora_adapter_path)
            tokenizer.save_pretrained(lora_adapter_path)
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: LoRA adapters saved.", flush=True)
        except Exception as e_lora_save:
             print(f"⚠️ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Error saving LoRA adapters: {e_lora_save}", flush=True)
             import traceback
             traceback.print_exc(file=sys.stdout)
             sys.stdout.flush()

        # --- Start of more aggressive VRAM clearing + original saving logic ---
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Preparing for full save. Unwrapping model...", flush=True)
        unwrapped_model_for_save = accelerator.unwrap_model(trainer.model)

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Moving unwrapped model to CPU...", flush=True)
        # This is the original step that might cause issues with Unsloth's fast_dequantize
        try:
            unwrapped_model_for_save = unwrapped_model_for_save.to("cpu")
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Unwrapped model moved to CPU.", flush=True)
        except Exception as e_cpu_move:
            print(f"⚠️ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Could not move unwrapped model to CPU: {e_cpu_move}", flush=True)
            # Decide if this should be a fatal error or just a warning. Let's print and continue to merge/save attempt.
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()


        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Forcing GC + emptying CUDA cache...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Merging LoRA into base model (on CPU)...", flush=True)
        try:
            unwrapped_model_for_save.merge_and_unload()
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Merge successful.", flush=True)
        except Exception as e_merge:
            print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] MAIN PROCESS: ERROR during merge_and_unload on CPU: {e_merge}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            # Raise here as merging is critical
            raise

        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving merged 16-bit model to {merged_model_16bit_path}...", flush=True)
        try:
            unwrapped_model_for_save.save_pretrained_merged(merged_model_16bit_path, tokenizer, save_method="merged_16bit")
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Merged 16-bit model saved.", flush=True)
        except Exception as e_save_16bit:
             print(f"⚠️ [PID {pid}, Rank {rank_idx}] MAIN PROCESS: Error saving merged 16-bit model: {e_save_16bit}", flush=True)
             import traceback
             traceback.print_exc(file=sys.stdout)
             sys.stdout.flush()


        # Attempt the GGUF save using the original approach (save_pretrained after merge on CPU)
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Saving model to GGUF (from CPU copy) to {full_merged_model_path}...", flush=True)
        try:
            # This is the call that caused the 'fast_dequantize on CPU' warnings
            # It attempts to save the merged model, which Unsloth intercepts for GGUF conversion
            # if the model is a FastLanguageModel and the output directory indicates GGUF is desired
            # or if save_pretrained_gguf was explicitly called (which we are reverting).
            # The underlying issue might be how Unsloth's GGUF conversion handles a model already explicitly moved to CPU.
            unwrapped_model_for_save.save_pretrained(full_merged_model_path) # This should trigger GGUF save in Unsloth
            tokenizer.save_pretrained(full_merged_model_path) # Save tokenizer alongside GGUF
            print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Model saved to GGUF via save_pretrained.", flush=True)
        except Exception as e_gguf:
            print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] MAIN PROCESS: FATAL ERROR during GGUF save via save_pretrained: {e_gguf}", flush=True)
            print(f"🔥🔥🔥 [PID {pid}, Rank {rank_idx}] MAIN PROCESS: This is the step that previously failed. Check Unsloth/Torch/CUDA compatibility.", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            # Allow script to potentially finish for other ranks' outputs, but exit with error
            sys.exit(1)


        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Deleting trainer and model references to free VRAM...", flush=True)
        # These were already deleted conceptually earlier, but ensure they are gone.
        # The unwrapped_model_for_save reference still exists, we should delete that too.
        del trainer # Should be None or already deleted
        del model   # Should be None or already deleted
        del unwrapped_model_for_save # Delete the reference to the model on CPU
        
        print(f"[PID {pid}, Rank {rank_idx}] MAIN PROCESS: Forcing garbage collection and emptying CUDA cache... (Post-Save)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        # --- End of reverted saving logic ---

    # No need for a final barrier after saving, as only the main process does it.
    # The script will exit naturally.

    print(f"[PID {pid}, Rank {rank_idx}] Script finished successfully for this process.", flush=True)
    return metrics if 'metrics' in locals() else None
if __name__ == "__main__":
    main_pid = os.getpid()
    print(f"[PID {main_pid}] Script __main__ started.", flush=True)
    try:
        results = main()
        # This final print will only be from processes that complete main()
        # The Accelerator object in main() goes out of scope.
        # For a final status, rely on the prints from within main() and accelerator's handling of processes.
        # If we need a global "all done" from main rank:
        try:
            temp_accelerator_check = Accelerator() # Create a new one just for this check
            if temp_accelerator_check.is_main_process:
                print(f"[PID {main_pid}, MainRank] __main__: Training complete. Metrics: {results}", flush=True)
            # The temp_accelerator_check will clean up when it goes out of scope
        except Exception as e_temp_accel:
             print(f"⚠️ [PID {main_pid}] Could not create temp accelerator for final print: {e_temp_accel}", flush=True)

    except Exception as e_main_fatal:
        print(f"🔥🔥🔥 [PID {main_pid}] FATAL ERROR in __main__ execution: {e_main_fatal}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1) # Ensure non-zero exit code for accelerate
    print(f"[PID {main_pid}] Script __main__ exiting normally.", flush=True)