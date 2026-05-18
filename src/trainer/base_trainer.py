"""
This module defines the core training routine for fine‑uning causal language
models with LowRank Adapters (LoRA).
"""


import os
import time
import torch
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from src.config import TrainingConfig
from src.data import ft_dataset_builder_map


class ThroughputVRAMCallback(TrainerCallback):
    """Tracks per-step wall time and token count across the full training
    step (forward + backward + optimizer) via on_step_begin / on_step_end.
    Note: VRAM tracking removed as per implementation requirements."""

    def __init__(self):
        self._step_start: float | None = None
        self._elapsed = 0.0
        self._tokens = 0
        self._hook_handle = None

    def _count_tokens(self, module, args, kwargs):
        # Prevent counting tokens during evaluation runs
        if not module.training:
            return

        attention_mask = kwargs.get("attention_mask")
        input_ids = kwargs.get("input_ids")
        
        # Fallback: HF sometimes passes input_ids as the first positional argument
        if input_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
            
        # Count local tokens
        if attention_mask is not None:
            local_tokens = attention_mask.sum().item()
        elif input_ids is not None:
            local_tokens = input_ids.numel()
        else:
            local_tokens = 0
            
        # Safely scale token count for multi-GPU training
        if dist.is_initialized():
            self._tokens += local_tokens * dist.get_world_size()
        else:
            self._tokens += local_tokens

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._hook_handle = model.register_forward_pre_hook(
                self._count_tokens, with_kwargs=True,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def on_step_begin(self, args, state, control, **kwargs):
        # Crucial: Wait for GPU to finish previous work before starting timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            # Crucial: Wait for the forward/backward/opt steps to actually finish on GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._elapsed += time.perf_counter() - self._step_start
            self._step_start = None

    def consume(self) -> tuple[int, float]:
        """Return accumulated (tokens, elapsed_seconds) and reset."""
        tokens, elapsed = self._tokens, self._elapsed
        self._tokens = 0
        self._elapsed = 0.0
        return tokens, elapsed


class BaseTrainer(Trainer):
    """
    A custom Trainer subclass that can dynamically load and train on either
    the MATH dataset, the BBQ dataset, or a combination of both.

    Parameters
    ----------
    config : TrainingConfig
        Hyperparameters and model/dataset paths controlling the training run.
    dataset_type : str, optional
        Which dataset to load for training.  Accepts ``"sql"``, ``"bbq"``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        # Persist configuration and dataset selection for later reference
        self.config = config
        self.dataset_type = config.dataset_type

        # Ensure reproducibility
        set_seed(config.seed)

        # Load tokenizer and set padding token to EOS if not already defined
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        # Load base language model and inject LoRA adapters
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, 
                                                     torch_dtype=torch.bfloat16,
                                                     device_map={"": "cuda"}, 
                                                     attn_implementation="sdpa")
        model.config.use_cache = False
        if config.mode == 'lora':
            lora_config = config.lora_config.get_lora_config()
            model = get_peft_model(model, lora_config)
        
        
        self.train_ds_builder = ft_dataset_builder_map[self.dataset_type.lower()](self.tokenizer)
        self.train_dataset = self.train_ds_builder.build_dataset()

        # Construct TrainingArguments.  Disable evaluation since we do not
        # include validation sets during fine‑tuning.
       
        # Call the base Trainer constructor with the prepared objects
        super().__init__(
            model=model,
            args=config.get_trainer_args(),
            train_dataset=self.train_dataset,
            tokenizer=tokenizer,
        )
        self._perf_callback = ThroughputVRAMCallback()
        self.add_callback(self._perf_callback)
        
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        loader = DataLoader(self.train_dataset, 
                            batch_size=self.config.per_device_batch_size, 
                            shuffle=True, 
                            collate_fn=self.train_ds_builder.collate_fn,
                            num_workers=self.args.dataloader_num_workers,
                            pin_memory=self.args.dataloader_pin_memory)
   
        return loader

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        cb: ThroughputVRAMCallback | None = getattr(self, '_perf_callback', None)
        if cb is not None:
            tokens, elapsed = cb.consume()
            if elapsed > 0:
                logs["throughput_tok/s"] = round(tokens / elapsed, 1)
                num_steps = max(self.args.logging_steps, 1)
                logs["avg_step_time_ms"] = round(elapsed / num_steps * 1000, 1)

        if torch.cuda.is_available():
            logs["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            logs["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
            logs["vram_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

        super().log(logs, start_time)

    def merge_and_save_final_model(self, output_dir: str | None = None) -> None:
        """
        Persist the trained model and tokenizer to disk.

        LoRA adapters are merged into the base model before saving, ensuring the
        checkpoint contains all weights needed for inference.
        If output_dir is None, defaults to self.args.output_dir.

        Without merging, PEFT's save_pretrained() only saves the adapter weights,
        not the base model.
        """
        target_dir = output_dir or self.args.output_dir

        # If the model is a PEFT model, merge LoRA weights into the base model.
        model_to_save = self.model
        if hasattr(self.model, "merge_and_unload"):
            try:
                model_to_save = self.model.merge_and_unload()
            except Exception:
                model_to_save = self.model

        # Save the full (merged) model and tokenizer.
        if not hasattr(model_to_save, "_tied_weights_keys") or model_to_save._tied_weights_keys is None:
            model_to_save._tied_weights_keys = []

        key_to_tie = 'language_model.lm_head.weight'
        if key_to_tie not in model_to_save._tied_weights_keys:
            model_to_save._tied_weights_keys.append(key_to_tie)
            print(f"   Appended '{key_to_tie}' to _tied_weights_keys.")
        model_to_save.save_pretrained(target_dir)
        
        self.tokenizer.save_pretrained(target_dir)
