import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    SchedulerType,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
from typing import Optional

from src.model.molf import (
    MixtureOfLoRAFull, 
    wrap_model_with_molf, 
    merge_molf_and_unwrap
)
from src.data import ft_dataset_builder_map
from src.optim import (
    get_param_group_molf, MoLFAdamW, SCORE_F_MAP
)
from .base_trainer import BaseTrainer, ThroughputVRAMCallback
from src.config import TrainingConfig
from accelerate import Accelerator
from accelerate.logging import get_logger

accelerator = Accelerator()
logger = get_logger(__name__)

class MoLFTrainer(BaseTrainer):
    """
    Trainer for MixtureOfLoRAFull (MoLF) models.
    Overrides model wrapping, optimizer creation, and saving logic.
    """

    def __init__(self, config: TrainingConfig) -> None:
        assert config.mode == 'molf', "MoLFTrainer only work with 'molf' mode"
        # standard setup (same as BaseTrainer)
        self.config = config
        self.dataset_type = config.dataset_type
        set_seed(config.seed)
        
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        # model Loading & MoLF Wrapping
        # load base model in bfloat16
        is_gemma = "gemma" in config.model_name_or_path.lower()
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, 
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"}, 
            attn_implementation="eager" if is_gemma else "sdpa",
        )
        model.config.use_cache = False
        if is_gemma:
            model.generation_config.use_cache = False
            model.generation_config.cache_implementation = None
        # wrap molf
        logger.info("Wrapping model with MixtureOfLoRAFull...")
        model = wrap_model_with_molf(model, config.molf_config)
        
        # get score function
        assert self.config.molf_score_fn in SCORE_F_MAP, f"{self.config.molf_score_fn} not a score function implemented in src/optim/molf_metric.py"
        self.molf_score_fn = SCORE_F_MAP[self.config.molf_score_fn]
        
        # dataset
        self.train_ds_builder = ft_dataset_builder_map[self.dataset_type.lower()](self.tokenizer)
        self.train_dataset = self.train_ds_builder.build_dataset()

        # initialize Parent (Trainer)
        # Note: We pass the wrapped model here.
        Trainer.__init__(
            self,
            model=model,
            args=config.get_trainer_args(),
            train_dataset=self.train_dataset,
            tokenizer=tokenizer,
        )
        
        # Throughput and latency tracking
        self._perf_callback = ThroughputVRAMCallback()
        self.add_callback(self._perf_callback)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a specific group of parameters with different learning rates and weight decay
        assignments for base weights vs experts.
        """
        if self.optimizer is None:
            print("Creating custom MoLFAdamW optimizer...")
            
            decay_parameters = get_param_group_molf(
                self.model,
                lr_fft=self.config.learning_rate,
                lr_lora=self.config.lora_learning_rate,
                weight_decay_fft=self.config.weight_decay,
                weight_decay_lora=self.config.lora_weight_decay,
                bias_zero_weight_decay=self.config.molf_config.bias_zero_weight_decay,
                norm_zero_weight_decay=self.config.molf_config.norm_zero_weight_decay
            )
            
            self.optimizer = MoLFAdamW(
                decay_parameters,
                score_func=self.molf_score_fn,
                lr=self.config.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.config.weight_decay,
                molf_topk=self.config.molf_topk,
                log_file_path=self.config.ckpt_dir + "/molf_expert_selection.jsonl"
            )
            
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type in SchedulerType:
                # We just use the standard HF implementation
                super().create_scheduler(num_training_steps, optimizer)
            else:
                # TODO implement non-uniform scheduler is needed
                raise NotImplementedError(f"{self.args.lr_scheduler_type} is not implemented for yet")     
        return self.lr_scheduler

    def merge_and_save_final_model(self, output_dir: Optional[str] = None) -> None:
        """
        Persist the trained model and tokenizer to disk.
        Uses custom merge_molf_and_unwrap to handle MoLF modules.
        """
        target_dir = output_dir or self.args.output_dir
        logger.info(f"Merging MoLF weights and saving to {target_dir}...")

        # unwrap/merge
        # This modifies self.model in-place to be a standard nn.Module with merged weights
        # Note: If using FSDP, ensure weights are gathered first (Trainer handles this mostly, 
        # but manual unwrapping on FSDP requires care. For Single/DDP this is fine).
        try:
            merged_model = merge_molf_and_unwrap(self.model)
        except Exception as e:
            logger.error(f"Error during merge: {e}")
            # Fallback to saving raw model if merge fails
            merged_model = self.model
        logger.info(f"Model Saved at {merged_model}")
        # save
        if not hasattr(merged_model, "_tied_weights_keys") or merged_model._tied_weights_keys is None:
            merged_model._tied_weights_keys = []

        key_to_tie = 'language_model.lm_head.weight'
        if key_to_tie not in merged_model._tied_weights_keys:
            merged_model._tied_weights_keys.append(key_to_tie)
            print(f"   Appended '{key_to_tie}' to _tied_weights_keys.")
        merged_model.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)