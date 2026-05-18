"""
This module defines a dataclass that bundles together the various
hyperparameters and settings needed for finetuning a causal language model
using parameterefficient LoRA adapters.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Literal, Tuple
from transformers import TrainingArguments
from .adapter_config import BaseLoraConfig, MoLFConfig
from .lora_param_heuristic import get_lora_params


@dataclass(frozen=False)
class TrainingConfig:
    """Container for training hyperparameters and LoRA settings."""

    # Name or path of the pretrained model to fine‑tune.  When training
    # Qwen‑2.5 models, provide the appropriate identifier from the Hub.
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    only_eval: bool = False

    # Maximum sequence length for tokenization.  Examples longer than this
    # value will be truncated on the right.  Adjust based on GPU memory.
    # max_seq_length: int = 4096

    # Learning rate used by the AdamW optimizer.
    learning_rate: float = 2e-5
    lora_learning_rate: float = 5e-5
    weight_decay: float = 0.0
    lora_weight_decay: float = 0.0

    # Total number of training epochs.
    num_train_epochs: int = 1
    lr_scheduler_type: str = "cosine"
    warm_up_ratio: float = 0.05

    per_device_batch_size: int = 2

    gradient_accumulation_steps: int = 4

    mode: Literal['fft', 'lora', 'molf'] = 'lora'

    # Directory where checkpoints and the final model will be saved.
    ckpt_dir_root: str = "./ckpts/baseline"

    # Random seed for reproducibility.
    seed: int = 42
    
    # Logging
    log_dir_root: str = "logs"
    logging_steps: int = 2
    report_to: Literal["wandb", "json"] = "wandb"
    
    # Evaluation
    evaluation_strategy: str = "no"
    
    # checkpointing
    save_strategy: str = "no"
    save_steps: int = 200
    save_total_limit: int = 1
    clean_ckpt_at_end: bool = False
    
    # fine-tuning dataset type — one of {"sql", "med", "fact"}
    dataset_type: str = "sql"
    
    # LoRA Config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_rslora: bool = True
    lora_effective_scale: int = 2
    
    # MoLF Config
    molf_lora_ranks: tuple[int, ...] = (128,)
    # Routing score function: 'true_projected' (EPD, paper Eq. 10, default) or
    # 'projected' (PFN, paper Eq. 11, ablation comparator).
    molf_score_fn: str = 'true_projected'
    molf_topk: int = 1
    
    # pre-init
    lora_config: Any = field(default=None, repr=False)
    run_name: str = field(default="", init=False)
    log_dir: str = field(default="", init=False)
    ckpt_dir: str = field(default="", init=False)
    molf_config: Any = field(default=None, repr=False)
    
    def __post_init__(self):
        model_name = (self.model_name_or_path.split('/')[-1]).replace('.', 'd')
        if self.mode == 'lora':
            self.weight_decay = self.lora_weight_decay
            self.learning_rate = self.lora_learning_rate
            self.lora_alpha, self.lora_dropout = get_lora_params(
                lora_rank=self.lora_rank, use_rslora=self.use_rslora,
                scale=self.lora_effective_scale,
            )
            self.lora_config = BaseLoraConfig(
                rank=self.lora_rank, lora_alpha=self.lora_alpha, all_linear=False,
                lora_dropout=self.lora_dropout, use_rslora=self.use_rslora,
            )
            rs_tag = "_rs" if self.use_rslora else ""
            self.run_name = f"baseline_lftr={self.lora_config.rank}{rs_tag}_{model_name}_{self.dataset_type}"
        elif self.mode == 'fft':
            self.run_name = f"baseline_fft_{model_name}_{self.dataset_type}"
        elif self.mode == 'molf':
            self.molf_config = MoLFConfig(
                lora_alphas=tuple([self.lora_alpha for _ in self.molf_lora_ranks]),
                lora_experts_ranks=self.molf_lora_ranks,
                lora_dropout=self.lora_dropout,
                use_rslora=self.use_rslora,
            )
            self.run_name = f"molf_{model_name}_{self.dataset_type}_k={self.molf_topk}_scf={self.molf_score_fn}"
        else:
            raise ValueError(f"[mode]{self.mode} not a valid fine-tuning Mode")

        if self.mode == 'molf':
            lr_str = str(self.learning_rate).replace('.', 'd')
            lora_lr_str = str(self.lora_learning_rate).replace('.', 'd')
            self.run_name += f"_lr={lr_str}_llr={lora_lr_str}"
        else:
            lr_str = str(self.learning_rate).replace('.', 'd')
            self.run_name += f"_{lr_str}"
            
        wu_str = str(self.warm_up_ratio).replace('.', 'd')
        self.run_name += f"_{self.lr_scheduler_type}{wu_str}"
        
        os.makedirs(self.log_dir_root, exist_ok=True)
        os.makedirs(self.ckpt_dir_root, exist_ok=True)
        
        self.log_dir = f"{self.log_dir_root}/{self.run_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.ckpt_dir = f"{self.ckpt_dir_root}/{self.run_name}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
    
    def get_trainer_args(self):
        print("batch_size:", self.per_device_batch_size)
        training_args = TrainingArguments(
            run_name=self.run_name,
            do_train=True,
            do_eval=False,
            do_predict=False,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            
            logging_dir=self.log_dir, 
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            report_to=self.report_to,
            
            lr_scheduler_type=self.lr_scheduler_type, 
            warmup_ratio=self.warm_up_ratio,
            

            
            output_dir=self.ckpt_dir,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.evaluation_strategy,
            
            dataloader_pin_memory=True,
            dataloader_num_workers=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,                     # mixed precision autocast in bf16
            # bf16_full_eval=True,           # eval/generation also in bf16
            # optim="adamw_torch_fused",     # good fused optimizer on recent PyTorch
            tf32=True,    
            
            # DDP
            ddp_find_unused_parameters=(self.mode!='molf')
        )
        return training_args
         
    