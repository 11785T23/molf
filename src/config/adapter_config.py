import os, torch
from dataclasses import dataclass, field
from peft import LoraConfig
from typing import List, Optional, Tuple

@dataclass(frozen=False)
class BaseModuleConfig:
    bias: str = "none"
    exclude_target_modules: tuple[str, ...] = field(default=("lm_head",),)
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    )
    task_type: str = "CAUSAL_LM"
    copy_weight: bool = False

@dataclass(frozen=False)
class BaseLoraConfig(BaseModuleConfig):

    rank: int = 32
    lora_alpha: int = 64                            # 2 * r
    lora_dropout: float = 0.1
    # # Nice-to-have for stability with higher ranks:
    use_rslora: bool = False
    all_linear: bool = True

    def get_lora_config(
        self,
    ) -> LoraConfig:
        common_kwargs = dict(
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            task_type=self.task_type,
            use_rslora=self.use_rslora,
            target_modules=self.target_modules,
        )
        return LoraConfig(**common_kwargs)
    
@dataclass(frozen=False)
class MoLFConfig(BaseModuleConfig):
    bias: str = "all"
    lora_alphas: tuple[int, ...] = (32, 32)
    lora_experts_ranks: tuple[int, ...] = (16, 128)
    lora_dropout: float = 0.1
    use_rslora: bool = True
    all_linear: bool = True
    bias_zero_weight_decay: bool = True
    norm_zero_weight_decay: bool = True