import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math, warnings
from src.config.adapter_config import MoLFConfig
from src.utils import get_parent_and_attr

class MixtureOfLoRAFull(nn.Module):
    """
    Module to forward using both full-finetuning path and low rank adaption path
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        lora_alphas: Tuple[int] = (32, 32),
        lora_experts_ranks: Tuple[int] = (16, 128),
        use_rslora: bool = True,
        lora_dropout: float = 0.0,
        copy_weight: bool = False,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        assert len(lora_experts_ranks)>0 and max(lora_experts_ranks)>0, f"Invalid lora_experts_ranks: {lora_experts_ranks}"
        assert len(lora_experts_ranks) == len(lora_alphas), f"len(lora_experts_ranks) != len(lora_alphas): {len(lora_experts_ranks)} != {len(lora_alphas)}"
        # Rank Management
        self.lora_experts_ranks = lora_experts_ranks
        self.rs_lora = use_rslora
        self.lora_alphas = lora_alphas
        self.merged = False
        self.freeze_base = freeze_base

        self.device = base_linear.weight.device
        self.dtype = base_linear.weight.dtype

        # Base linear (copy or link)
        if copy_weight:
            self.base_layer = MixtureOfLoRAFull.copy_linear(lin=base_linear)
        else:
            self.base_layer = base_linear

        if freeze_base:
            for p in self.base_layer.parameters():
                p.requires_grad = False
        
        # List of lora experts
        self.lora_experts_A = nn.ModuleList()
        self.lora_experts_B = nn.ModuleList()
        self.scaling_factors = []
        for alpha, r in zip(self.lora_alphas, self.lora_experts_ranks):
            self.scaling_factors.append(self.scaling_for_r(alpha, r=r))
            lora_A = nn.Linear(self.in_features, r, bias=False, 
                               device=self.device, dtype=self.dtype)
            lora_B = nn.Linear(r, self.out_features, bias=False, 
                               device=self.device, dtype=self.dtype)
            self.lora_experts_A.append(lora_A)
            self.lora_experts_B.append(lora_B)
        self.dropout_p = lora_dropout

        self.reset_lora_parameters()
    
    def scaling_for_r(self, alpha, r: int) -> float:
        if r == 0: return 0
        if self.rs_lora:
            return alpha / math.sqrt(r)
        else:
            return alpha / r
        
    def reset_lora_parameters(self, optimizer=None):
        for (lora_A, lora_B) in zip(self.lora_experts_A, self.lora_experts_B):
            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_B.weight)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path
        result = self.base_layer(x)
        
        # TODO does this work with torch.compile?
        if self.merged:
            return result
        
        # LoRA path
        dropout_x = F.dropout(x, p=self.dropout_p, training=self.training)
        for (s, lora_A, lora_B) in zip(self.scaling_factors, self.lora_experts_A, self.lora_experts_B):
            lora_expert_out = lora_B(lora_A(dropout_x))
            result = result + s*lora_expert_out
        return result
    
    @classmethod
    def copy_linear(cls, lin: nn.Linear) -> nn.Linear:
        duplicate = nn.Linear(in_features=lin.in_features, out_features=lin.out_features,
                              bias=lin.bias is not None, device=lin.weight.device, dtype=lin.weight.dtype)
        with torch.no_grad():
            duplicate.weight.copy_(lin.weight)
            if lin.bias is not None:
                duplicate.bias.copy_(lin.bias)
        return duplicate
    
    @torch.no_grad()
    def merge_for_save(self):
        """
        Merges the LoRA weights into the base_layer and switches the module 
        to 'merged' mode.
        
        Formula: W_base_new = W_base + sum(s * B @ A)
        """
        if self.merged:
            return

        for s, lora_A, lora_B in zip(self.scaling_factors, self.lora_experts_A, self.lora_experts_B):
            delta_weight = s * (lora_B.weight @ lora_A.weight)
            self.base_layer.weight.data.add_(delta_weight)
            
        self.merged = True

# --------------------------------------------------------------------------
# Wrapper Functions
# --------------------------------------------------------------------------

def wrap_model_with_molf(model, molf_cfg: MoLFConfig):
    """
    Brief:
        Identifies target Linear layers in a model and replaces them with 
        MixtureOfLoRAFull modules for parameter-efficient fine-tuning.

    Usage:
        model = wrap_model_with_molf(model, molf_cfg)

    Distributed Warnings (FSDP/DDP):
        - FSDP: You MUST call this function BEFORE wrapping the model with FSDP. 
            FSDP needs to shard the parameters based on the final module structure. 
            If called after sharding, FSDP will not 'see' the LoRA experts.
        Correct:
        model = MyLLM()
        model = wrap_model_with_molf(model, config)  # <--- HERE
        model = FSDP(model, auto_wrap_policy=...)
        
        - DDP: Recommended to call before wrapping with DDP. If calling after, 
            ensure all ranks have identical initializations (usually via seed).
    """
    target_modules = molf_cfg.target_modules
    
    list_of_linear_cands = []
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear) and (any(name.endswith(t) for t in target_modules) or molf_cfg.all_linear)
            and not (any(t in name for t in molf_cfg.exclude_target_modules))):
            list_of_linear_cands.append((name, module))
        elif molf_cfg.freeze_non_linear:
            for p in module.parameters(recurse=False):
                p.requires_grad = False
        
    for (name, module) in list_of_linear_cands:
        parent, attr = get_parent_and_attr(model, name)

        molf_module = MixtureOfLoRAFull(
            base_linear=module,
            lora_alphas=molf_cfg.lora_alphas,
            lora_experts_ranks=molf_cfg.lora_experts_ranks,
            use_rslora=molf_cfg.use_rslora,
            lora_dropout=molf_cfg.lora_dropout,
            copy_weight=molf_cfg.copy_weight,
            freeze_base=not molf_cfg.molf_fft,
        )
        
        # If parent is ModuleList, attr will be an index string
        if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr)] = molf_module
        else:
            setattr(parent, attr, molf_module)
    
    return model



def merge_molf_and_unwrap(model):
    """
    Brief:
        Recursively merges all LoRA expert weights into their respective 
        base layers and replaces the MixtureOfLoRAFull wrappers with 
        standard nn.Linear modules.

    Usage:
        # After training is complete
        model = merge_molf_and_unwrap(model)

    Distributed Warnings (FSDP/DDP):
        - FSDP: DO NOT call this on a sharded model. You must either:
            1. Call this within a `summon_full_params` context.
            2. Call this on a gathered/consolidated CPU model after training.
            Merging requires access to the full weight matrices (B @ A); 
            running this on a local shard will result in mathematical errors.
        Correct:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                # Now model parameters are full (unsharded) on this context
                merged_model = merge_molf_and_unwrap(model)
                torch.save(merged_model.state_dict(), "final.pt")
                
        - DDP: It is safer to run this on Rank 0 only and then broadcast or 
            simply save the result, as the operation is deterministic.
    """
    # Iterate over children. We use list() to create a copy so we can modify the module in-place.
    for name, module in list(model.named_children()):
        # 1. recurse (dfs)
        merge_molf_and_unwrap(module)

        # 2. check if this child is MixtureOfLoRAFull wrapper
        if isinstance(module, MixtureOfLoRAFull):
            # merge weight
            module.merge_for_save()
            
            # 'base' contains the final combined weights
            base = module.base_layer
            
            # replace the MixtureOfLoRAFull with the new standard Linear
            # must determine how to set the attribute on the parent
            if isinstance(model, (nn.ModuleList, nn.Sequential)) and name.isdigit():
                model[int(name)] = base
            else:
                setattr(model, name, base)
                
    return model