import torch
import torch.nn as nn
from typing import Dict, List, Any, Union

from src.model import MixtureOfLoRAFull

import torch
import torch.nn as nn

_NORM_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)

def _is_norm_module(module: nn.Module) -> bool:
    if isinstance(module, _NORM_TYPES):
        return True
    cls = module.__class__.__name__.lower()
    # class name suffix-ish matches
    return cls.endswith(("norm", "rmsnorm", "layernorm")) or ("rmsnorm" in cls) or ("layernorm" in cls)

def is_norm(param_name: str, module: nn.Module) -> bool:
    """
    Returns True if (param_name, module) corresponds to a normalization parameter
    that you typically want to put into the no-weight-decay group.
    """
    n = param_name.lower()
    name_says_norm = (
        ".norm" in n
        or "layernorm" in n
        or "layer_norm" in n
        or "rmsnorm" in n
    )
    return (_is_norm_module(module) or name_says_norm)


def is_bias(param_name: str, module: nn.Module) -> bool:
    """
    Returns True if this parameter is a bias term (should be no-weight-decay).
    """
    n = param_name.lower()
    return "bias" in param_name

def get_param_group_molf(
    model: nn.Module,
    lr_fft: float = 2e-5,
    lr_lora: float = 5e-5,
    weight_decay_fft: float = 0.1,
    weight_decay_lora: float = 0.01,
    bias_zero_weight_decay: bool = True,
    norm_zero_weight_decay: bool = True,
    molf_fft: bool = True,
) -> List[Dict[str, Any]]:
    """
    Constructs parameter groups with granular LR and WD control.
    
    TODO currently only support Qwen 2.5 and Gemma 3
    """
    molf_groups = []
    visited_ids = set()
    
    # ---------------------------------------------------------
    # 1. MoLF Groups (One group per Module, Mixed Sub-settings)
    # ---------------------------------------------------------
    wd_bias = 0.0 if bias_zero_weight_decay else weight_decay_fft
    for name, module in model.named_modules():
        if isinstance(module, MixtureOfLoRAFull):
            # We create a list of subgroups with specific overrides
            subgroups = []

            if molf_fft:
                expert_idx = 0
                expert_names = {0: 'base'}
                # base layer weight
                base_w = [p for n, p in module.base_layer.named_parameters() if "weight" in n]
                if base_w:
                    subgroups.append({
                        'name': 'base_weight',
                        'params': base_w,
                        'lr': lr_fft,
                        'weight_decay': weight_decay_fft,
                        'expert_idx': expert_idx
                    })

                # base layer bias
                base_b = [p for n, p in module.base_layer.named_parameters() if "bias" in n]
                if base_b:
                    subgroups.append({
                        'name': 'base_bias',
                        'params': base_b,
                        'lr': lr_fft,
                        'weight_decay': wd_bias,
                        'expert_idx': expert_idx
                    })
            else:
                expert_idx = -1
                expert_names = {}

            # lora experts (A&B)
            for i, (exp_A, exp_B) in enumerate(zip(module.lora_experts_A, module.lora_experts_B)):
                expert_idx += 1
                rank = module.lora_experts_ranks[i]
                expert_names[expert_idx] = f'lora_r{rank}'
                expert_params = list(exp_A.parameters()) + list(exp_B.parameters())
                subgroups.append({
                    'name': f'expert_{i}', 
                    'params': expert_params,
                    'lr': lr_lora,
                    'weight_decay': weight_decay_lora,
                    'expert_idx': expert_idx
                })
            
            # flatten
            all_module_params = []
            for sg in subgroups:
                all_module_params.extend(sg['params'])
                for p in sg['params']:
                    visited_ids.add(id(p))
            
            # create the outer group
            molf_groups.append({
                'params': all_module_params,
                'type': 'molf',
                'module_name': name,
                'expert_names': expert_names,
                'molf_subgroups': subgroups, 
                # defaults place holder just in case, though sub-groups override them
                'lr': lr_fft, 
                'weight_decay': weight_decay_fft,
                'num_experts': expert_idx+1
            })

    # ---------------------------------------------------------
    # 2. Standard Parameters (Split by Decay vs No-Decay)
    # ---------------------------------------------------------
    standard_decay = []
    standard_no_decay = []
    
    for name, module in model.named_modules():
        if isinstance(module, MixtureOfLoRAFull):
            continue
        # iterate immediate parameters of this module
        for param_name, param in module.named_parameters(recurse=False):
            if id(param) in visited_ids or not param.requires_grad:
                continue
            
            use_wd = True
            
            # bias or norm weights
            if ((is_bias(param_name, module=module) and bias_zero_weight_decay) 
                or (is_norm(param_name, module=module) and norm_zero_weight_decay)):
                use_wd = False
            # otherwise, this is embedding, linear, etc. that will use weight decay
                
            if use_wd:
                standard_decay.append(param)
            else:
                standard_no_decay.append(param)
                
            visited_ids.add(id(param)) # Mark as processed

    # handle any remaining loose parameters not caught by module loop
    for p in model.parameters():
        if id(p) not in visited_ids and p.requires_grad:
            # use weight decay by default
            standard_decay.append(p)

    # ---------------------------------------------------------
    # 3. Combine Groups
    # ---------------------------------------------------------
    final_groups = []
    
    if standard_decay:
        final_groups.append({
            'params': standard_decay, 
            'type': 'standard', 
            'lr': lr_fft,
            'weight_decay': weight_decay_fft
        })
        
    if standard_no_decay:
        final_groups.append({
            'params': standard_no_decay, 
            'type': 'standard', 
            'lr': lr_fft,
            'weight_decay': 0.0
        })
    
    final_groups.extend(molf_groups)
    final_groups = [g for g in final_groups if len(g['params']) > 0]
    
    return final_groups