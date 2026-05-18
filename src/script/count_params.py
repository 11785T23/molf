"""
Count total vs trainable (requires_grad) parameters for a MoLF-wrapped model.
Usage:
    python src/script/count_params.py --model_name_or_path google/gemma-3-1b-pt
"""

import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from src.config.adapter_config import MoLFConfig
from src.model.molf import wrap_model_with_molf, MixtureOfLoRAFull


def count_parameters(model):
    total = 0
    trainable = 0
    # Breakdown buckets
    base_weight_total, base_weight_train = 0, 0
    base_bias_total, base_bias_train = 0, 0
    lora_total, lora_train = 0, 0
    other_total, other_train = 0, 0

    counted = set()

    # Count MoLF modules explicitly
    for name, module in model.named_modules():
        if isinstance(module, MixtureOfLoRAFull):
            # base layer weight
            p = module.base_layer.weight
            pid = id(p)
            if pid not in counted:
                counted.add(pid)
                n = p.numel()
                base_weight_total += n
                if p.requires_grad:
                    base_weight_train += n

            # base layer bias
            if module.base_layer.bias is not None:
                p = module.base_layer.bias
                pid = id(p)
                if pid not in counted:
                    counted.add(pid)
                    n = p.numel()
                    base_bias_total += n
                    if p.requires_grad:
                        base_bias_train += n

            # LoRA experts
            for expert_list in [module.lora_experts_A, module.lora_experts_B]:
                for expert in expert_list:
                    for p in expert.parameters():
                        pid = id(p)
                        if pid not in counted:
                            counted.add(pid)
                            n = p.numel()
                            lora_total += n
                            if p.requires_grad:
                                lora_train += n

    # Everything else (embeddings, norms, lm_head, etc.)
    for name, p in model.named_parameters():
        pid = id(p)
        if pid not in counted:
            counted.add(pid)
            n = p.numel()
            other_total += n
            if p.requires_grad:
                other_train += n

    total = base_weight_total + base_bias_total + lora_total + other_total
    trainable = base_weight_train + base_bias_train + lora_train + other_train

    return {
        "total": total,
        "trainable": trainable,
        "base_weight": (base_weight_total, base_weight_train),
        "base_bias": (base_bias_total, base_bias_train),
        "lora_experts": (lora_total, lora_train),
        "other": (other_total, other_train),
    }


def fmt(n):
    """Format large numbers with commas and M/B suffix."""
    if n >= 1e9:
        return f"{n:>14,}  ({n/1e9:.2f}B)"
    elif n >= 1e6:
        return f"{n:>14,}  ({n/1e6:.2f}M)"
    else:
        return f"{n:>14,}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--molf_lora_ranks", type=int, nargs="+", default=[128])
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_rslora", action="store_true", default=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Model: {args.model_name_or_path}")
    print(f"MoLF lora_ranks: {args.molf_lora_ranks}")
    print(f"{'='*70}")

    # Load model on CPU in bfloat16 (no GPU needed for counting)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Count before MoLF wrapping
    pre_total = sum(p.numel() for p in model.parameters())
    pre_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n--- Before MoLF wrapping ---")
    print(f"  Total params:     {fmt(pre_total)}")
    print(f"  Trainable params: {fmt(pre_trainable)}")
    pct_pre = 100.0 * pre_trainable / pre_total if pre_total > 0 else 0
    print(f"  Trainable / Total = {pct_pre:.2f}%")

    # Build MoLF config (matching train_config.py __post_init__ for molf mode)
    molf_cfg = MoLFConfig(
        lora_alphas=tuple([args.lora_alpha] * len(args.molf_lora_ranks)),
        lora_experts_ranks=tuple(args.molf_lora_ranks),
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
    )

    print("\nWrapping model with MoLF...")
    model = wrap_model_with_molf(model, molf_cfg)

    stats = count_parameters(model)

    print(f"\n--- After MoLF wrapping ---")
    print(f"  {'Category':<25} {'Total':>30}  {'Trainable':>30}")
    print(f"  {'-'*25} {'-'*30}  {'-'*30}")
    for key in ["base_weight", "base_bias", "lora_experts", "other"]:
        t, tr = stats[key]
        print(f"  {key:<25} {fmt(t)}  {fmt(tr)}")
    print(f"  {'-'*25} {'-'*30}  {'-'*30}")
    print(f"  {'TOTAL':<25} {fmt(stats['total'])}  {fmt(stats['trainable'])}")
    pct = 100.0 * stats["trainable"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\n  Trainable / Total = {pct:.2f}%")
    added = stats["total"] - pre_total
    print(f"  Added by MoLF:    {fmt(added)}")
    ratio = stats["trainable"] / pre_trainable if pre_trainable > 0 else float('inf')
    print(f"  Trainable after / before MoLF = {ratio:.4f}x")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
