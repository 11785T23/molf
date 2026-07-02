# MoLF-Efficient: Frozen-Base Mixture of LoRA Experts

<div align="center">
  <a href="https://arxiv.org/abs/2605.07111v1"><img src="https://img.shields.io/static/v1?label=arXiv&message=MoLF&color=red&logo=arxiv"></a> &ensp;
</div>

Reference implementation for **MoLF-Efficient** (§4.4 of **"Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation"**).

MoLF-Efficient is the parameter-efficient variant of MoLF. It **freezes the base weight — dropping the FFT expert entirely** — and reparameterizes each target linear as a superposition of **two or more LoRA experts of potentially different ranks**: `W_frozen + Σ s_i · B_i · A_i`. The same custom AdamW optimizer scores the experts with the *Expected Preconditioned Descent* (EPD) statistic and applies a Top-K sparse update at every step, while every LoRA expert keeps receiving full-batch gradient signals through the dense forward pass. Because the large dense weight never updates, MoLF-Efficient trains a small fraction of the parameters of the full MoLF while keeping its gradient-guided routing.

> **Branches.** This repo ships the two paper methods on separate branches:
> - **`main`** — MoLF: FFT + LoRA mixture (§4.1–4.3 of the paper).
> - **`molf-e`** (this branch) — MoLF-Efficient (§4.4): base weight frozen, routing over a pair of LoRA experts of potentially different ranks.
>
> Switch to the full FFT + LoRA method with `git checkout main`.

## Methods provided

| Mode (`--mode`) | Description |
|---|---|
| `fft` | Full fine-tuning baseline. |
| `lora` | LoRA baseline (via `peft`). Rank set with `--lora_rank`. |
| `molf` | Mixture of experts routed at the optimizer level. Set `--molf_fft False` for **MoLF-Efficient** (base weight frozen, LoRA experts only); leave `--molf_fft True` (default) for the full FFT + LoRA mixture. |

Key MoLF-Efficient flags:

- `--molf_fft False` — freeze the base weight and drop the FFT expert; route only over the LoRA experts.
- `--molf_lora_ranks "[32,128]"` — the ranks of the LoRA experts (any count; the paper uses a pair of unequal ranks).
- `--molf_topk` — number of experts updated per step (e.g. `2` to route over both experts of a pair).
- `--molf_freeze_non_linear True` — additionally freeze the leftover norm / embedding / bias parameters so that *only* the LoRA experts train.

Score functions (`--molf_score_fn`):

- `true_projected` *(default)* — **EPD** (paper Eq. 10); the score used throughout the main results.
- `projected` — **PFN** (paper Eq. 11); scale-invariant ablation comparator.

## Datasets and models

Three benchmarks, three base models — matching the paper's setup:

| Benchmark | HF id | Metric | Module |
|---|---|---|---|
| **SQL** (Text-to-SQL) | `gretelai/synthetic_text_to_sql` | exact-match accuracy on held-out queries | `src/data/sql.py`, eval in `src/evaluation/eval_helper/eval_sql.py` |
| **Med** (Medical QA) | `openlifescienceai/medmcqa` | 4-way MCQ accuracy on validation split | `src/data/medmcqa.py`, eval in `eval_medmcqa.py` |
| **Fact** (CounterFact) | downloaded from `rome.baulab.info` on first use, cached at `src/data/data_source/counterfact.json` | Efficacy Score (Meng et al.) | `src/data/fact.py`, eval in `eval_helper/eval_fact.py` |

| Model | HF id |
|---|---|
| Gemma-3-1B | `google/gemma-3-1b-pt` |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` |

## Installation

```bash
conda create -n molf python=3.10 -y
conda activate molf
pip install -e .
# Optional: FlashAttention 2 (Ampere+ GPUs only)
# pip install flash-attn --no-build-isolation
```

Hardware: the paper used a single NVIDIA H100 or RTX PRO 6000 Blackwell per run. V100 is unsupported (no FlashAttention 2; the code falls back to SDPA/eager attention).

## Quickstart: a single MoLF-Efficient run

Two LoRA experts of ranks 32 and 128, base weight frozen, routing over both experts each step:

```bash
torchrun --nnodes 1 --nproc_per_node=1 src/script/train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --dataset_type med \
    --mode molf \
    --molf_fft False \
    --molf_freeze_non_linear True \
    --molf_lora_ranks "[32,128]" \
    --molf_topk 2 \
    --molf_score_fn true_projected \
    --learning_rate 5e-5 \
    --lora_learning_rate 5e-5 \
    --weight_decay 0.1 \
    --lora_weight_decay 0.01 \
    --use_rslora True \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --warm_up_ratio 0.1 \
    --ckpt_dir_root ckpts/molf-e
```

With `--molf_fft False`, the base weight is frozen and the FFT expert is dropped; `--lora_learning_rate` / `--lora_weight_decay` govern the LoRA experts (the FFT-side `--learning_rate` / `--weight_decay` only affect any leftover trainable non-linear params, which `--molf_freeze_non_linear True` removes). Keep `--molf_fft True` to reproduce the full MoLF mixture from the `main` branch.

`train.py` first fine-tunes, then merges the LoRA experts back into the base linears so the saved checkpoint is a vanilla `transformers` model. After training, the corresponding evaluator (`evaluate_sql` / `evaluate_medmcqa` / `evaluate_fact`) is called automatically and a result row is appended to the CSV at `$LOG_FILE`. The per-step expert-selection log (`molf_expert_selection.jsonl`) is written under `ckpts/.../<run_name>/` and is what produces the routing-dynamics plots (paper Figures 4–6).

To inspect how few parameters MoLF-Efficient trains, use the parameter-count utility with `--no-use_fft`:

```bash
python src/script/count_params.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --molf_lora_ranks 32 128 \
    --no-use_fft --freeze_non_linear
```

## Reproducing the baseline sweeps

Two SLURM array scripts at the top of `scripts/` reproduce the FFT and LoRA baseline hyperparameter searches from the paper appendix (Table 4) on the Med benchmark; adapt the `DATASET_TYPE` / model list / LR grid to cover the other paper cells:

```
scripts/
├── fft_med_exp.sh   # FFT baseline sweep (paper Table 1, Table 2 "FFT")
└── lora_med_32.sh   # LoRA baseline sweep at rank 32 (paper Table 1)
```

Each script runs a `3 models × 3 LRs × 2 LR-schedulers = 18`-job array. The SLURM headers (partitions, `conda activate dft`, `CTIME_DATA=/data/user_data/...`) reflect the cluster they were authored on; adapt them to your environment. For MoLF-Efficient runs, use the single-command quickstart above and vary `--molf_*` / `--dataset_type` as needed.

## Code layout

```
src/
├── config/         # TrainingConfig (draccus) + adapter configs
├── data/           # SQL / MedMCQA / CounterFact dataset builders
├── evaluation/
│   └── eval_helper/   # evaluators that train.py imports after training
├── model/molf.py   # MixtureOfLoRAFull module (supports a frozen base) + wrap/merge utilities
├── optim/
│   ├── molf_adamw.py     # Top-K sparse AdamW with per-module routing
│   ├── molf_metric.py    # EPD (true_projected) + PFN (projected) score functions
│   └── param_group.py    # Per-module subgroup builder (LoRA experts, plus the FFT expert when molf_fft=True)
├── trainer/
│   ├── base_trainer.py   # FFT + LoRA baselines
│   └── molf_trainer.py   # MoLF training loop & checkpoint merge
└── script/
    ├── train.py        # Entry point (draccus-wrapped main)
    └── count_params.py # Parameter-count utility (--no-use_fft for MoLF-Efficient)
```

## License & Acknowledgments

The MoLF source code in this repository is released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for the full attribution required by the third-party assets we depend on. Summary:

- **Pre-trained models** are pulled from the Hugging Face Hub at runtime and are **not** redistributed by this repo. Each carries its own license and downstream users must comply independently:
  - `google/gemma-3-1b-pt` — [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (research + downstream fine-tuning, subject to Google's prohibited-use policy)
  - `Qwen/Qwen2.5-1.5B` — [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE) (research + commercial)
  - `Qwen/Qwen2.5-3B` — [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE) (**non-commercial research only**)
- **Datasets**:
  - **CounterFact** ([MIT](https://github.com/kmeng01/rome/blob/main/LICENSE), Meng et al., NeurIPS 2022) is fetched from `https://rome.baulab.info/data/dsets/counterfact.json` on first call to `CounterfactDatasetBuilder` / `evaluate_fact` and cached at `src/data/data_source/counterfact.json`. **Not** redistributed by this repo.
  - **MedMCQA** ([Apache 2.0](https://huggingface.co/datasets/openlifescienceai/medmcqa)) and **Gretel Synthetic Text-to-SQL** ([Apache 2.0](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)) are pulled from the Hugging Face Hub at training/eval time. **Not** redistributed by this repo.

## Citation

```bibtex
@article{tang2026molf,
  title={Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation},
  author={Tang, Haozhan and Zhu, Xiuqi and Zhang, Xinyin and Li, Boxun and Smith, Virginia and Kuo, Kevin},
  journal={arXiv preprint arXiv:2605.07111},
  year={2026}
}
```
