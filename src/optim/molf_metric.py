"""
Score functions for MoLF's per-module Top-K expert routing.

Two variants are exposed, matching the formulations in the paper:

- ``projected`` — Preconditioned Frobenius Norm (PFN), Eq. (11) of the paper.
    Pure direction-only score, scale-invariant under RS-LoRA's alpha/sqrt(r):
        S_PFN = (1/sqrt(N_params)) * sqrt( sum_theta ( m / (sqrt(v) + eps) )^2 )

- ``true_projected`` — Expected Preconditioned Descent (EPD), Eq. (10).
    First-order expected loss reduction normalized by parameter count; this
    is the score used by MoLF (and the default for new configs):
        S_EPD = (eta / N_params) * sum_theta m^2 / (sqrt(v) + eps)
"""

import torch
from typing import Callable, List

ScoreFuncType = Callable[
    [
        List[torch.Tensor],  # grads
        List[torch.Tensor],  # exp_avgs (m)
        List[torch.Tensor],  # exp_avg_sqs (v)
        List[torch.Tensor],  # steps
        List[float],         # lrs (one per param in the expert; all equal within an expert)
        List[float]          # wds (one per param in the expert)
    ],
    float                    # return: scalar score
]


@torch.compile(mode="reduce-overhead")
def mean_projected_reduction(
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    steps: List[torch.Tensor],
    lrs: List[float],
    wds: List[float],
) -> float:
    """Preconditioned Frobenius Norm (PFN). Paper Eq. (11)."""
    ref_device = exp_avgs[0].device
    total_norm = torch.tensor(0.0, device=ref_device, dtype=torch.float32)
    num_parameters = torch.tensor(0.0, device=ref_device)
    eps = 1e-8

    for m, v in zip(exp_avgs, exp_avg_sqs):
        adam_dir = m / (torch.sqrt(v) + eps)
        total_norm += torch.square(adam_dir).sum()
        num_parameters += m.numel()

    score = torch.sqrt(total_norm) / torch.sqrt(num_parameters)
    return score


def true_projected_reduction(
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    steps: List[torch.Tensor],
    lrs: List[float],
    wds: List[float],
) -> float:
    """Expected Preconditioned Descent (EPD). Paper Eq. (10)."""
    ref_device = exp_avgs[0].device
    lr_val = lrs[0]
    return _true_projected_reduction_inner(exp_avgs, exp_avg_sqs, lr_val, ref_device)


@torch.compile(mode="reduce-overhead")
def _true_projected_reduction_inner(
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    lr_val: float,
    ref_device: torch.device,
) -> float:
    total_reduction = torch.tensor(0.0, device=ref_device, dtype=torch.float32)
    num_parameters = torch.tensor(0.0, device=ref_device)
    eps = 1e-8

    for m, v in zip(exp_avgs, exp_avg_sqs):
        step_reduction = torch.square(m) / (torch.sqrt(v) + eps)
        total_reduction += step_reduction.sum()
        num_parameters += m.numel()

    score = total_reduction * lr_val / num_parameters
    return score


SCORE_F_MAP = {
    "projected": mean_projected_reduction,       # PFN — paper Eq. 11
    "true_projected": true_projected_reduction,  # EPD — paper Eq. 10 (default)
}
