from torch.optim import AdamW
import torch
import torch.distributed as dist
from torch.optim import (
    Optimizer,
)
from torch.optim.adamw import adamw
from torch.optim.optimizer import (
    ParamsT,
    _get_value,
    _stack_if_compiling
)
import math, json, os, logging
from torch import Tensor
from typing import cast, List, Optional, Tuple, Union
from src.model.molf import MixtureOfLoRAFull
from .molf_metric import ScoreFuncType

# --------------------------------------------------------------------------
# Expert Selection Logging
# Set MOLF_LOG_EXPERT_SELECTION = True to write per-step expert selection
# info to MOLF_LOG_FILE (JSONL format). Only rank 0 writes in distributed.
# --------------------------------------------------------------------------
MOLF_LOG_EXPERT_SELECTION = True
MOLF_LOG_FILE = "molf_expert_selection.jsonl"

logger = logging.getLogger(__name__)

class MoLFAdamW(Optimizer):
    """
    Custom AdamW that supports sparse Mixture-of-LoRA-Full updates.
    
    score_func take gradient, first moment, second moment, steps, lrs, wds
    (1) gradient, first moment, second moment, steps are list of Tensor
    (2) lrs, wds are list of scalars
    """
    def __init__(
        self, 
        params: ParamsT,
        score_func: ScoreFuncType,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        molf_topk: int = 1,
        amsgrad: bool = False,
        log_file_path: str = MOLF_LOG_FILE,
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if amsgrad:
            # TODO
            raise NotImplementedError("MoLFAdamW currently don't support amsgrad")
        self.amsgrad = amsgrad
        
        self.score_func = score_func
        self.molf_topk = molf_topk
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        super().__init__(params, defaults)
        # This use dict.setdefault(name, default) for each parameter group, so only non-existing field will be set
        # and the established weight decay will not be set.
        
        self._molf_step = 0
        self._log_fh = None
        self._log_file_path = log_file_path

    # ------------------------------------------------------------------
    # Expert Selection Logging Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rank0() -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def _log_selections(self, step_selections: dict):
        if not MOLF_LOG_EXPERT_SELECTION or not self._is_rank0():
            return
        if self._log_fh is None:
            logger.info(f"Creating MoLF expert selection log: {os.path.abspath(self._log_file_path)}...")
            self._log_fh = open(self._log_file_path, "a")
            logger.info(f"MoLF expert selection log: {os.path.abspath(self._log_file_path)}")
        record = {"step": self._molf_step, "selections": step_selections}
        self._log_fh.write(json.dumps(record) + "\n")
        self._log_fh.flush()

    @staticmethod
    def _rebuild_subgroups_if_needed(group):
        """
        Rebuild molf_subgroups params from the authoritative group['params'].

        Accelerate's prepare() can replace the parameter objects in
        group['params'] (e.g. for device placement or DDP remapping) without
        touching custom nested structures like molf_subgroups.  This leaves
        the subgroups holding stale references to the old (pre-prepare) tensors,
        which no longer receive gradients.

        Because group['params'] was built by extending subgroup params in order
        (sub0 + sub1 + ...), we can slice it back using the original sizes.
        Only runs the rebuild once per group.
        """
        if group.get('_subgroups_rebuilt', False):
            return

        subgroups = group.get('molf_subgroups', [])
        flat_ids = {id(p) for p in group['params']}
        sub_ids = {id(p) for sub in subgroups for p in sub['params']}

        if flat_ids == sub_ids:
            group['_subgroups_rebuilt'] = True
            return

        expected_total = sum(len(sub['params']) for sub in subgroups)
        actual_total = len(group['params'])
        assert expected_total == actual_total, (
            f"[MoLF] Cannot rebuild subgroups: subgroup param count ({expected_total}) "
            f"!= group['params'] count ({actual_total}) for module={group.get('module_name', '?')}. "
            f"This means the flat list structure diverged from the subgroup structure."
        )

        offset = 0
        for sub in subgroups:
            n = len(sub['params'])
            sub['params'] = list(group['params'][offset:offset + n])
            offset += n

        group['_subgroups_rebuilt'] = True
        logger.info(
            f"[MoLF] Rebuilt subgroup params for module={group.get('module_name', '?')} "
            f"({offset} params remapped)"
        )

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):
        """Helper to prepare lists for functional API."""
        for p in group['params']:
            if p.grad is None: continue
            
            params_with_grad.append(p)
            grads.append(p.grad)
            
            state = self.state[p]
            if len(state) == 0:
                # Critical: Must be Tensor for functional API
                state['step'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            state_steps.append(state['step'])
    """
    Core mechanism of MixtureOfLoRAFull AdamW
    TODO currently don't support closure (gradient wrt hyperparameter of optimizer)
    """
    @torch.no_grad()
    def step(self, closure=None):
        # CUDA graph compatibility
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            raise NotImplementedError("MoLFAdamW currently do not support closure")
            with torch.enable_grad():
                loss = closure()

        step_selections = {}
        for group in self.param_groups:
            group_type = group.get('type', 'standard')
            
            if group_type == 'standard':
                self._step_functional_standard(group)
            elif group_type == 'molf':
                selection_info = self._step_custom_molf(group)
                if selection_info is not None:
                    step_selections.update(selection_info)
        
        if MOLF_LOG_EXPERT_SELECTION and step_selections:
            self._log_selections(step_selections)
        self._molf_step += 1
                
        return loss

    """
    Simple AdamW on non-MixtureOfLoRAFull modules, utilizing optimized torch implementation
    TODO currently only use foreach, but not fused, since haven't check all tensor has float type
    """
    def _step_functional_standard(self, group):
        """Use PyTorch's optimized functional API for standard params."""
        params, grads, exp_avgs, exp_avg_sqs, state_steps = [], [], [], [], []
        beta1, beta2 = cast(Tuple[float, float], group["betas"])
        self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, state_steps)

        if not params: return
        adamw(
            params=params,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=[],
            state_steps=state_steps,
            amsgrad=self.amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            eps=group['eps'],
            maximize=False,
            foreach=True, # Enable fast kernels
            capturable=False,
            differentiable=False,
            fused=None
        )

    def _step_custom_molf(self, group):
        """
        Split-Phase Update:
        1. Gather params & init state (All)
        2. Update Moments (All)
        3. Increment Steps (All) — keeps bias correction in sync with moment updates
        4. Score (All)
        5. Select Winners (Top-K)
        6. Update Weights (Winners Only)
        """
        self._rebuild_subgroups_if_needed(group)
        beta1, beta2 = cast(Tuple[float, float], group["betas"])
        eps = group['eps']
        topk = self.molf_topk
        subgroups = group.get('molf_subgroups', [])
        num_experts = group.get('num_experts', 0)
        experts_data = [
            {
                'p': [], 'grads': [], 'exp_avgs': [], 'exp_avg_sqs': [], 
                'steps': [], 'lrs': [], 'wds': [], 'score': 0.0
            } 
            for _ in range(num_experts)
        ]
        
        # [Bug 1 fix] Compute LR schedule scale from the scheduler-managed group lr.
        # PyTorch's LRScheduler sets 'initial_lr' at creation and updates 'lr' each step.
        # Subgroup lrs are fixed at init, so we scale them by the scheduler's ratio.
        initial_lr = group.get('initial_lr', group['lr'])
        lr_scale = group['lr'] / initial_lr if initial_lr > 0 else 1.0
        
        # -----------------------------------------------------
        # Phase 1: Gather params & init state (ALL)
        # -----------------------------------------------------
        all_params_with_grad = []
        all_grads = []
        all_exp_avgs = []
        all_exp_avg_sqs = []
        for sub in subgroups:
            expert_idx = sub.get('expert_idx')
            # Safety check for index
            if expert_idx is None or expert_idx >= num_experts:
                continue

            e_data = experts_data[expert_idx]
            
            for p in sub['params']:
                if p.grad is None:
                    continue
                
                # Lazy State Initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # Populate Expert Data
                e_data['p'].append(p)
                e_data['grads'].append(p.grad)
                e_data['exp_avgs'].append(state['exp_avg'])
                e_data['exp_avg_sqs'].append(state['exp_avg_sq'])
                e_data['steps'].append(state['step'])
                e_data['lrs'].append(sub['lr'] * lr_scale)
                e_data['wds'].append(sub['weight_decay'])
                
                # Populate Flattened Lists (Refs to same objects)
                all_params_with_grad.append(p)
                all_grads.append(p.grad)
                all_exp_avgs.append(state['exp_avg'])
                all_exp_avg_sqs.append(state['exp_avg_sq'])

        if not all_params_with_grad:
            return
        # -----------------------------------------------------
        # Phase 2: Update Moments (ALL)
        # -----------------------------------------------------
        # m = beta1 * m + (1 - beta1) * g
        torch._foreach_lerp_(all_exp_avgs, all_grads, 1-beta1)

        # v = beta2 * v + (1 - beta2) * (g * g)
        torch._foreach_mul_(all_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(all_exp_avg_sqs, all_grads, all_grads, value=1 - beta2)
        del all_grads, all_exp_avgs, all_exp_avg_sqs # save vram?
        
        # -----------------------------------------------------
        # Phase 3: Increment Steps (ALL)
        # [Bug 3 fix] Step counter must track moment updates, not weight updates,
        # so that bias correction stays in sync with the EMA state.
        # -----------------------------------------------------
        for e_data in experts_data:
            if e_data['steps']:
                torch._foreach_add_(e_data['steps'], 1)
        
        # -----------------------------------------------------
        # Phase 4: Scoring
        # -----------------------------------------------------
        for i, e_data in enumerate(experts_data):
            e_data['idx'] = i
            if not e_data['p']:
                e_data['score'] = -float('inf')
                continue
            e_data['score'] = self.score_func(
                e_data['grads'], 
                e_data['exp_avgs'], 
                e_data['exp_avg_sqs'], 
                e_data['steps'], 
                e_data['lrs'], 
                e_data['wds']
            ).item()
        # -----------------------------------------------------
        # Phase 5: Select Winners
        # -----------------------------------------------------
        experts_data.sort(key=lambda x: x['score'], reverse=True)
        winner_data = experts_data[:topk]

        # -----------------------------------------------------
        # Phase 5b: Build selection info for logging
        # -----------------------------------------------------
        selection_info = None
        if MOLF_LOG_EXPERT_SELECTION:
            module_name = group.get('module_name', 'unknown')
            expert_names = group.get('expert_names', {})
            winner_idxs = [e['idx'] for e in winner_data if e['p']]
            selection_info = {
                module_name: {
                    'winners': [expert_names.get(idx, str(idx)) for idx in winner_idxs],
                    'scores': {
                        expert_names.get(e['idx'], str(e['idx'])): round(float(e['score']), 6)
                        for e in experts_data if e['p']
                    }
                }
            }

        # -----------------------------------------------------
        # Phase 6: Update Weights (Winners Only)
        # -----------------------------------------------------
        for e_data in winner_data:
            if not e_data['p']: continue
            
            # weight decay
            decay_factors = [1 - lr * wd for lr, wd in zip(e_data['lrs'], e_data['wds'])]
            torch._foreach_mul_(e_data['p'], decay_factors)
            
            # adam
            steps_values = [step.item() for step in e_data['steps']]
            bias_correction1 = [1 - beta1 ** s for s in steps_values]
            bias_correction2 = [1 - beta2 ** s for s in steps_values]
            step_size = _stack_if_compiling([(lr / bc) * -1 for (lr, bc) in zip(e_data['lrs'], bias_correction1)])
            bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]  # type: ignore[arg-type]
            denom = torch._foreach_sqrt(e_data['exp_avg_sqs'])
            
            torch._foreach_div_(denom, bias_correction2_sqrt)
            torch._foreach_add_(denom, eps)
            # p = p + step_size * (m / denom)
            torch._foreach_addcdiv_(
                e_data['p'],
                e_data['exp_avgs'],
                denom,
                step_size,  # type: ignore[arg-type]
            )
        
        return selection_info