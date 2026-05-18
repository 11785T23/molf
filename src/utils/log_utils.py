import re, os
from typing import List, Dict, Any, Optional, Tuple

def parse_model_hyperparams(model_path: str) -> Dict[str, str]:
    run_name = os.path.basename(os.path.normpath(model_path))
    params = {
        "model_name": "unknown",
        "dataset_type": "unknown", "top_k": "unknown", "score_fn": "unknown",
        "lr": "unknown", "lora_lr": "unknown", "scheduler": "unknown", "warmup_ratio": "unknown"
    }
    try:
        model_name_match = re.search(r"molf_([^_]+)_", run_name)
        if model_name_match: params["model_name"] = model_name_match.group(1)
        dataset_match = re.search(r"_([^_]+)_k=", run_name)
        if dataset_match: params["dataset_type"] = dataset_match.group(1)
        k_match = re.search(r"k=([^_]+)", run_name)
        if k_match: params["top_k"] = k_match.group(1)
        scf_match = re.search(r"scf=([^_]+)", run_name)
        if scf_match: params["score_fn"] = scf_match.group(1)
        lr_match = re.search(r"_lr=([^_]+)", run_name)
        if lr_match: params["lr"] = lr_match.group(1).replace('d', '.')
        llr_match = re.search(r"llr=([^_]+)", run_name)
        if llr_match: params["lora_lr"] = llr_match.group(1).replace('d', '.')
        last_segment = run_name.split('_')[-1]
        sched_match = re.match(r"([a-zA-Z]+)(.*)", last_segment)
        if sched_match:
            params["scheduler"] = sched_match.group(1)
            params["warmup_ratio"] = sched_match.group(2).replace('d', '.')
    except Exception as e:
        print(f"Warning: Failed to parse hyperparams: {e}")
    return params
