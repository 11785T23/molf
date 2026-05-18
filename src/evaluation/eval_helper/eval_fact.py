import argparse
import os
import random
import sys
import gc
import csv
import re
import torch
# Default CounterFact JSON bundled at src/data/data_source/counterfact.json.
_DEFAULT_COUNTERFACT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "data_source", "counterfact.json"
)
_DEFAULT_COUNTERFACT_PATH = os.path.normpath(_DEFAULT_COUNTERFACT_PATH)
import torch._dynamo
# Give the compiler a massive cache so it doesn't hit the limit and crash
torch._dynamo.config.cache_size_limit = 1024
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from src.utils.log_utils import parse_model_hyperparams

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def make_deterministic(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

class FactEvaluator:
    def __init__(self, model_path: str, device: str, tokenizer: AutoTokenizer, dtype: torch.dtype):
        self.device = device
        self.tokenizer = tokenizer
        print(f"Loading model from {model_path} on device {device} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        self.model.eval()

    def generate_answer(self, prompts: List[str], max_new_tokens: int = 64, temperature: float = 0.0) -> Tuple[List[str], torch.Tensor]:
        """Used for Efficacy: We actually need the generated text here."""
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        input_len = inputs["input_ids"].shape[1]
        output_ids = outputs.sequences
        generated_ids = output_ids[:, input_len:]
        decoded_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract first token probs, then immediately free the massive scores tuple from VRAM
        first_token_logits = outputs.scores[0] 
        first_token_probs = F.softmax(first_token_logits, dim=-1)
        
        del outputs
        
        return [text.strip() for text in decoded_texts], first_token_probs

    def get_next_token_probs(self, prompts: List[str], batch_size: int = 48) -> torch.Tensor:
        """
        Used for Generalization & Specificity: 
        Does a single forward pass to get the exact probabilities of the next token. 
        Zero autoregressive generation = instant execution and minimal VRAM usage.
        """
        all_probs = []
        
        # Chunk the potentially massive list of prompts to prevent VRAM explosion
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract the logits for the very last token in the prompt
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            all_probs.append(probs)
            
            del outputs, inputs
            
        return torch.cat(all_probs, dim=0)

    def eval_efficacy(self, batch_exs: List[Dict], max_new_tokens: int, temperature: float):
        prompts = []
        for ex in batch_exs:
            rw = ex['requested_rewrite']
            prompts.append(rw['prompt'].format(rw['subject'].strip()).strip())
        
        gen_texts, batch_probs = self.generate_answer(prompts, max_new_tokens, temperature)
        
        batch_results = []
        for idx, ex in enumerate(batch_exs):
            rw = ex['requested_rewrite']
            target_new = rw['target_new']['str'].strip()
            target_true = rw['target_true']['str'].strip()
            new_id = self.tokenizer.encode(" " + target_new, add_special_tokens=False)[0]
            true_id = self.tokenizer.encode(" " + target_true, add_special_tokens=False)[0]
            
            p_new = batch_probs[idx, new_id].item()
            p_true = batch_probs[idx, true_id].item()
            is_acc_new = gen_texts[idx].lower().startswith(target_new.lower())
            is_acc_true = gen_texts[idx].lower().startswith(target_true.lower())
            is_es = p_new > p_true

            batch_results.append({
                "prompt": prompts[idx], "gen": gen_texts[idx],
                "target_new": target_new, "target_true": target_true,
                "p_new": p_new, "p_true": p_true,
                "acc_new": is_acc_new, "acc_true": is_acc_true, "es": is_es, "diff": p_new - p_true
            })
            
        return batch_results
    
    def eval_generalization(self, batch_exs: List[Dict], max_new_tokens: int, temperature: float, inference_batch_size: int = 48):
        all_prompts, metadata = [], []
        for samp_idx, ex in enumerate(batch_exs):
            rw = ex['requested_rewrite']
            for p_prompt in ex.get('paraphrase_prompts', []):
                all_prompts.append(p_prompt.strip())
                metadata.append({
                    "samp_idx": samp_idx,
                    "target_new": rw['target_new']['str'].strip(),
                    "target_true": rw['target_true']['str'].strip()
                })

        if not all_prompts: return []

        # Replaced generate_answer with get_next_token_probs
        batch_probs = self.get_next_token_probs(all_prompts, batch_size=inference_batch_size)
        
        batch_results = []
        for idx, meta in enumerate(metadata):
            new_id = self.tokenizer.encode(" " + meta["target_new"], add_special_tokens=False)[0]
            true_id = self.tokenizer.encode(" " + meta["target_true"], add_special_tokens=False)[0]
            p_new = batch_probs[idx, new_id].item()
            p_true = batch_probs[idx, true_id].item()
            batch_results.append({
                "samp_idx": meta["samp_idx"], "prompt": all_prompts[idx],
                "p_new": p_new, "p_true": p_true, "ps": p_new > p_true, "pm": p_new - p_true
            })
            
        return batch_results
    
    def eval_specificity(self, batch_exs: List[Dict], max_new_tokens: int, temperature: float, inference_batch_size: int = 48):
        all_prompts, metadata = [], []
        for samp_idx, ex in enumerate(batch_exs):
            rw = ex['requested_rewrite']
            for n_prompt in ex.get('neighborhood_prompts', []):
                all_prompts.append(n_prompt.strip())
                metadata.append({
                    "samp_idx": samp_idx,
                    "target_new": rw['target_new']['str'].strip(),
                    "target_true": rw['target_true']['str'].strip()
                })
        
        if not all_prompts: return []

        # Replaced generate_answer with get_next_token_probs
        batch_probs = self.get_next_token_probs(all_prompts, batch_size=inference_batch_size)
        
        batch_results = []
        for idx, meta in enumerate(metadata):
            new_id = self.tokenizer.encode(" " + meta["target_new"], add_special_tokens=False)[0]
            true_id = self.tokenizer.encode(" " + meta["target_true"], add_special_tokens=False)[0]
            p_new = batch_probs[idx, new_id].item()
            p_true = batch_probs[idx, true_id].item()
            batch_results.append({
                "samp_idx": meta["samp_idx"], "prompt": all_prompts[idx],
                "p_new": p_new, "p_true": p_true, "ns": p_true > p_new, "nm": p_true - p_new
            })
            
        return batch_results

    def run_evaluation(self, test_data: List[Dict], batch_size: int, temperature: float = 0.0, max_new_tokens: int = 64):
        metrics = {
            "acc_new": [], "acc_true": [], "eff_es": [], "eff_em": [],
            "gen_ps": [], "gen_pm": [],
            "spec_ns": [], "spec_nm": []
        }

        print(f"Starting Multi-Metric Evaluation on {len(test_data)} samples...")

        for i in tqdm(range(0, len(test_data), batch_size)):
            batch_exs = test_data[i : i + batch_size]
            eff_res = self.eval_efficacy(batch_exs, max_new_tokens, temperature)
            
            # Pass down the batch_size so our single-forward-pass method can chunk properly
            gen_res = self.eval_generalization(batch_exs, max_new_tokens, temperature, inference_batch_size=batch_size)
            spec_res = self.eval_specificity(batch_exs, max_new_tokens, temperature, inference_batch_size=batch_size)

            if i == 0:
                print(f"\n{'='*20} Debug Sample 1 {'='*20}")
                e0 = eff_res[0]
                print(f"[Efficacy]\n Prompt: {e0['prompt']}\n Target: {e0['target_new']} | Gen: {e0['gen']}")
                print(f" P_new: {e0['p_new']:.4f} | P_true: {e0['p_true']:.4f} | ES: {e0['es']}")
                
                print(f"\n[Generalization]")
                for g in gen_res:
                    if g['samp_idx'] == 0:
                        print(f" Prompt: {g['prompt']}\n P_new: {g['p_new']:.4f} | P_true: {g['p_true']:.4f} | PS: {g['ps']}")
                
                print(f"\n[Specificity]")
                for s in spec_res:
                    if s['samp_idx'] == 0:
                        print(f" Prompt: {s['prompt']}\n P_new: {s['p_new']:.4f} | P_true: {s['p_true']:.4f} | NS: {s['ns']}")
                print(f"{'='*56}\n")

            for r in eff_res:
                metrics["acc_new"].append(r["acc_new"])
                metrics["acc_true"].append(r["acc_true"])
                metrics["eff_es"].append(r["es"])
                metrics["eff_em"].append(r["diff"])
            
            for r in gen_res:
                metrics["gen_ps"].append(r["ps"])
                metrics["gen_pm"].append(r["pm"])
                
            for r in spec_res:
                metrics["spec_ns"].append(r["ns"])
                metrics["spec_nm"].append(r["nm"])
            
        def avg(lst): return sum(lst) / len(lst) if lst else 0

        print("\n" + "=" * 30 + "\nEvaluation Results:\n" + "-" * 30)
        print(f"Efficacy:\n  ACC NEW: {avg(metrics['acc_new']):.4%}\n  ACC TRUE: {avg(metrics['acc_true']):.4%}\n  ES: {avg(metrics['eff_es']):.4%}\n  EM: {avg(metrics['eff_em']):.4f}")
        print(f"Generalization:\n  PS: {avg(metrics['gen_ps']):.4%}\n  PM: {avg(metrics['gen_pm']):.4f}")
        print(f"Specificity:\n  NS: {avg(metrics['spec_ns']):.4%}\n  NM: {avg(metrics['spec_nm']):.4f}\n" + "=" * 30)
        
        return metrics


def append_result(log_file: str, record: Dict[str, Any]) -> None:
    if not log_file: return
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_exists = os.path.isfile(log_file)
    fieldnames = list(record.keys())
    try:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            writer.writerow(record)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error appending results to CSV: {e}")


# -------------------------------------------------------------------------
# Helper Function for External Calls
# -------------------------------------------------------------------------

def evaluate_fact(
    model_path: str,
    log_file: Optional[str] = None,
    batch_size: int = 48,
    dataset: str = _DEFAULT_COUNTERFACT_PATH,
    dtype: str = "bf16",
    max_problems: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    temperature: float = 0.0,
    max_new_tokens: int = 32
) -> Dict[str, Any]:
    """
    Evaluates a model on CounterFact and optionally logs the result to a CSV.

    Args:
        model_path: Path to the HuggingFace model or local checkpoint.
        log_file: Path to CSV file to append results. If None, skips logging.
        batch_size: Inference batch size.
        dataset: Path to counterfact JSON data file.
        dtype: Model precision ("bf16", "fp16", "fp32").
        max_problems: Limit number of test samples (for debugging).
        seed: Random seed for reproducibility.
        device: "cuda" or "cpu".
        temperature: Generation temperature.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary containing hyperparameters and evaluation metrics.
    """

    make_deterministic(seed)
    print(f"Loading dataset from {dataset}...")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    selected_dtype = dtype_map.get(dtype, torch.float32)

    test_ds = load_dataset("json", data_files=dataset, split="train")
    if max_problems is not None:
        print(f"Limiting to {max_problems} problems...")
        test_ds = test_ds.select(range(min(max_problems, len(test_ds))))
    test_data_list = [row for row in test_ds]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    evaluator = FactEvaluator(
        model_path=model_path,
        device=device,
        tokenizer=tokenizer,
        dtype=selected_dtype
    )

    metrics = evaluator.run_evaluation(
        test_data_list,
        batch_size=batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    hyperparams = parse_model_hyperparams(model_path)

    result_metrics = {
        "acc_new": avg(metrics["acc_new"]) * 100,
        "acc_true": avg(metrics["acc_true"]) * 100,
        "eff_es": avg(metrics["eff_es"]) * 100,
        "eff_em": avg(metrics["eff_em"]),
        "gen_ps": avg(metrics["gen_ps"]) * 100,
        "gen_pm": avg(metrics["gen_pm"]),
        "spec_ns": avg(metrics["spec_ns"]) * 100,
        "spec_nm": avg(metrics["spec_nm"]),
    }

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        **hyperparams,
        **result_metrics,
    }

    if log_file:
        append_result(log_file, record)
        print(f"[Logged] Appended results to: {log_file}")

    del evaluator.model
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record


# -------------------------------------------------------------------------
# Main Block
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on CounterFact (Last 10%)")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--log-file", type=str, default=None, help="Path to output CSV")
    parser.add_argument("--dataset", type=str, default="../../src/data/data_source/counterfact.json")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-problems", type=int, default=None)

    args = parser.parse_args()

    evaluate_fact(
        model_path=args.model,
        log_file=args.log_file,
        batch_size=args.batch_size,
        dataset=args.dataset,
        dtype=args.dtype,
        max_problems=args.max_problems,
        seed=args.seed,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
