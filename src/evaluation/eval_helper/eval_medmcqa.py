import argparse
import os
import random
import sys
import re
import gc
import csv
import torch
import torch._dynamo
# Give the compiler a massive cache so it doesn't hit the limit and crash
torch._dynamo.config.cache_size_limit = 1024
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from huggingface_hub import snapshot_download
from src.utils.log_utils import parse_model_hyperparams
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from src.data.prompts import MEDMCQA_PROMPT_TEMPLATE
except ImportError:
    MEDMCQA_PROMPT_TEMPLATE = (
        "Answer the medical question below by choosing the correct option letter.\n"
        "Question: {question}\n"
        "Options:\n"
        "A) {opa}\n"
        "B) {opb}\n"
        "C) {opc}\n"
        "D) {opd}\n"
        "Answer:"
    )

def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class MedMCQAEvaluator:
    def __init__(self, model_path: str, device: str, tokenizer: AutoTokenizer, dtype: torch.dtype):
        self.device = device
        self.tokenizer = tokenizer
        print(f"Loading model from {model_path} on device {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        self.total_cases = 0
        self.invalid_answer = 0

    def build_prompts(self, batch_exs: List[dict], demos: Optional[List[dict]] = None) -> List[str]:
        prompts = []
        demo_texts = []
        if demos is not None:
            for demo in demos:
                demo_prompt = MEDMCQA_PROMPT_TEMPLATE.format(
                    question=demo['question'].strip(),
                    opa=demo['opa'].strip(),
                    opb=demo['opb'].strip(),
                    opc=demo['opc'].strip(),
                    opd=demo['opd'].strip()
                )
                demo_texts.append(demo_prompt)

        demo_section = "\n".join(demo_texts)

        for ex in batch_exs:
            full_prompt_text = demo_section + "\n" + MEDMCQA_PROMPT_TEMPLATE.format(
                question=ex['question'].strip(),
                opa=ex['opa'].strip(),
                opb=ex['opb'].strip(),
                opc=ex['opc'].strip(),
                opd=ex['opd'].strip()
            )
            prompts.append(full_prompt_text)

        return prompts

    def generate_answer(self, prompts: List[str], max_new_tokens: int = 256, temperature: float = 0.0) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_len:]
        decoded_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        results = [text.strip() for text in decoded_texts]
        return results

    def extract_answer(self, outputs: List[str]) -> List[str]:
        answers = []
        for output in outputs:
            found_answer = None
            self.total_cases += 1
            if len(output) == 1 and output.upper() in ['A', 'B', 'C', 'D']:
                found_answer = output.upper()
            if not found_answer and len(output) >= 2:
                prefix_match = re.match(r"^([A-D])([\)\n\.])", output, re.IGNORECASE)
                if prefix_match:
                    found_answer = prefix_match.group(1).upper()
            if not found_answer:
                self.invalid_answer += 1
                print("-"*30)
                print(f"Invalid Answer #{self.invalid_answer}:")
                print(f"No. {self.total_cases}")
                print(output)
                print("-"*30)
                found_answer = ""
            answers.append(found_answer)
        return answers

    def print_results_table(self, final_results: Dict[str, Tuple[float, int]]):
        subject_width = 25
        acc_width = 15
        count_width = 10

        header_border = "=" * (subject_width + acc_width + count_width + 4)
        line_border = "-" * (subject_width + acc_width + count_width + 4)

        print(f"\n{header_border}")
        print(f"| {'Subject':<{subject_width}} | {'Accuracy (%)':<{acc_width}} | {'Count':<{count_width}} |")
        print(line_border)

        for key, (acc, count) in sorted(final_results.items()):
            if key != "total":
                print(f"| {key:<{subject_width}} | {acc:<{acc_width}.2f} | {count:<{count_width}} |")

        if "total" in final_results:
            acc, count = final_results["total"]
            print(line_border)
            print(f"| {'OVERALL TOTAL':<{subject_width}} | {acc:<{acc_width}.2f} | {count:<{count_width}} |")

        print(f"{header_border}\n")

    def run_evaluation(
        self,
        test_data,
        batch_size: int = 128,
        demos: Optional[List[dict]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256
    ) -> Dict[str, Tuple[float, int]]:

        metrics = {"total": []}
        test_list = [ex for ex in test_data]
        print(f"Starting MedMCQA Evaluation on {len(test_list)} samples with Batch Size {batch_size}...")

        for i in tqdm(range(0, len(test_list), batch_size)):
            batch_exs = test_list[i : i + batch_size]
            prompts = self.build_prompts(batch_exs, demos=demos)
            outputs = self.generate_answer(prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            answers = self.extract_answer(outputs)
            option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

            for ex, pred_answer in zip(batch_exs, answers):
                correct_answer = option_map.get(ex['cop'], '')
                is_correct = int(pred_answer == correct_answer)
                metrics["total"].append(is_correct)
                subject = ex.get('subject_name', 'unknown')
                if subject not in metrics:
                    metrics[subject] = []
                metrics[subject].append(is_correct)

        final_results = {}
        for key, vals in metrics.items():
            accuracy = sum(vals) / len(vals) * 100
            final_results[key] = (accuracy, len(vals))
        self.print_results_table(final_results)
        return final_results

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

def evaluate_medmcqa(
    model_path: str,
    log_file: Optional[str] = None,
    batch_size: int = 256,
    dataset: str = "openlifescienceai/medmcqa",
    dtype: str = "bf16",
    n_shots: int = 0,
    max_problems: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    temperature: float = 0.0,
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    Evaluates a model on MedMCQA and optionally logs the result to a CSV.

    Args:
        model_path: Path to the HuggingFace model or local checkpoint.
        log_file: Path to CSV file to append results. If None, skips logging.
        batch_size: Inference batch size.
        dataset: HuggingFace dataset name.
        dtype: Model precision ("bf16", "fp16", "fp32").
        n_shots: Number of few-shot examples to use.
        max_problems: Limit number of test samples (for debugging).
        seed: Random seed for reproducibility.
        device: "cuda" or "cpu".
        temperature: Generation temperature.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary containing hyperparameters and evaluation metrics.
    """

    make_deterministic(seed)
    print(f"Loading {dataset} dataset...")
    shared_cache_dir = os.getenv("HF_DATASETS_CACHE", "/data/hf_cache/datasets")
        
    print(f"Looking for dataset in: {shared_cache_dir}")

    # try:
    #     local_snapshot_path = snapshot_download(
    #         repo_id=dataset,
    #         repo_type="dataset",
    #         cache_dir=shared_cache_dir,
    #         local_files_only=True 
    #     )
    #     print(f"Found local snapshot at: {local_snapshot_path}")
    # except FileNotFoundError:
    #     raise FileNotFoundError(f"Dataset {dataset} not found in {shared_cache_dir}. Please run the download script first.")
    
    #---
    local_snapshot_path = snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        cache_dir=shared_cache_dir,
    )
    print(f"Dataset at: {local_snapshot_path}")
    #----
    

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    selected_dtype = dtype_map.get(dtype, torch.float32)

    # Load Test Data (MedMCQA uses validation split as test)
    test_ds = load_dataset(local_snapshot_path, split="validation")
    if max_problems is not None:
        print(f"Limiting to {max_problems} problems...")
        max_problems = min(max_problems, len(test_ds))
        test_ds = test_ds.select(range(max_problems))

    # Load Demos for Few-Shot
    demos = None
    if n_shots > 0:
        print(f"Using {n_shots}-shot setting...")
        train_ds = load_dataset(local_snapshot_path, split="train")
        demos = train_ds.shuffle(seed=seed).select(range(n_shots))

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    evaluator = MedMCQAEvaluator(
        model_path=model_path,
        device=device,
        tokenizer=tokenizer,
        dtype=selected_dtype
    )

    # Run Evaluation
    subject_results = evaluator.run_evaluation(
        test_ds,
        batch_size=batch_size,
        demos=demos,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    # Extract overall accuracy
    overall_acc, overall_count = subject_results.get("total", (0.0, 0))

    # Parse hyperparams from path name
    hyperparams = parse_model_hyperparams(model_path)

    # Compile Final Record
    result_metrics = {
        "accuracy": overall_acc,
        "total_count": overall_count,
        "invalid_answers": evaluator.invalid_answer,
    }

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        **hyperparams,
        **result_metrics,
    }

    # Log to CSV if requested
    if log_file:
        append_result(log_file, record)
        print(f"[Logged] Appended results to: {log_file}")

    # Clean up
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
    parser = argparse.ArgumentParser(description="Evaluate Model on MedMCQA")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--log-file", type=str, default=None, help="Path to output CSV")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="openlifescienceai/medmcqa")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--n-shots", type=int, default=0)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args()

    evaluate_medmcqa(
        model_path=args.model,
        log_file=args.log_file,
        batch_size=args.batch_size,
        dataset=args.dataset,
        dtype=args.dtype,
        n_shots=args.n_shots,
        max_problems=args.max_problems,
        seed=args.seed,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
