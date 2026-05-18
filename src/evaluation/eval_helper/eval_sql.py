import argparse
import os
import random
import sys
import re
import gc
import sqlite3
import torch
import torch._dynamo
# Give the compiler a massive cache so it doesn't hit the limit and crash
torch._dynamo.config.cache_size_limit = 1024
import csv
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import sqlparse
from sqlglot import exp, parse_one
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from src.utils.log_utils import parse_model_hyperparams
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Ensure these imports exist in your project structure
try:
    from src.data.prompts import SQL_PROMPT_TEMPLATE, SQL_PROMPT_NO_CONTEXT
except ImportError:
    # Fallback if running outside specific project structure for testing
    SQL_PROMPT_TEMPLATE = "{sql_prompt}\n\n{sql_context}\n\n# SQL:"
    SQL_PROMPT_NO_CONTEXT = "{sql_prompt}\n\n# SQL:"

def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

class SQLExecutor:
    def execute(self, db_context: str, sql: str) -> Tuple[bool, Any]:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        res = None
        success = False
        try:
            cursor.executescript(db_context)
            cursor.execute(sql)
            upper_sql = sql.upper().strip()
            is_dml = any(upper_sql.startswith(kw) for kw in ["INSERT", "UPDATE", "DELETE"])
            if is_dml:
                table_match = re.search(r"(?:INTO|UPDATE|FROM)\s+[`\"'\[]?(\w+)[`\"'\]]?", upper_sql)
                if table_match:
                    table_name = table_match.group(1)
                    cursor.execute(f"SELECT * FROM {table_name}")
                    res = cursor.fetchall()
                else:
                    res = [("rows_affected", cursor.rowcount)]
            else:
                res = cursor.fetchall()
            success = True
        except Exception as e:
            res = str(e)
            success = False
        finally:
            conn.close()
        return success, res

    def compare_results(self, res1: Any, res2: Any) -> bool:
        if res1 is None and res2 is None:
            return True
        if res1 is None or res2 is None:
            return False
        try:
            return set(res1) == set(res2)
        except:
            return res1 == res2
    
    def compute_text_match(self, text1: str, text2: str) -> Tuple[int, float, float]:
        t1 = text1.strip().upper()
        t2 = text2.strip().upper()
        em = int(t1==t2)
        smoothie_1 = SmoothingFunction().method1
        smoothie_4 = SmoothingFunction().method4
        t1_tokens = t1.split()
        t2_tokens = t2.split()
        bleu_1 = sentence_bleu([t1_tokens], t2_tokens, smoothing_function=smoothie_1)
        bleu_4 = sentence_bleu([t1_tokens], t2_tokens, smoothing_function=smoothie_4)
        return em, bleu_1, bleu_4    

    def compute_logic_match(self, sql1: str, sql2: str) -> int:
        try:
            def normalize_tree(sql):
                tree = parse_one(sql, read="sqlite")
                new_tree = tree.transform(
                    lambda node: node.this if isinstance(node, exp.Alias) else node
                )
                return new_tree.sql(normalize=True).strip().strip(";")
            norm1 = normalize_tree(sql1)
            norm2 = normalize_tree(sql2)
            return 1 if norm1 == norm2 else 0
        except Exception:
            return 0

class SQLEvaluator:
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
        self.executor = SQLExecutor()

    def postprocess_sql(self, sql: str) -> str:
        code_block_match = re.search(r"```(?:sql)?\s+(.*?)\s+```", sql, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            sql = code_block_match.group(1)
        else:
            sql_start_keywords=[
                r"\bSELECT\b", r"\bWITH\b", r"\bINSERT\b", r"\bUPDATE\b", 
                r"\bDELETE\b", r"\bCREATE\b", r"\bALTER\b", r"\bDROP\b", 
                r"\bREPLACE\b", r"\bTRUNCATE\b"
            ]
            sql_start_pattern = re.compile("|".join(sql_start_keywords), re.IGNORECASE)
            all_matches = list(sql_start_pattern.finditer(sql))
            if all_matches:
                start_pos = all_matches[-1].start()
                sql = sql[start_pos:]
        sql = sql.split(';')[0].strip()
        sql = sql.split("# SQL:")[-1].strip()
        sql = sql.split('\n\n')[0].strip()
        try:
            return sqlparse.format(sql, reindent=False, keyword_case='upper').strip()
        except:
            return sql.strip()

    def generate_sql(self, prompts: List[str], max_new_tokens: int = 1024, temperature: float = 0.0) -> List[Tuple[str, str]]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False
            
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        results = []
        input_len = inputs["input_ids"].shape[1]
        for i in range(len(prompts)):
            generated_tokens = outputs[i][input_len:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            pred_sql = self.postprocess_sql(decoded) + ";"
            results.append((decoded.strip(), pred_sql.strip()))
        return results
    
    def build_prompt(self, ex: Dict[str, Any], demos: Optional[List[Dict[str, Any]]] = None) -> str:
        prompt_parts = []
        if demos:
            for demo in demos:
                context = demo.get('sql_context', '').strip()
                template = SQL_PROMPT_TEMPLATE if context else SQL_PROMPT_NO_CONTEXT
                demo_text = template.format(
                    sql_prompt=demo['sql_prompt'].strip(),
                    sql_context=context
                ) if context else template.format(sql_prompt=demo['sql_prompt'].strip())
                prompt_parts.append(demo_text + demo['sql'].strip() + "\n\n")
        test_context = ex.get('sql_context', '').strip()
        test_template = SQL_PROMPT_TEMPLATE if test_context else SQL_PROMPT_NO_CONTEXT
        test_text = test_template.format(
            sql_prompt=ex['sql_prompt'].strip(),
            sql_context=test_context
        ) if test_context else test_template.format(sql_prompt=ex['sql_prompt'].strip())
        prompt_parts.append(test_text)
        return "\n".join(prompt_parts)

    def run_evaluation(self, test_data, batch_size: int, demos=None, temperature: float = 0.0, max_new_tokens: int = 1024) -> Dict[str, float]:
        total, skip_gt_count, correct_exec, valid_sql_count = 0, 0, 0, 0
        total_em, total_bleu_1, total_bleu_4, total_logic_match = 0, 0, 0, 0

        test_list = [ex for ex in test_data]
        print(f"Starting SQL Evaluation on {len(test_data)} samples with Batch Size {batch_size}...")

        for i in tqdm(range(0, len(test_list), batch_size)):
            batch_exs = test_list[i : i + batch_size]
            batch_prompts = []
            valid_batch_exs = []
            for ex in batch_exs:
                gt_sql = ex['sql'].strip()
                context = ex.get('sql_context', '').strip()
                gt_ok, _ = self.executor.execute(context, gt_sql)
                if not gt_ok:
                    skip_gt_count += 1
                    continue    
                batch_prompts.append(self.build_prompt(ex, demos=demos))
                valid_batch_exs.append(ex)
            if not batch_prompts: continue
            batch_results = self.generate_sql(batch_prompts, max_new_tokens, temperature)
            
            for (decoded, pred_sql), ex in zip(batch_results, valid_batch_exs):
                gt_sql = ex['sql'].strip()
                context = ex.get('sql_context', '').strip()
                gt_ok, gt_res = self.executor.execute(context, gt_sql)
                pred_ok, pred_res = self.executor.execute(context, pred_sql)
                res_match = self.executor.compare_results(gt_res, pred_res)
                em, b1, b4 = self.executor.compute_text_match(gt_sql, pred_sql)
                logic_match = self.executor.compute_logic_match(gt_sql, pred_sql)
                
                total_logic_match += logic_match
                total_em += em
                total_bleu_1 += b1
                total_bleu_4 += b4
                if pred_ok: valid_sql_count += 1
                if pred_ok and res_match:
                    correct_exec += 1
                total += 1

                # Debug logs for first 2 to keep output cleaner when running as library
                if total <= 2:
                    print(f"\n[Sample {total}]")
                    print(f"[Generated SQL]: {decoded}")
                    print(f"[Pred]: {pred_sql}")
                    print(f"[GT]: {gt_sql}")
                    print(f"[Exec Match]: {res_match} | [Logic Match]: {logic_match}")

        acc = correct_exec / total if total > 0 else 0
        ves = valid_sql_count / total if total > 0 else 0
        avg_logic_match = total_logic_match / total if total > 0 else 0
        avg_em = total_em / total if total > 0 else 0
        avg_bleu_1 = total_bleu_1 / total if total > 0 else 0
        avg_bleu_4 = total_bleu_4 / total if total > 0 else 0
        
        print(f"Results -> ACC: {acc:.4%}, EM: {avg_em:.4%}, Logic: {avg_logic_match:.4%}")
        
        return {"acc": acc, "ves": ves, "logic_match": avg_logic_match, 
                'avg_em': avg_em, 'avg_bleu-1': avg_bleu_1, 'avg_bleu-4': avg_bleu_4}

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

def evaluate_sql(
    model_path: str,
    log_file: Optional[str] = None,
    batch_size: int = 64,
    dataset: str = "gretelai/synthetic_text_to_sql",
    dtype: str = "bf16",
    n_shots: int = 0,
    max_problems: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    temperature: float = 0.0,
    max_new_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Evaluates a SQL generation model and optionally logs the result to a CSV.
    
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
    
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    selected_dtype = dtype_map.get(dtype, torch.float32)

    # Load Test Data
    test_ds = load_dataset(dataset, split="test")
    if max_problems is not None:
        print(f"Limiting to {max_problems} problems...")
        max_problems = min(max_problems, len(test_ds))
        test_ds = test_ds.select(range(max_problems))
    
    # Load Demos for Few-Shot
    demos = None
    if n_shots > 0:
        print(f"Using {n_shots}-shot setting...")
        train_ds = load_dataset(dataset, split="train")
        demos = train_ds.shuffle(seed=seed).select(range(n_shots))

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    evaluator = SQLEvaluator(
        model_path=model_path, 
        device=device, 
        tokenizer=tokenizer, 
        dtype=selected_dtype
    )
    
    # Run Evaluation
    result_metrics = evaluator.run_evaluation(
        test_ds, 
        batch_size=batch_size, 
        demos=demos, 
        temperature=temperature, 
        max_new_tokens=max_new_tokens
    )
    
    # Parse hyperparams from path name
    hyperparams = parse_model_hyperparams(model_path)

    # Compile Final Record
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--log-file", type=str, default=None, help="Path to output CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="gretelai/synthetic_text_to_sql")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--n-shots", type=int, default=0)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    
    args = parser.parse_args()

    # Call the helper function using command line args
    evaluate_sql(
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