import torch
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# CounterFact is NOT bundled with this repo (the original release lives in the
# ROME repository under the MIT License). On first call we lazily fetch the
# canonical JSON from the upstream-published mirror and cache it locally; pass
# ``dataset_path`` explicitly to use a copy you have elsewhere.
_DEFAULT_COUNTERFACT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data_source", "counterfact.json"
)
# Same URL used by ``dsets/counterfact.py`` in the ROME repository
# (https://github.com/kmeng01/rome).
_COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"


def _ensure_counterfact_dataset(path: str) -> str:
    """Ensure the CounterFact JSON exists at ``path``; fetch on first use.

    The CounterFact dataset (Meng et al., NeurIPS 2022) is the property of the
    ROME authors and distributed under MIT. We don't redistribute it: this
    helper downloads it from the original source on the first call and caches
    the result at ``path`` for subsequent runs. Returns the resolved path.
    """
    path = os.fspath(path)
    if os.path.isfile(path):
        return path
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    print(f"[CounterFact] {path} not found — downloading from {_COUNTERFACT_URL}")
    torch.hub.download_url_to_file(_COUNTERFACT_URL, path)
    return path


class CounterfactDatasetBuilder:
    def __init__(self, tokenizer, dataset_path=_DEFAULT_COUNTERFACT_PATH, num_samples=None):
        # Fetch upstream JSON on first use (no-op if already cached).
        dataset_path = _ensure_counterfact_dataset(dataset_path)
        full_dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # Shuffle and select samples
        if num_samples is not None and num_samples < len(full_dataset):
            self.dataset = full_dataset.shuffle(seed=42).select(range(num_samples))
        else:
            self.dataset = full_dataset

        self.tokenizer = tokenizer
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, ex):
        rw = ex['requested_rewrite']
        subject = rw['subject'].strip()
        prompt_template = rw['prompt']
        
        # 1. Format Prompt
        full_prompt_text = prompt_template.format(subject).strip()
        
        # 2. Format Target
        # Add leading space for continuity
        target_fact = rw['target_new']['str'].strip()
        formatted_target = " " + target_fact + self.tokenizer.eos_token
        
        # 3. Tokenize
        prompt_ids = self.tokenizer.encode(full_prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(formatted_target, add_special_tokens=False)
        
        input_ids = prompt_ids + target_ids
        
        # Mask the prompt so we only train on the NEW fact
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids) 
        }

    def build_dataset(self):
        return self.dataset.map(
            self.preprocess,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing Counterfact without template",
            num_proc=4
        )
    
    def collate_fn(self, batch):
        # 1. Pad Inputs and Labels
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # 2. Attention Mask based on actual lengths
        seq_lens = torch.tensor([b["length"] for b in batch])
        max_len = input_ids_padded.shape[1]
        
        range_tensor = torch.arange(max_len).unsqueeze(0).expand(len(batch), max_len)
        attention_mask = (range_tensor < seq_lens.unsqueeze(1)).long()
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

if __name__ == "__main__":
    class Colors:
        PROMPT = '\033[93m'
        TARGET = '\033[96m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

    print(f"{Colors.BOLD}Loading Full CounterFact Dataset (NeelNanda version)...{Colors.ENDC}")
    config = {"model_name": "Qwen/Qwen2.5-1.5B", "batch_size": 4}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Initialize builder
    builder = CounterfactDatasetBuilder(tokenizer=tokenizer, num_samples=None) 
    dataset = builder.build_dataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=builder.collate_fn, shuffle=True)
    
    print(f"\nLength={config["batch_size"] * len(loader)}\n")
    
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    print(f"\n{Colors.BOLD}--- Inspecting Samples ---{Colors.ENDC}")
    
    for i in range(len(input_ids)):
        print(f"\n{Colors.BOLD}{'='*40} Sample {i+1} {'='*40}{Colors.ENDC}")
        is_target = labels[i] != -100
        is_real_token = batch["attention_mask"][i] == 1
        is_prompt = is_real_token & (~is_target)
        
        prompt_tokens = input_ids[i][is_prompt]
        target_tokens = input_ids[i][is_target]
        
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
        
        print(f"{Colors.PROMPT}[PROMPT]:\n{prompt_text}{Colors.ENDC}")
        print(f"{Colors.TARGET}[TARGET (New Fact)]:\n{target_text}{Colors.ENDC}")