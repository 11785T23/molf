import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from src.data.prompts import MEDMCQA_PROMPT_TEMPLATE

class MedMCQADatasetBuilder:
    def __init__(self, tokenizer, dataset_path="openlifescienceai/medmcqa", num_samples=None):
        shared_cache_dir = os.getenv("HF_DATASETS_CACHE", "/data/hf_cache/datasets")
        
        print(f"Looking for dataset in: {shared_cache_dir}")

        local_snapshot_path = snapshot_download(
            repo_id=dataset_path,
            repo_type="dataset",
            cache_dir=shared_cache_dir,
        )
        print(f"Dataset at: {local_snapshot_path}")

        full_dataset = load_dataset(
            path=local_snapshot_path, 
            split="train", 
            trust_remote_code=True # deprecated
        )
        
        if num_samples and num_samples < len(full_dataset):
            self.dataset = full_dataset.shuffle(seed=42).select(range(num_samples))
        else:
            self.dataset = full_dataset

        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess(self, ex):

        full_prompt_text = MEDMCQA_PROMPT_TEMPLATE.format(
            question=ex['question'].strip(),
            opa=ex['opa'].strip(),
            opb=ex['opb'].strip(),
            opc=ex['opc'].strip(),
            opd=ex['opd'].strip()
        )

        options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        option_text = options_map.get(ex['cop'], 'A')
        target = " " + option_text + self.tokenizer.eos_token

        prompt_ids = self.tokenizer.encode(full_prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        input_ids = prompt_ids + target_ids
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
            desc="Tokenizing MedMCQA",
            num_proc=4
        )
    
    def collate_fn(self, batch):
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        
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

    print(f"{Colors.BOLD}Loading MedMCQA Dataset...{Colors.ENDC}")
    config = {"model_name": "Qwen/Qwen2.5-1.5B", "batch_size": 4}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    builder = MedMCQADatasetBuilder(tokenizer=tokenizer, dataset_path="openlifescienceai/medmcqa", num_samples=None) 
    dataset = builder.build_dataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=builder.collate_fn, shuffle=True)
    print(f"\nLength={config['batch_size'] * len(loader)}\n")
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    print(f"\n{Colors.BOLD}--- Inspecting MedMCQA Samples ---{Colors.ENDC}")
    
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
        print(f"{Colors.TARGET}[TARGET]:\n{target_text}{Colors.ENDC}")



