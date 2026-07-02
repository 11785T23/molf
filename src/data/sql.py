from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from src.data.prompts import SQL_PROMPT_NO_CONTEXT, SQL_PROMPT_TEMPLATE



class SQLDatasetBuilder:
    def __init__(self, tokenizer, dataset_name="gretelai/synthetic_text_to_sql", num_samples=None):
        # Load dataset and subsample immediately
        full_dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        
        # Shuffle and select top N to get diverse examples
        if num_samples and num_samples < len(full_dataset):
            self.dataset = full_dataset.shuffle(seed=42).select(range(num_samples))
        else:
            self.dataset = full_dataset

        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, ex):
        # 1. Format the Prompt
        # The dataset fields are usually: 'sql_prompt', 'sql_context', 'sql'
        context = ex.get('sql_context', '').strip()
        prompt_text = ""
        
        if context:
            prompt_text = SQL_PROMPT_TEMPLATE.format(
                sql_prompt=ex['sql_prompt'].strip(),
                sql_context=context
            )
        else:
            prompt_text = SQL_PROMPT_NO_CONTEXT.format(
                sql_prompt=ex['sql_prompt'].strip()
            )

        # 2. Format the Target
        formatted_target = ex['sql'].strip() + self.tokenizer.eos_token
        
        # 3. Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(formatted_target, add_special_tokens=False)
        
        input_ids = prompt_ids + target_ids
        # Mask prompt with -100
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
            desc="Tokenizing SQL",
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

    print(f"{Colors.BOLD}Loading SQL Dataset...{Colors.ENDC}")
    config = {"model_name": "Qwen/Qwen2.5-1.5B", "batch_size": 2}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    builder = SQLDatasetBuilder(tokenizer=tokenizer, num_samples=None) # Small sample for test
    dataset = builder.build_dataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=builder.collate_fn, shuffle=True)
    print(f"\nLength={config["batch_size"] * len(loader)}\n")
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    print(f"\n{Colors.BOLD}--- Inspecting SQL Samples ---{Colors.ENDC}")
    
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