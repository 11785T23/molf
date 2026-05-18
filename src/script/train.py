"""
Entry point for finetuning a causal language model on the MATH and BBQ
datasets using LoRA adapters.  This script parses commandline arguments,
constructs a ``TrainingConfig`` instance, and invokes the training routine
implemented in ``trainer.py``.
"""

import draccus, os, json, uuid, shutil
from pathlib import Path
from dataclasses import asdict
from src.config import TrainingConfig
from src.trainer import BaseTrainer, MoLFTrainer
from src.evaluation.eval_helper.eval_sql import evaluate_sql
from src.evaluation.eval_helper.eval_medmcqa import evaluate_medmcqa
from src.evaluation.eval_helper.eval_fact import evaluate_fact
from src.utils.dist import is_master

# current wandb version triggers warnings
import wandb
def init_wandb(cfg):
    VERSION_STR = os.environ.get("VERSION_STR", "none")
    runid_file = Path(f"{cfg.log_dir}") / f"wandb_runid_{VERSION_STR}.json"
    if runid_file.exists():
        run_id = json.loads(runid_file.read_text())["run_id"]
    else:
        run_id = str(uuid.uuid4())+VERSION_STR
        runid_file.write_text(json.dumps({"run_id": run_id}))

    os.environ["WANDB_RUN_ID"] = run_id
    
    run = wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        dir=cfg.log_dir,
        name=cfg.run_name,
        id=run_id,
        resume=os.environ.get("WANDB_RESUME", "allow"),
        job_type=os.environ.get("WANDB_RUN_TYPE", "train"),
        config=asdict(cfg)
    )
    cfg.run = run

@draccus.wrap()
def main(cfg: TrainingConfig):
    if not cfg.only_eval:
        if cfg.report_to == "wandb":
            init_wandb(cfg=cfg)
        if cfg.mode == 'lora' or cfg.mode == 'fft':
            trainer = BaseTrainer(config=cfg)
        elif cfg.mode == 'molf':
            trainer = MoLFTrainer(config=cfg)
        
        trainer.train()
        trainer.merge_and_save_final_model(cfg.ckpt_dir)
        
        if cfg.report_to == "wandb":
            wandb.finish()

    if is_master():
        if cfg.dataset_type == "sql":
            evaluate_sql(model_path=cfg.ckpt_dir, log_file=os.environ.get("LOG_FILE", "result/molf_sql.csv"))
        elif cfg.dataset_type == "med":
            evaluate_medmcqa(model_path=cfg.ckpt_dir, log_file=os.environ.get("LOG_FILE", "result/molf_med.csv"))
        elif cfg.dataset_type == "fact":
            evaluate_fact(model_path=cfg.ckpt_dir, log_file=os.environ.get("LOG_FILE", "result/molf_fact.csv"))
            
    if cfg.clean_ckpt_at_end:
        target_dir = Path(cfg.ckpt_dir)
        # Search for the pattern inside cfg.ckpt_dir
        for path in target_dir.glob("checkpoint-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True) # rm -rf
            else:
                path.unlink(missing_ok=True) # rm

if __name__ == "__main__":
    main()