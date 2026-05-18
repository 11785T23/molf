#!/bin/bash
#SBATCH --job-name=dft_param_sweep
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:RTX_PRO_6000:1     # Request 1 specific GPUs
#SBATCH --cpus-per-task=16            # Request 16 CPU per GPU
#SBATCH --mem-per-cpu=4G              # memory per cpu
#SBATCH --array=0-17                                # 3 models x 3 LRs x 2 schedulers = 18 jobs
#SBATCH --output=logs/babel/fft/dft_fft_med_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haozhant@andrew.cmu.edu

source $HOME/miniconda3/bin/activate
conda activate dft

# set hf cache
export CTIME_DATA="/data/user_data/haozhant"
export HF_HOME=/data/user_data/haozhant/.hf_cache
export HF_HUB_CACHE="${CTIME_DATA}/hf_cache"
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cd /home/haozhant/dlora

export WANDB_ENTITY="spanningtree"
export WANDB_PROJECT="dlora"
export WANDB_RUN_TYPE="med_sweep"
export WANDB_RESUME=allow

export VERSION_STR="baseline.med"

declare -A MODEL_NAME_ALIAS=( 
  ["Qwen/Qwen2.5-3B"]="qwen3B" 
  ["Qwen/Qwen2.5-1.5B"]="qwen1d5B" 
  ["google/gemma-3-1b-pt"]="gemma1B"
)

DATASET_TYPE="med"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
gpu_count=1
echo "Available GPU count: $gpu_count"


# ==========================================
# ARRAY CONFIGURATION LOGIC
# ==========================================

# Define your parameter lists
MODEL_CANDIDATES=("Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-1.5B" "google/gemma-3-1b-pt")
FFT_LRS=(0.0001 0.00005 0.00002)
SCHEDULERS=(cosine linear)

# Calculate indices from SLURM_ARRAY_TASK_ID (0-17)
# Logic: We treat this like a base-N counter
idx=$SLURM_ARRAY_TASK_ID

# 1. Get Scheduler (changes every job)
sched_idx=$(( idx % 2 ))
LR_SCHEDULER_TYPE=${SCHEDULERS[$sched_idx]}

# 2. Get FFT LR (changes every 2 jobs)
fft_idx=$(( (idx / 2) % 3 ))
FFT_LEARNING_RATE=${FFT_LRS[$fft_idx]}

# 3. Get FFT Model (changes every 6 jobs)
model_idx=$(( (idx / 6) % 3 ))
BASE_MODEL_NAME=${MODEL_CANDIDATES[$model_idx]}

# Fixed parameters
# FFT_LEARNING_RATE=0.00005
# LORA_LR=0.0005
BASE_MODEL_ALIAS=${MODEL_NAME_ALIAS[$BASE_MODEL_NAME]}
FFT_WD=0.1
LORA_WD=0.01
CKPT_DIR_ROOT="${CTIME_DATA}/dft/baseline/${BASE_MODEL_ALIAS}"

echo "--- JOB CONFIGURATION ---"
echo "Array ID: $BASE_MODEL_NAME"
echo "MODEL: $LORA_LR"
echo "FFT LR: $FFT_LEARNING_RATE"
echo "Scheduler: $LR_SCHEDULER_TYPE"
echo "-------------------------"

get_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
# Assign the free port to a variable
FREE_PORT=$(get_free_port)

echo "-----------------------------------------------------------"
echo "Detected available port: $FREE_PORT"
echo "Launching torchrun..."
echo "-----------------------------------------------------------"

TRAIN_CMD="torchrun --master_port=$FREE_PORT --nnodes 1 --nproc_per_node=$gpu_count src/script/train.py \
  --model_name_or_path $BASE_MODEL_NAME \
  --per_device_batch_size 8 \
  --num_train_epochs 2 \
  --learning_rate $FFT_LEARNING_RATE \
  --weight_decay $FFT_WD \
  --mode fft \
  --gradient_accumulation_steps 4 \
  --dataset_type $DATASET_TYPE \
  --lr_scheduler_type $LR_SCHEDULER_TYPE \
  --warm_up_ratio 0.1 \
  --clean_ckpt_at_end True \
  --ckpt_dir_root $CKPT_DIR_ROOT"

# Print the command for debugging
echo -e "\n[PRE-FLIGHT] Executing the following command via bash -c:\n"
echo "-----------------------------------------------------------"
echo "$TRAIN_CMD"
echo "-----------------------------------------------------------"
echo -e "\n"

export LOG_FILE="result/baseline/${BASE_MODEL_ALIAS}/molf_med.csv"
# Execute in a subshell
bash -c "$TRAIN_CMD"