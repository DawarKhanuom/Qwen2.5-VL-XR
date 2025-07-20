#!/bin/bash
#SBATCH --job-name=qwen-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 # Requesting one A100 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G # Ample system memory
#SBATCH --time=21:40:00
#SBATCH --output=logs/qwen-%j.out
#SBATCH --error=logs/qwen-%j.err

# Load environment
source ~/.bashrc
conda activate qwenvlm
echo "Conda environment: $CONDA_DEFAULT_ENV"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


module load cuda/12.1
module load ffmpeg/5.1.2/gnu-12.2.0
# Ensure this path is correct for your system and the FFmpeg module
export LD_LIBRARY_PATH=/sw/rl9g/ffmpeg/5.1.2/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH

echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Job started at $(date)"

# Move to project directory
cd /ibex/scratch/khand0b/qwen/qwen-vl-finetune

# Run training
python qwenvl/train/train_qwen.py \
    --model_name_or_path "/ibex/scratch/khand0b/qwen/Qwen2.5-VL-3B-Instruct" \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --dataset_use my_dataset \
    --output_dir ./checkpoints3wen50k \
    --cache_dir ./cache \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-7 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 4096 \
    --data_flatten True \
    --data_packing False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3
