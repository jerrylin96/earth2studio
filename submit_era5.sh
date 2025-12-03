#!/bin/bash -l

# ---------------------------------------------------------------------
# BU SCC SGE Directives
# ---------------------------------------------------------------------
#$ -P eb-general              # Project Name
#$ -N ERA5_Infill_Year        # Job Name
#$ -j y                       # Merge Output/Error
#$ -o logs/job_$JOB_ID_$TASK_ID.log
#$ -l h_rt=24:00:00           # 24 Hour Hard Limit
#$ -l gpus=1                  # Request 1 GPU
#$ -l gpu_c=8.0               # Capability >= 8.0 (Ensures A100/H100/H200 class)
#$ -l gpu_type=H200|A100|A40  # Request high-memory GPUs (H200 is first pref)
#$ -pe omp 4                  # 4 CPU Cores
#$ -l mem_per_core=8G         # 32GB Total RAM

# JOB ARRAY: 1959 to 2023
# The Task ID ($SGE_TASK_ID) will be the Year
#$ -t 1959-2023
#$ -tc 10                     # Max 10 GPUs running at once
# ---------------------------------------------------------------------

# 1. Environment Setup (Matched to your working script)
module load miniconda/25.3.1
module load gcc/12.2.0
module load cuda/12.8

# Activate your specific environment
conda activate cbottle_env_experimental

# Prevent Numpy/Pytorch from spawning too many threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 2. Cache Isolation (Critical for SCC)
# Use the job-specific local scratch space
export EARTH2STUDIO_CACHE="/scratch/$USER/e2_cache_$JOB_ID_$SGE_TASK_ID"
# Explicitly set Modulus cache too, as earth2studio uses it under the hood
export MODULUS_CACHE="$EARTH2STUDIO_CACHE" 

mkdir -p "$EARTH2STUDIO_CACHE"

echo "=========================================================="
echo "Job ID: $JOB_ID | Task ID (Year): $SGE_TASK_ID"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Cache: $EARTH2STUDIO_CACHE"
echo "=========================================================="

# 3. Run
# We map the Task ID directly to the Year argument
python3 fetch_infill_year.py \
    --year $SGE_TASK_ID \
    --output_dir "/projectnb/eb-general/shared_data/data/processed/era5_infilled/$SGE_TASK_ID"

# 4. Cleanup
echo "Cleaning up local cache..."
rm -rf "$EARTH2STUDIO_CACHE"
echo "Done."
