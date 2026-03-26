#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_comparch_trtllm-inferencex.minimax
#SBATCH --output=/home/pcicotti/fsw/minimax/minimax%j/minimax_%j.out
#SBATCH --partition=b200-a01r
#SBATCH --account=coreai_comparch_trtllm

model=/lustre/share/coreai_dlalgo_ci/artifacts/model/minimaxai_minimax-m2_5/safetensors_mode-instruct/hf-3040beaf-nim
workspace=/home/pcicotti/fsw/minimax/minimax$SLURM_JOBID
cache=/home/pcicotti/fsw/cache
infx=/home/pcicotti/project/minimax/infx
container_image=$SQSH_IMAGES_DIR/release-1.3.0rc8_x64.sqsh
container_name=r130rc8
container_mounts=$workspace:/workspace,$infx:/infx,$model:/model,$cache:$cache

export MODEL=/model
export ISL=X_ISL
export OSL=X_OSL
export CONC=X_CONC
export TP=X_TP
export EP_SIZE=X_EP
export DP_ATTENTION=X_DP_ATTN
export RANDOM_RANGE_RATIO=0.0
export RESULT_FILENAME=minimax_test_result
export MAX_MODEL_LEN=$(( $ISL + $OSL ))

srun -N 1 \
    --container-image=$container_image \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-workdir=/workspace \
    --mpi=pmix \
    --export MODEL,ISL,OSL,CONC,TP,EP_SIZE,DP_ATTENTION,RANDOM_RANGE_RATIO,RESULT_FILENAME,MAX_MODEL_LEN \
    /infx/benchmarks/single_node/minimaxm2.5_fp8_b200_trt.sh
