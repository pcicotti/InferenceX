#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_trtllm_minimax
#SBATCH --output=X_WORKSPACE/minimax%j/minimax_%j.out
#SBATCH --partition=X_PARTITION
#SBATCH --gpus=X_GPUS

model=X_MODEL
workspace=X_WORKSPACE
cache=X_CACHE
container_image=$SQSH_IMAGES_DIR/release-1.3.0rc8_x64.sqsh
container_name=r130rc8
container_mounts=$workspace:/workspace,$model:/model,$cache:$cache

ISL=X_ISL
OSL=X_OSL
CONC=X_CONC
TP=X_TP
EP_SIZE=X_EP_SIZE
DP_ATTENTION=X_DP_ATTENTION
RANDOM_RANGE_RATIO=0.0
RESULT_FILENAME=minimax_test_result.json,
MAX_SEQ_LEN=$(( $ISL + $OSL ))

srun -N 1 \
    --container-image=$container_image \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-workdir=/workspace \
    --mpi=pmix \
    --gpus=X_GPUS \
    --export ISL,OSL,CONC,TP,EP_SIZE,DP_ATTENTION,RANDOM_RANGE_RATIO,RESULT_FILENAME,MAX_SEQ_LEN \
    INFX/benchmarks/single_node/X_SCRIPT.sh
