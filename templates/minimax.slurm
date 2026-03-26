#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_trtllm_minimax
#SBATCH --output=X_JOBDIR/minimax_X_NAME_%j/output.txt
#SBATCH --partition=X_PARTITION
#SBATCH --gpus=X_GPUS

model=X_MODEL
workspace=X_WORKSPACE
cache=X_CACHE
infx=X_INFX
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
    /infx/benchmarks/single_node/X_SCRIPT.sh
