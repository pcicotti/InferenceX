#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=coreai_trtllm_minimax
#SBATCH --output=X_JOBDIR/minimax_X_NAME_%j/output.txt
#SBATCH --partition=X_PARTITION
#SBATCH --gpus=X_GPUS

model=X_MODEL
workspace=X_JOBDIR/minimax_X_NAME_$SLURM_JOBID
cache=X_CACHE
infx=X_INFX
container_image=$SQSH_IMAGES_DIR/release-1.3.0rc9_x64.sqsh
container_name=r130rc9
container_mounts=$workspace:/workspace,$infx:/infx,$model:/model,$cache:$cache

export MODEL=/model
export ISL=X_ISL
export OSL=X_OSL
export CONC=X_CONC
export TP=X_TP
export EP_SIZE=X_EP
export DP_ATTENTION=X_DP_ATTN
export RANDOM_RANGE_RATIO=0.0
export RESULT_FILENAME=minimax_result
export MAX_MODEL_LEN=$(( $ISL + $OSL ))

export FRAMEWORK=trt
export DISAGG=None
export SPEC_DECODING=None
export MODEL_PREFIX=minimaxm2.5

srun -N 1 \
    --container-image=$container_image \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-workdir=/workspace \
    --mpi=pmix \
    --export MODEL,ISL,OSL,CONC,TP,EP_SIZE,DP_ATTENTION,RANDOM_RANGE_RATIO,RESULT_FILENAME,MAX_MODEL_LEN \
    /infx/benchmarks/single_node/X_SCRIPT.sh

export RUNNER_TYPE=b200
export PRECISION=fp8
export IMAGE=release:1.3.0rc9

srun -N 1 \
    --container-image=$container_image \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-workdir=/workspace \
    --mpi=pmix \
    --export MODEL,ISL,OSL,CONC,TP,EP_SIZE,DP_ATTENTION,RANDOM_RANGE_RATIO,RESULT_FILENAME,MAX_MODEL_LEN,RUNNER_TYPE,PRECISION,RESULT_FILENAME,IMAGE \
    python3 ../infx/utils/process_result.py $RESULT_FILENAME
