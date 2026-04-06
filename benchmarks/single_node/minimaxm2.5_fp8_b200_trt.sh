#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    ISL \
    OSL \
    CONC \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

#if [[ -n "$SLURM_JOB_ID" ]]; then
#  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
#fi

echo "CONC: $CONC, ISL: $ISL, OSL: $OSL, TP: $TP, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"
MAX_NUM_TOKENS=$(( ($CONC+$ISL+64+63)/64*64 ))
MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192 ))
capture_tokens=(1 2 4 8 16 32 64 128)
capture_tokens+=( $(seq 256 256 $MAX_NUM_TOKENS))
CAPTURE_TOKENS_LIST=$(printf "%s, " "${capture_tokens[@]}")

#hf download "$MODEL"

CONFIG_FILENAME=/workspace/conf.yaml
LOG_FILENAME=/workspace/server.log
PORT=8033
cat << EOF > $CONFIG_FILENAME
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CONC
moe_config:
    backend: DEEPGEMM
enable_attention_dp: false
EOF
#torch_compile_config:
#  capture_num_tokens: [${CAPTURE_TOKENS_LIST%, }]
#  enable_piecewise_cuda_graph: true
#kv_cache_config:
#    dtype: fp8
#    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
#    enable_block_reuse: false 
#stream_interval: 10
#moe_config:
#  backend: WIDEEP
#moe_config:
    #backend: TRTLLM
    #backend: DEEPGEMM

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x

mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve $MODEL \
--max_batch_size $CONC \
--max_num_tokens $MAX_NUM_TOKENS \
--max_seq_len $MAX_MODEL_LEN \
--tp_size $TP \
--ep_size $EP_SIZE \
--trust_remote_code \
--host 0.0.0.0 \
--port $PORT \
--config $CONFIG_FILENAME \
> $LOG_FILENAME 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$LOG_FILENAME" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --bench-serving-dir /infx \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency $CONC \
    --result-filename $RESULT_FILENAME \
    --result-dir /workspace

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi


#export RUNNER_TYPE=b200
#export FRAMEWORK=trt
#export PRECISION=fp8
#export SPEC_DECODING=None
#export DISAGG=None
#export MODEL_PREFIX=minimaxm2.5
#export IMAGE=release:1.3.0rc8

# Stop GPU monitoring
stop_gpu_monitor
set +x
