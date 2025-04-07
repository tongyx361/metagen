#!/usr/bin/env bash
# pip install uv
set -uxo pipefail

MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
MODEL_NAME=${MODEL_NAME:-${MODEL_PATH}}
num_tot_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((num_tot_gpus - 1)))}
TP_SIZE=${TP_SIZE:-1}
BASE_PORT=${BASE_PORT:-8000} # For DP router
DEBUG=${DEBUG:-0}
IP=${IP:-"0.0.0.0"}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-"./logs/server"}
ROUTER_LOG_DIR=${ROUTER_LOG_DIR:-"./logs/router"}

read -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES//,/ }"
i_server=0
num_gpus=${#gpu_ids[@]}
worker_urls=()
while [ $((i_server * TP_SIZE)) -lt ${num_gpus} ]; do
    port=$((BASE_PORT + i_server + 1))
    worker_url="http://localhost:${port}"
    worker_urls+=("${worker_url}")
    # [i_server * TP_SIZE, i_server * TP_SIZE + TP_SIZE) concatenated with commas
    read -a serve_gpu_ids <<< "${gpu_ids[@]:$((i_server * TP_SIZE)):${TP_SIZE}}" # e.g. 0 1
    serve_gpu_ids_str=$(IFS=,; echo "${serve_gpu_ids[*]}") # e.g. 0,1

    # Define server parameters using heredoc
    read -r -d '' serve_cmd << EOF
CUDA_VISIBLE_DEVICES=${serve_gpu_ids_str} \
uvx vllm serve "${MODEL_PATH}" \
--served-model-name "${MODEL_NAME}" \
--tensor-parallel-size ${TP_SIZE} \
--host localhost --port ${port} \
> "${SERVER_LOG_DIR}/$(date +%Y%m%d-%H%M%S)-port${port}.log" 2>&1  &
EOF

    if [ "${DEBUG}" -eq 1 ]; then
        echo "${serve_cmd}"
    else
        mkdir -p "${SERVER_LOG_DIR}"
        eval "${serve_cmd}"
    fi
    i_server=$((i_server + 1))
done

read -r -d '' router_cmd << EOF
uv run --with sglang-router \
    python -m sglang_router.launch_router \
    --worker-urls ${worker_urls[@]} \
    --host "${IP}" --port "${BASE_PORT}" \
    2>&1 | tee "${ROUTER_LOG_DIR}/$(date +%Y%m%d-%H%M%S)-port${BASE_PORT}.log"
EOF

if [ "${DEBUG}" -eq 1 ]; then
    echo "${router_cmd}"
else
    unset http_proxy
    mkdir -p "${ROUTER_LOG_DIR}"
    eval "${router_cmd}"
fi
