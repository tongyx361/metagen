#!/usr/bin/env bash
set -uxo pipefail

MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
MODEL_NAME=${MODEL_NAME:-${MODEL_PATH}}
num_tot_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((num_tot_gpus - 1)))}
TP_SIZE=${TP_SIZE:-1}
ROUTER_PORT=${ROUTER_PORT:-8000} # For DP router
ROUTER_IP=${ROUTER_IP:-"0.0.0.0"}
# Logging
SERVER_LOG_DIR=${SERVER_LOG_DIR:-"./logs/server"}
ROUTER_LOG_DIR=${ROUTER_LOG_DIR:-"./logs/router"}
PRINT_TO_CONSOLE=${PRINT_TO_CONSOLE:-0}
DEBUG=${DEBUG:-0}
VLLM_USE_V1=${VLLM_USE_V1:-1}
USE_UV=${USE_UV:-0}

export VLLM_USE_V1

if [ "${USE_UV}" -eq 1 ]; then
    pip install uv --user
fi

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
i_server=0
num_gpus=${#gpu_ids[@]}
worker_urls=()
while [ $((i_server * TP_SIZE)) -lt "${num_gpus}" ]; do
    port=$((ROUTER_PORT + i_server + 1))
    worker_url="http://localhost:${port}"
    worker_urls+=("${worker_url}")
    # [i_server * TP_SIZE, i_server * TP_SIZE + TP_SIZE) concatenated with commas
    IFS=' ' read -r -a serve_gpu_ids <<< "${gpu_ids[@]:$((i_server * TP_SIZE)):${TP_SIZE}}" # e.g. 0 1
    serve_gpu_ids_str=$(IFS=,; echo "${serve_gpu_ids[*]}") # e.g. 0,1

    server_log_path="${SERVER_LOG_DIR}/server-$(pip list | grep vllm | awk '{print $1"-"$2}')-$(date +%Y%m%d-%H%M%S)-port${port}.log"
    read -r -d '' serve_cmd << EOF
CUDA_VISIBLE_DEVICES=${serve_gpu_ids_str} \
vllm serve "${MODEL_PATH}" \
--served-model-name "${MODEL_NAME}" \
--tensor-parallel-size ${TP_SIZE} \
--host localhost --port ${port} \
> "${server_log_path}" 2>&1  &
EOF

    if [ "${USE_UV}" -eq 1 ]; then
        serve_cmd="uvx ${serve_cmd}"
    else
        pip install vllm --user
    fi

    if [ "${DEBUG}" -eq 1 ]; then
        echo "${serve_cmd}"
    else
        mkdir -p "${SERVER_LOG_DIR}" && echo "${serve_cmd}" > "${server_log_path}" && eval "${serve_cmd}"
    fi
    i_server=$((i_server + 1))
done

read -r -d '' route_cmd << EOF
python -m sglang_router.launch_router \
--worker-urls ${worker_urls[@]} \
--host "${ROUTER_IP}" --port "${ROUTER_PORT}"
EOF

if [ "${USE_UV}" -eq 1 ]; then
    route_cmd="uv run --with sglang-router ${route_cmd}"
else
    pip install sglang-router --user
fi

router_log_path="${ROUTER_LOG_DIR}/router-$(pip list | grep sglang-router | awk '{print $1"-"$2}')-$(date +%Y%m%d-%H%M%S)-port${ROUTER_PORT}.log"
if [ "${PRINT_TO_CONSOLE}" -eq 1 ]; then
    route_cmd="${route_cmd} 2>&1 | tee ${router_log_path}"
else
    route_cmd="${route_cmd} > ${router_log_path} 2>&1 &"
fi

if [ "${DEBUG}" -eq 1 ]; then
    echo "${route_cmd}"
else
    unset http_proxy # https_proxy no_proxy # If they are hard to resolve
    mkdir -p "${ROUTER_LOG_DIR}" && echo "${route_cmd}" > "${router_log_path}" && eval "${route_cmd}"
fi
