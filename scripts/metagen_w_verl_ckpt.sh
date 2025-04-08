#!/usr/bin/env bash
set -uxo pipefail

pip install "." --user

VERL_ACTOR_CKPT_DIR=${VERL_ACTOR_CKPT_DIR-""}
[ -z "${VERL_ACTOR_CKPT_DIR}" ] && echo "VERL_ACTOR_CKPT_DIR should be set as /path/to/global_step_*/actor/" && exit 1
MODEL_NAME=${MODEL_NAME-""}
[ -z "${MODEL_NAME}" ] && echo "Better set a MODEL_NAME" && exit 1
TP_SIZE=${TP_SIZE:-1}
OVERRIDES=${OVERRIDES:-""}
VERL_REPO_DIR=${VERL_REPO_DIR-"../verl"}
SAVE_HOME=${SAVE_HOME-"${HOME}/verl/data/metagen-runs"}
LOG_HOME=${LOG_HOME-"${HOME}/verl/logs/metagen-runs"}
CACHE_ACTOR_CKPT_DIR=${CACHE_ACTOR_CKPT_DIR-""}

if [ -z "${CACHE_ACTOR_CKPT_DIR}" ]; then
    MODEL_PATH="${VERL_ACTOR_CKPT_DIR}/huggingface"
else
    mkdir -p "$(dirname "${CACHE_ACTOR_CKPT_DIR}")"
    cp -r "${VERL_ACTOR_CKPT_DIR}" "${CACHE_ACTOR_CKPT_DIR}"
    MODEL_PATH="${CACHE_ACTOR_CKPT_DIR}/huggingface"
fi

# If there exist no ${MODEL_PATH}/model.safetensors.index.json
if [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
    echo "Merging actor model from checkpoint shards..."
    python "${VERL_REPO_DIR}/scripts/model_merger.py" --local_dir "${VERL_ACTOR_CKPT_DIR}"
fi

if [ -z "${CACHE_ACTOR_CKPT_DIR}" ]; then
    cp "${MODEL_PATH}/"* "${VERL_ACTOR_CKPT_DIR}/huggingface/"
fi

MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
TP_SIZE="${TP_SIZE}" \
    bash scripts/launch_dp_server.sh

log_path="${LOG_HOME}/metagen-run-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S).log"
read -r -d '' metagen_cmd << EOF
python -m metagen.cli.metagen \
    --config-dir=configs/metagen +run=eval_reasoning_dlc \
    model=${MODEL_NAME} \
    tokenizer=${MODEL_PATH} \
    save.records_home=${SAVE_HOME} \
    save.config_home=${SAVE_HOME} \
    ${OVERRIDES} \
    2>&1 | tee ${log_path}
EOF

mkdir -p "${LOG_HOME}" && echo "${metagen_cmd}" > "${log_path}" && eval "${metagen_cmd}"
