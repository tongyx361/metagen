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

# If there exist no ${VERL_ACTOR_CKPT_DIR}/huggingface/*.modeltensors
if [ ! -f "${VERL_ACTOR_CKPT_DIR}/huggingface/model.safetensors.index.json" ]; then
    echo "Merging actor model from checkpoint shards..."
    python "${VERL_REPO_DIR}/scripts/model_merger.py" --local_dir "${VERL_ACTOR_CKPT_DIR}"
fi

MODEL_PATH="${VERL_ACTOR_CKPT_DIR}/huggingface/model.safetensors" \
MODEL_NAME=${MODEL_NAME} \
TP_SIZE=${TP_SIZE} \
    bash scripts/launch_dp_server.sh

read -r -d '' metagen_cmd << EOF
python -m metagen.cli.metagen \
    --config-dir=configs/metagen +run=eval_reasoning_dlc \
    model=${MODEL_NAME} \
    tokenizer=${MODEL_PATH} \
    save.records_home=${SAVE_HOME} \
    save.config_home=${SAVE_HOME} \
    ${OVERRIDES} \
    2>&1 | tee ${LOG_HOME}/metagen-run-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S).log
EOF

mkdir -p "${LOG_HOME}"
eval "${metagen_cmd}"
