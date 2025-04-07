from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

logger = getLogger(__name__)


@dataclass
class VerifyConfig:
    input_file_pattern: str = "./data/metagen-records/*/metagen-records.jsonl"
    input_files: list[str] = field(default_factory=list)
    pred_path: list[Any] = field(default_factory=list)


cs = ConfigStore.instance()
VERIFY_MAIN_CONFIG_NAME = "verify_config"
cs.store(name=VERIFY_MAIN_CONFIG_NAME, node=VerifyConfig)


@hydra.main(version_base=None, config_name=VERIFY_MAIN_CONFIG_NAME)
def run_verify(cfg: VerifyConfig) -> None:
    logger.info(OmegaConf.to_container(cfg))


if __name__ == "__main__":
    run_verify()
