from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class ClientConfig:
    api_base: str = "http://localhost:8000/v1"
    model: str = MISSING


@dataclass
class MetaGenJobConfig:
    client: ClientConfig = MISSING
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="metagen_job_config", node=MetaGenJobConfig)


@hydra.main(version_base=None, config_name="metagen_job_config")
def my_app(cfg: MetaGenJobConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
