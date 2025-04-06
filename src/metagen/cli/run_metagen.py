from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

logger = getLogger(__name__)


@dataclass
class ClientConfig:
    api_base: str = "http://localhost:8000/v1"
    model: str = MISSING


@dataclass
class DatasetConfig:
    """
    input kwargs to HuggingFace `Datasets.load_dataset`.
        c.f. https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset
    """

    path: str = MISSING
    name: Optional[str] = None
    split: str = "train"
    data_files: Optional[list[str]] = None
    kwargs: Optional[dict] = None
    field_map: Optional[dict[str, str]] = None
    drop_fields: Optional[list[str]] = None


@dataclass
class MessageConfig:
    role: str = MISSING
    content_template: str = MISSING


@dataclass
class TokenizationConfig:
    add_generation_prompt: bool = True


@dataclass
class TextSequencePreprocessingConfig:
    remove_bos: bool = True
    template: str = MISSING


@dataclass
class GenerationConfig:
    max_tokens: int = MISSING
    temperature: float = 1.0
    top_p: float = 1.0
    kwargs: Optional[dict] = None


@dataclass
class ApiRequestConfig:
    max_retries: int = 5
    gen_params: GenerationConfig = MISSING


@dataclass
class StageConfig:
    messages: dict[str, MessageConfig] = MISSING
    tokenization: TokenizationConfig = MISSING
    text_seq_preprocessing: TextSequencePreprocessingConfig = MISSING
    api_request: ApiRequestConfig = MISSING
    new_msg_preprocessing: MessageConfig = MISSING


@dataclass
class PipelineConfig:
    stages: dict[str, StageConfig] = MISSING


@dataclass
class MetaGenJobConfig:
    datasets: dict[str, DatasetConfig] = MISSING
    pipelines: dict[str, PipelineConfig] = MISSING


@dataclass
class MetaGenTaskConfig:
    client: ClientConfig = MISSING
    debug: bool = False
    dataset: DatasetConfig = MISSING
    pipeline: PipelineConfig = MISSING


@dataclass
class MetaGenRunConfig:
    client: ClientConfig = MISSING
    debug: bool = False
    jobs: dict[str, MetaGenJobConfig] = MISSING


class MetaGenRunner:
    def __init__(self, cfg: MetaGenRunConfig):
        self.cfg = cfg
        logger.info(OmegaConf.to_container(self.cfg))
        logger.info(f"{len(self.task_cfg_lst)}")

    @property
    def task_cfg_lst(self) -> list[MetaGenTaskConfig]:
        return [
            MetaGenTaskConfig(
                client=self.cfg.client,
                debug=self.cfg.debug,
                dataset=dataset_cfg,
                pipeline=pipeline_cfg,
            )
            for job_cfg in self.cfg.jobs.values()
            for dataset_cfg in job_cfg.datasets.values()
            for pipeline_cfg in job_cfg.pipelines.values()
        ]

    def run(self) -> None:
        pass


cs = ConfigStore.instance()
METAGEN_MAIN_CONFIG_NAME = "metagen_run_config"
cs.store(name=METAGEN_MAIN_CONFIG_NAME, node=MetaGenRunConfig)


@hydra.main(version_base=None, config_name=METAGEN_MAIN_CONFIG_NAME)
def run_metagen(cfg: MetaGenRunConfig) -> None:
    runner = MetaGenRunner(cfg)
    runner.run()


if __name__ == "__main__":
    run_metagen()
