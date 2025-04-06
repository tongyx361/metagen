import asyncio
import os
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, Awaitable, Optional, cast

import hydra
import orjson
import uvloop
from datasets import Dataset, load_dataset  # type: ignore[import]
from hydra.core.config_store import ConfigStore
from jinja2 import Template
from omegaconf import MISSING, OmegaConf
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = getLogger(__name__)


@dataclass
class ClientConfig:
    base_url: str = "http://localhost:8000/v1"  # vLLM
    api_key: str = "EMPTY"
    max_retries: int = 5
    kwargs: dict = field(default_factory=dict)


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
    kwargs: dict = field(default_factory=dict)
    column_mapping: Optional[dict[str, str]] = None
    rm_cols: Optional[list[str]] = None


@dataclass
class MessageConfig:
    role: str = MISSING
    content_template: str = MISSING


@dataclass
class TokenizationConfig:
    # tokenize: bool = False
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    kwargs: dict = field(default_factory=dict)


@dataclass
class TextSequencePreprocessingConfig:
    remove_bos: bool = True
    template: str = MISSING


@dataclass
class GenerationConfig:
    max_tokens: int = MISSING
    temperature: float = 1.0
    top_p: float = 1.0
    kwargs: dict = field(default_factory=dict)


@dataclass
class ApiRequestConfig:
    gen_params: GenerationConfig = MISSING


@dataclass
class StageConfig:
    messages: Optional[dict[str, MessageConfig]] = None
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
    seeds: list[int] = field(default_factory=lambda: [0])


@dataclass
class AsyncSchedulerConfig:
    max_concurrency: int = 1024


@dataclass
class MetaGenRunConfig:
    async_scheduler: AsyncSchedulerConfig = field(default_factory=AsyncSchedulerConfig)
    client: ClientConfig = MISSING
    model: str = MISSING
    jobs: dict[str, MetaGenJobConfig] = MISSING
    save_dir: str = "./data/metagen-records"


class AsyncScheduler:
    """Asynchronous scheduler with concurrency control."""

    def __init__(self, max_concurrency: int = 1024):
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self, task: Awaitable[Any], pbar: Optional[tqdm_asyncio] = None
    ) -> Any:
        async with self.semaphore:
            resp = await task
        if pbar:
            pbar.update(1)
        return resp


class MetaGenRunner:
    def __init__(self, cfg: MetaGenRunConfig):
        self.cfg = cfg
        logger.info(OmegaConf.to_container(self.cfg))
        self.async_scheduler = self.load_async_scheduler()
        self.client = self.load_client()
        self.tokenizer = self.load_tokenizer()
        self.dataset_cache = self.load_datasets()

    def load_async_scheduler(self) -> AsyncScheduler:
        return AsyncScheduler(self.cfg.async_scheduler.max_concurrency)

    def load_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.cfg.client.base_url,
            api_key=self.cfg.client.api_key,
            max_retries=self.cfg.client.max_retries,
            **self.cfg.client.kwargs,
        )

    def load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        return tokenizer

    def load_dataset(self, dataset_cfg: DatasetConfig) -> Dataset:
        dataset: Dataset = load_dataset(
            path=dataset_cfg.path,
            name=dataset_cfg.name,
            split=dataset_cfg.split,
            data_files=dataset_cfg.data_files,
            **dataset_cfg.kwargs,
        )
        if dataset_cfg.rm_cols is not None:
            dataset = dataset.remove_columns(column_names=dataset_cfg.rm_cols)
        if dataset_cfg.column_mapping is not None:
            dataset = dataset.rename_columns(column_mapping=dataset_cfg.column_mapping)
        return dataset

    def load_datasets(self) -> dict[str, Dataset]:
        dataset_cache = {}
        for job_cfg in self.cfg.jobs.values():
            for dataset_id, dataset_cfg in job_cfg.datasets.items():
                logger.info(f"Loading {dataset_id=}...")
                if dataset_id in dataset_cache:
                    continue
                dataset_cache[dataset_id] = self.load_dataset(dataset_cfg)
        return dataset_cache

    async def run_pipeline(
        self,
        sample: dict[str, Any],
        pipeline_cfg: PipelineConfig,
        seed: int,
        pbar: Optional[tqdm_asyncio] = None,
    ) -> dict[str, Any]:
        for stage_cfg in pipeline_cfg.stages.values():
            if stage_cfg.messages is not None:
                messages = [
                    {
                        "role": msg_cfg.role,
                        "content": Template(msg_cfg.content_template).render(
                            sample=sample
                        ),
                    }
                    for msg_cfg in stage_cfg.messages.values()
                ]
            text_seq = cast(
                str,
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=stage_cfg.tokenization.add_generation_prompt,
                    continue_final_message=stage_cfg.tokenization.continue_final_message,
                    **stage_cfg.tokenization.kwargs,
                ),
            )
            if text_seq.startswith(self.tokenizer.bos_token):
                text_seq = text_seq[len(self.tokenizer.bos_token) :]
                # NOTE: Most inference engines add a BOS token at the beginning of the text sequence
            processed_text_seq = Template(
                stage_cfg.text_seq_preprocessing.template
            ).render(text_seq=text_seq)
            gen_params = stage_cfg.api_request.gen_params
            text_api_req = self.client.completions.create(
                model=self.cfg.model,
                prompt=processed_text_seq,
                max_tokens=gen_params.max_tokens,
                temperature=gen_params.temperature,
                top_p=gen_params.top_p,
                **gen_params.kwargs,
                seed=seed,
            )
            text_api_resp = await self.async_scheduler.run(task=text_api_req, pbar=pbar)
            new_msg_content = Template(
                stage_cfg.new_msg_preprocessing.content_template
            ).render(text_api_resp=text_api_resp)
            new_msg = {
                "role": stage_cfg.new_msg_preprocessing.role,
                "content": new_msg_content,
            }
            messages.append(new_msg)

        dump_record = {
            "input": {
                "sample": sample,
                "pipeline_cfg": OmegaConf.to_container(pipeline_cfg),
            },
            "output": {
                "messages": messages,
            },
        }
        return dump_record

    async def run_task_group(
        self,
        job_id: str,
        dataset_id: str,
        pipeline_id: str,
        seed: int,
    ) -> None:
        job_cfg = self.cfg.jobs[job_id]
        dataset = self.dataset_cache[dataset_id]
        pipeline_cfg = job_cfg.pipelines[pipeline_id]
        pbar = tqdm_asyncio(
            total=len(dataset), desc=f"{dataset_id}/{pipeline_id}/{seed}"
        )
        task_group = [
            self.run_pipeline(
                sample=sample, pipeline_cfg=pipeline_cfg, seed=seed, pbar=pbar
            )
            for sample in dataset
        ]
        records = await asyncio.gather(*task_group)
        save_dir = (
            Path(self.cfg.save_dir)
            / f"{self.cfg.model.replace('/', '__')}__{dataset_id}__{pipeline_id}__{seed}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metagen-records.jsonl", "wb") as f:
            for record in records:
                f.write(orjson.dumps(record) + b"\n")
        dump_cfg = deepcopy(self.cfg)
        dump_cfg.jobs = {
            job_id: MetaGenJobConfig(
                datasets={dataset_id: job_cfg.datasets[dataset_id]},
                pipelines={pipeline_id: job_cfg.pipelines[pipeline_id]},
                seeds=[seed],
            )
        }
        with open(save_dir / "metagen-config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(dump_cfg))

    async def run(self) -> None:
        task_group_lst = []
        for job_id, job_cfg in self.cfg.jobs.items():
            for dataset_id in job_cfg.datasets.keys():
                for pipeline_id in job_cfg.pipelines.keys():
                    for seed in job_cfg.seeds:
                        task_group_lst.append(
                            self.run_task_group(
                                job_id=job_id,
                                dataset_id=dataset_id,
                                pipeline_id=pipeline_id,
                                seed=seed,
                            )
                        )
        await asyncio.gather(*task_group_lst)


cs = ConfigStore.instance()
METAGEN_MAIN_CONFIG_NAME = "metagen_run_config"
cs.store(name=METAGEN_MAIN_CONFIG_NAME, node=MetaGenRunConfig)


@hydra.main(version_base=None, config_name=METAGEN_MAIN_CONFIG_NAME)
def run_metagen(cfg: MetaGenRunConfig) -> None:
    runner = MetaGenRunner(cfg)
    uvloop.run(runner.run())


if __name__ == "__main__":
    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    run_metagen()
