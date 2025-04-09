from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import hydra
import orjson
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

# TODO: fix this
from metagen.io import PathListConfig  # type: ignore

logger = getLogger(__name__)


@dataclass
class MetricConfig:
    id: str = "acc/mean"
    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordConfig:
    data_srcs: PathListConfig = MISSING
    variables: dict[str, Any] = MISSING
    metrics: list[MetricConfig] = MISSING


@dataclass
class ChartConfig:
    records: list[RecordConfig] = field(default_factory=list)
    pivot_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzeRunConfig:
    charts: list[ChartConfig] = field(default_factory=list)


cs = ConfigStore.instance()
ANALYZE_MAIN_CONFIG_NAME = "analyze_run_config"
cs.store(name=ANALYZE_MAIN_CONFIG_NAME, node=AnalyzeRunConfig)


@hydra.main(version_base=None, config_name=ANALYZE_MAIN_CONFIG_NAME)
def run_analyze(cfg: AnalyzeRunConfig) -> None:
    # Load data cache
    data_cache: dict[str, list[dict[str, Any]]] = {}
    for chart_cfg in cfg.charts:
        for record_cfg in chart_cfg.records:
            data_srcs_cfg: PathListConfig = OmegaConf.to_object(record_cfg.data_srcs)
            for data_src_path in data_srcs_cfg.paths:
                verify_records_path = data_src_path / "verify-records.jsonl"
                if verify_records_path.as_posix() in data_cache:
                    continue
                with open(verify_records_path, "rb") as f:
                    records = [orjson.loads(line) for line in f]
                data_cache[verify_records_path.as_posix()] = records

    for chart_cfg in cfg.charts:
        chart_records: list[dict[str, Any]] = []
        for record_cfg in chart_cfg.records:
            accs = []
            record_data_srcs_cfg: PathListConfig = OmegaConf.to_object(
                record_cfg.data_srcs
            )
            for data_src_path in record_data_srcs_cfg.paths:
                verify_records_path = data_src_path / "verify-records.jsonl"
                verify_records = data_cache[verify_records_path.as_posix()]
                acc = sum(record["correct"] for record in verify_records) / len(
                    verify_records
                )
                accs.append(acc)
            record = {
                **record_cfg.variables,
            }
            for metric_cfg in record_cfg.metrics:
                if metric_cfg.id == "acc/mean":
                    record[metric_cfg.name] = sum(accs) / len(accs)
                else:
                    raise ValueError(f"Unknown metric: {metric_cfg.id}")
            chart_records.append(record)

        records_df = pd.DataFrame(chart_records)
        pivot_df = records_df.pivot(**chart_cfg.pivot_kwargs)
        print(pivot_df.to_markdown())


if __name__ == "__main__":
    run_analyze()
