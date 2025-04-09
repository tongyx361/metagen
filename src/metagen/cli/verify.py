import re
from dataclasses import dataclass, field
from enum import Enum
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Union

import hydra
import orjson
from hydra.core.config_store import ConfigStore
from math_verify import parse, verify  # type: ignore[import]
from math_verify.errors import TimeoutException  # type: ignore[import]
from math_verify.parser import (  # type: ignore[import]
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
)
from omegaconf import OmegaConf
from tqdm import tqdm

from metagen.io import PathConfig  # type: ignore[import]

ParserExtractionConfig = Union[
    LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
]

logger = getLogger(__name__)


def get_unoredered_idx(lst: list[Any]) -> tuple[str, ...]:
    str_lst = [str(item) for item in lst]
    sorted_str_lst = sorted(str_lst)
    return tuple(sorted_str_lst)


@dataclass
class VerifyInputUnit:
    raw_pred: str
    raw_gold: str

    pred_extraction_configs: list[ParserExtractionConfig] = field(
        default_factory=lambda: [
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ]
    )
    gold_extraction_configs: list[ParserExtractionConfig] = field(
        default_factory=lambda: [
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ]
    )

    add_boxed_to_gold: bool = False

    golds: Optional[list[Any]] = None
    answers: Optional[list[Any]] = None

    def parse(self) -> None:
        gold_to_parse = self.raw_gold
        if self.add_boxed_to_gold and "\\boxed{" not in gold_to_parse:
            gold_to_parse = f"\\boxed{{{self.raw_gold}}}"
        self.golds = self.try_parse(gold_to_parse, self.gold_extraction_configs)
        self.answers = self.try_parse(self.raw_pred, self.pred_extraction_configs)

    def try_parse(
        self, text: str, extraction_config: ParserExtractionConfig
    ) -> Optional[Any]:
        try:
            return parse(text, extraction_config=extraction_config)
        except (Exception, TimeoutException) as e:
            logger.warning(f"Failed to parse {text=}: {e}")
            return None

    @property
    def verify_index(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return get_unoredered_idx(self.golds or []), get_unoredered_idx(
            self.answers or []
        )


def get_value_by_index(data: Union[dict, list], index: list[Any]) -> Any:
    v = data
    for idx in index:
        v = v[idx]
    return v


@dataclass
class VerifyInputGroup:
    path: Path
    units: list[VerifyInputUnit]

    @property
    def records_path(self) -> Path:
        dir = Path(self.path).parent
        return dir / "verify-records.jsonl"


class ExtractionType(Enum):
    LATEX = LatexExtractionConfig
    EXPR = ExprExtractionConfig
    STRING = StringExtractionConfig


@dataclass
class ExtractionConfig:
    extraction_type: ExtractionType
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyJobConfig:
    input: PathConfig
    pred_index: list[Any] = field(
        default_factory=lambda: ["output", "messages", -1, "content"]
    )
    pred_extraction_configs: list[ExtractionConfig] = field(
        default_factory=lambda: [
            ExtractionConfig(extraction_type=ExtractionType.LATEX),
            ExtractionConfig(extraction_type=ExtractionType.EXPR),
        ]
    )
    gold_index: list[Any] = field(
        default_factory=lambda: ["input", "sample", "ref_ans"]
    )
    add_boxed_to_gold: bool = True
    gold_extraction_configs: list[ExtractionConfig] = field(
        default_factory=lambda: [
            ExtractionConfig(extraction_type=ExtractionType.LATEX),
            ExtractionConfig(extraction_type=ExtractionType.EXPR),
        ]
    )


@dataclass
class VerifyRunConfig:
    jobs: dict[str, VerifyJobConfig]


cs = ConfigStore.instance()
VERIFY_MAIN_CONFIG_NAME = "verify_run_config"
cs.store(name=VERIFY_MAIN_CONFIG_NAME, node=VerifyRunConfig)


@hydra.main(version_base=None, config_name=VERIFY_MAIN_CONFIG_NAME)
def run_verify(cfg: VerifyRunConfig) -> None:
    logger.info(OmegaConf.to_container(cfg))

    input_groups: list[VerifyInputGroup] = []
    for job_name, job_cfg in cfg.jobs.items():
        input_cfg = job_cfg.input
        input_paths = input_cfg.paths
        if input_cfg.glob_pattern:
            matched_paths = glob(input_cfg.glob_pattern)
            if input_cfg.filter_for_regex_pattern:
                matched_paths = [
                    path
                    for path in matched_paths
                    if re.match(input_cfg.filter_for_regex_pattern, path)
                ]
            input_paths.extend([Path(path) for path in matched_paths])

        if len(input_paths) == 0:
            logger.warning(f"Found {len(input_paths)=} for {job_name}")
            continue

        logger.info(f"Loading {len(input_paths)=} for {job_name}")

        for input_path in tqdm(input_paths, desc="Loading input"):
            input_data: list[Any] = []
            with Path(input_path).open("rb") as f:
                for line in f:
                    input_data.append(orjson.loads(line))
            input_units: list[VerifyInputUnit] = []
            for input_item in input_data:
                input_unit = VerifyInputUnit(
                    raw_pred=get_value_by_index(input_item, job_cfg.pred_index),
                    raw_gold=get_value_by_index(input_item, job_cfg.gold_index),
                    pred_extraction_configs=[
                        cfg.extraction_type.value(**cfg.kwargs)
                        for cfg in job_cfg.pred_extraction_configs
                    ],
                    gold_extraction_configs=[
                        cfg.extraction_type.value(**cfg.kwargs)
                        for cfg in job_cfg.gold_extraction_configs
                    ],
                    add_boxed_to_gold=job_cfg.add_boxed_to_gold,
                )
                input_unit.parse()
                input_units.append(input_unit)
            input_groups.append(VerifyInputGroup(path=input_path, units=input_units))

    input_unit_map: dict[tuple[tuple[str, ...], tuple[str, ...]], VerifyInputUnit] = {}
    for input_group in input_groups:
        for unit in input_group.units:
            input_unit_map[unit.verify_index] = unit
    logger.info(f"{len(input_unit_map)=}")

    verify_map = {}
    for verify_index, input_unit in tqdm(input_unit_map.items(), desc="Verifying"):
        verify_map[verify_index] = verify(
            gold=input_unit.golds, target=input_unit.answers
        )

    for input_group in input_groups:
        record_group = []
        for input_unit in input_group.units:
            correct = verify_map[input_unit.verify_index]
            record = {
                "correct": correct,
                "answers": [str(answer) for answer in (input_unit.answers or [])],
                "golds": [str(gold) for gold in (input_unit.golds or [])],
                "raw_gold": input_unit.raw_gold,
                "add_boxed_to_gold": input_unit.add_boxed_to_gold,
                "raw_pred": input_unit.raw_pred,
                "pred_extraction_configs": [
                    str(cfg) for cfg in input_unit.pred_extraction_configs
                ],
                "gold_extraction_configs": [
                    str(cfg) for cfg in input_unit.gold_extraction_configs
                ],
            }
            record_group.append(record)

        with open(input_group.records_path, "wb") as f:
            for record in record_group:
                f.write(orjson.dumps(record) + b"\n")


if __name__ == "__main__":
    run_verify()
