import re
from dataclasses import dataclass, field
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import Optional

logger = getLogger(__name__)


@dataclass
class PathListConfig:
    enum_paths: list[Path] = field(default_factory=list)
    glob_pattern: Optional[str] = None
    filter_for_regex_pattern: Optional[str] = None

    @property
    def paths(self) -> list[Path]:
        paths = self.enum_paths
        if self.glob_pattern:
            matched_paths = glob(self.glob_pattern)
            if len(matched_paths) == 0:
                logger.warning(f"No paths found for {self.glob_pattern=}")
            if self.filter_for_regex_pattern:
                left_paths = [
                    path
                    for path in matched_paths
                    if re.match(self.filter_for_regex_pattern, path)
                ]
                if len(left_paths) == 0:
                    logger.warning(
                        f"No paths found after applying {self.filter_for_regex_pattern=} to {matched_paths=}"
                    )
                else:
                    matched_paths = left_paths
        paths.extend([Path(path) for path in matched_paths])
        if len(paths) == 0:
            logger.warning("No paths found!")
        return paths
