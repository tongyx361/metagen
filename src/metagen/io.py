import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Optional


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
            if self.filter_for_regex_pattern:
                matched_paths = [
                    path
                    for path in matched_paths
                    if re.match(self.filter_for_regex_pattern, path)
                ]
        paths.extend([Path(path) for path in matched_paths])
        return paths
