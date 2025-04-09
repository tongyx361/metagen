from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    paths: list[Path] = field(default_factory=list)
    glob_pattern: Optional[str] = None
    filter_for_regex_pattern: Optional[str] = None
