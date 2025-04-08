from math_verify import parse  # type: ignore[import]
from math_verify.parser import StringExtractionConfig  # type: ignore[import]

assert parse("\\boxed{A}", extraction_config=[StringExtractionConfig()]) == []
assert parse("The answer is: A", extraction_config=[StringExtractionConfig()]) == [
    "a",
    "A",
]
assert parse(
    "A", extraction_config=[StringExtractionConfig(try_extract_without_anchor=True)]
) == ["a", "A"]
