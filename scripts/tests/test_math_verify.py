from math_verify import parse, verify  # type: ignore[import]
from math_verify.parser import (  # type: ignore[import]
    LatexExtractionConfig,
    StringExtractionConfig,
)

assert parse("\\boxed{A}", extraction_config=[StringExtractionConfig()]) == []
assert parse("The answer is: A", extraction_config=[StringExtractionConfig()]) == [
    "a",
    "A",
]
assert parse(
    "A", extraction_config=[StringExtractionConfig(try_extract_without_anchor=True)]
) == ["a", "A"]

assert verify(
    gold=parse("A", extraction_config=[StringExtractionConfig()]),
    target=parse(
        "\\boxed{A}",
        extraction_config=[LatexExtractionConfig(), StringExtractionConfig()],
    ),
)

assert not verify(
    gold=parse("A", extraction_config=[StringExtractionConfig()]),
    target=parse(
        "apple",
        extraction_config=[LatexExtractionConfig(), StringExtractionConfig()],
    ),
)
