#!/usr/bin/env python3
"""
Test script for Jinja2 template from eval_reasoning_dlc.yaml
Demonstrates how the multiple choice question template works with sample data.
"""

from pathlib import Path

import jinja2
import yaml


def load_template_from_yaml():
    # Path to the YAML file
    yaml_path = (
        Path(__file__).parent.parent / "configs" / "jobs" / "eval_reasoning_dlc.yaml"
    )

    # Read the YAML file
    with open(yaml_path, "r") as f:
        yaml_content = yaml.safe_load(f)

    # Extract the template from the YAML structure
    template_str = yaml_content["eval_reasoning_mmlu_pro_official_qa"]["pipelines"][
        "mmlu_pro_official_qa"
    ]["stages"]["mmlu_pro_official_qa"]["messages"]["question"]["content_template"]
    return template_str


def test_mmlu_pro_template():
    # Sample data that mimics the structure in the YAML
    sample_data = {
        "category": "Mathematics",
        "problem": "What is 2 + 2?",
        "option_texts": ["3", "4", "5", "6"],
    }

    # Load template from YAML
    template_str = load_template_from_yaml()

    # Create a Jinja2 environment
    env = jinja2.Environment()
    template = env.from_string(template_str)

    # Render the template with our sample data
    rendered = template.render(sample=sample_data)

    print("Rendered template:")
    print("=" * 80)
    print(rendered)
    print("=" * 80)


if __name__ == "__main__":
    test_mmlu_pro_template()
