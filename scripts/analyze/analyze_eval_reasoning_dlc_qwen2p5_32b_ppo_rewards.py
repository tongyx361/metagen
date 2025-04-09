from metagen.cli.analyze import (
    AnalyzeRunConfig,
    ChartConfig,
    MetricConfig,
    RecordConfig,
    run_analyze,
)
from metagen.data import PathListConfig


def create_analyze_cfg_for_eval_reasoning_dlc_qwen2p5_32b_ppo_rewards() -> (
    AnalyzeRunConfig
):
    """Create the AnalyzeRunConfig programmatically."""

    # Common configuration for metrics
    accuracy_metric = MetricConfig(id="acc/mean", name="Accuracy")
    pivot_kwargs = {
        "index": ["Dataset", "Reward Design"],  # Use Dataset and Reward as multi-index
        "columns": ["Step"],
        "values": "Accuracy",
    }

    # Create a single chart with all datasets
    caption = "Accuracy curves using different reward designs for PPO from the `Qwen2.5-32B` pretrained model."
    records: list[RecordConfig] = []

    for dataset_name, dataset_id, pipeline_pfx in [
        ("MATH-500", "math_500", ""),
        ("AIME 2024", "aime_2024", ""),
        ("TheoremQA", "theorem_qa", ""),
        ("MMLU-Pro-1k", "mmlu_pro_1k", "mmlu_pro_official_prompt_"),
    ]:
        for reward_name, reward_id in [
            ("Classic", "classic"),
            ("Cosine", "cosine"),
            ("Cosine + Rep. Pen.", "cosine-rep-pen"),
        ]:
            for step in [40, 80, 120, 160, 200]:
                record = RecordConfig(
                    variables={
                        "Dataset": dataset_name,
                        "Reward Design": reward_name,
                        "Step": step,
                    },
                    data_srcs=PathListConfig(
                        glob_pattern=f"./data/metagen-runs/demystify-long-cot__qwen2.5-32b-ppo-{reward_id}-step{step}__{dataset_id}__{pipeline_pfx}cot_boxed_qa_resp16k_t0p7_p0p95__*"
                    ),
                    metrics=[accuracy_metric],
                )
                records.append(record)

    # Create a single chart with all records
    chart = ChartConfig(
        caption=caption,
        factor=100,
        fmt="{:.1f}",
        pivot_kwargs=pivot_kwargs,
        records=records,
    )

    # Create and return the main configuration with a single chart
    return AnalyzeRunConfig(charts=[chart])


if __name__ == "__main__":
    cfg = create_analyze_cfg_for_eval_reasoning_dlc_qwen2p5_32b_ppo_rewards()
    run_analyze(cfg)
