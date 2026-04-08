"""Phase 5 — Run-Level Aggregator for Food Planner Evaluation.

PURPOSE:
    Provides a run_evaluators function that can be plugged into Phase 4's
    run_experiment() call to compute aggregate pass rates at the end of the
    experiment run — without needing a separate Langfuse API fetch.

    This is an ALTERNATIVE to analyze_results.py.
    Use this approach if you want aggregates computed immediately after
    the Phase 4 run completes (in the same uv command).

USAGE:
    Add to Phase 4's run_evaluation.py:

        from run_level_aggregator import food_planner_run_level_grader

        result = run_experiment(
            ...
            run_evaluators=[food_planner_run_level_grader],
        )

    The grader returns Evaluation objects that appear in Langfuse
    under the experiment run's "Run Scores" section.

PASTE TO:
    /home/coder/eval-agents/EVALUATION/phase5/run_level_aggregator.py
"""

from typing import Any

from aieng.agent_evals.evaluation.types import Evaluation, ExperimentItemResult


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────

ALL_METRICS = [
    "required_tools",
    "output_present",
    "shopping_list",
    "dietary_compliance",
    "time_constraint",
    "recipe_present",
    "actionable_steps",
    "recipe_present_strict",
    "actionable_steps_strict",
    "recipe_present_lenient",
    "actionable_steps_lenient",
    "serving_size",
]


def food_planner_run_level_grader(
    *,
    item_results: list[ExperimentItemResult],
    **kwargs: Any,
) -> list[Evaluation]:
    """Compute aggregate pass rates across all 30 Food Planner eval items.

    Runs at the end of run_experiment() as a run_evaluators entry.
    Returns one Evaluation per metric (overall pass rate) plus one
    per test_category per metric (category pass rates).

    Parameters
    ----------
    item_results : list[ExperimentItemResult]
        All item results from the experiment run.

    Returns
    -------
    list[Evaluation]
        - ``{metric}_pass_rate``         : overall pass rate (0.0–1.0)
        - ``{category}_{metric}``        : per-category pass rate
        - ``overall_pass_rate``          : mean across all 8 metrics
        - ``n_items_with_2plus_failures``: count of poorly performing items
    """
    # ── Collect scores per item ──
    item_data: list[dict] = []
    for item_result in item_results:
        scores = {ev.name: float(ev.value) for ev in item_result.evaluations
                  if ev.value is not None}

        # Get test_category from metadata
        item = item_result.item
        meta = {}
        if isinstance(item, dict):
            meta = item.get("metadata", {}) or {}
        elif hasattr(item, "metadata"):
            meta = getattr(item, "metadata", {}) or {}
        test_category = meta.get("test_category", "unknown")

        item_data.append({"test_category": test_category, "scores": scores})

    if not item_data:
        return []

    evaluations: list[Evaluation] = []

    # ── Overall pass rates ──
    all_rates: list[float] = []
    for metric in ALL_METRICS:
        values = [
            d["scores"][metric]
            for d in item_data
            if metric in d["scores"]
        ]
        if values:
            rate = sum(values) / len(values)
            all_rates.append(rate)
            evaluations.append(
                Evaluation(
                    name=f"{metric}_pass_rate",
                    value=round(rate, 3),
                    comment=f"Pass rate across {len(values)} items with '{metric}' scores.",
                    metadata={"n_valid": len(values), "n_total": len(item_data)},
                )
            )

    if all_rates:
        evaluations.append(
            Evaluation(
                name="overall_pass_rate",
                value=round(sum(all_rates) / len(all_rates), 3),
                comment=f"Mean pass rate across {len(all_rates)} metrics.",
            )
        )

    # ── Per-category pass rates ──
    categories: dict[str, list[dict]] = {}
    for d in item_data:
        cat = d["test_category"]
        categories.setdefault(cat, []).append(d)

    for cat, items in sorted(categories.items()):
        for metric in ALL_METRICS:
            values = [d["scores"][metric] for d in items if metric in d["scores"]]
            if values:
                rate = sum(values) / len(values)
                evaluations.append(
                    Evaluation(
                        name=f"{cat}__{metric}",
                        value=round(rate, 3),
                        comment=f"Category '{cat}': {metric} pass rate ({len(values)} items).",
                        metadata={"category": cat, "metric": metric, "n_valid": len(values)},
                    )
                )

    # ── Failure count ──
    llm_metrics = [
        "dietary_compliance", "time_constraint",
        "recipe_present", "actionable_steps",
        "recipe_present_strict", "actionable_steps_strict",
        "recipe_present_lenient", "actionable_steps_lenient",
        "serving_size",
    ]
    n_failures = sum(
        1
        for d in item_data
        if sum(1 for m in llm_metrics if d["scores"].get(m) == 0.0) >= 2
    )
    evaluations.append(
        Evaluation(
            name="n_items_with_2plus_failures",
            value=float(n_failures),
            comment=f"{n_failures}/{len(item_data)} items failed ≥2 LLM-judge metrics.",
            metadata={"threshold": 2, "n_total": len(item_data)},
        )
    )

    return evaluations


__all__ = ["food_planner_run_level_grader"]
