"""Phase 4 — Full Evaluation: Rule-Based + LLM-as-Judge Evaluators.

PURPOSE:
    Runs the FoodPlanner agent against the Langfuse dataset and applies
    ALL evaluators in a single experiment run:

    Rule-based (from Phase 3):
        required_tools    — All expected tools were called
        output_present    — Agent produced a non-empty answer
        shopping_list     — Shopping list present when required

    LLM-as-judge (new in Phase 4, using create_llm_as_judge_evaluator):
        dietary_compliance — Recipe respects all stated dietary restrictions
        time_constraint    — Recipe fits within the stated time limit
        recipe_completeness_category — Categorical recipe response shape (string enum)
        recipe_present     — Output contains a concrete recipe (binary)
        actionable_steps   — Instructions are clear enough to follow (binary)
        serving_size       — Recipe serves the correct number of people

PASTE TO:
    /home/coder/eval-agents/EVALUATION/phase4/run_evaluation.py

    Also paste the rubrics/ subfolder:
    /home/coder/eval-agents/EVALUATION/phase4/rubrics/
        dietary_compliance.md
        time_constraint.md
        recipe_completeness.md
        serving_size.md

RUN WITH (from /home/coder/eval-agents/):
    uv run --env-file .env python EVALUATION/phase4/run_evaluation.py

    # Overrides:
    uv run --env-file .env python EVALUATION/phase4/run_evaluation.py \\
        --dataset-name FoodPlannerEval \\
        --experiment-name "food_planner_v1_full_eval" \\
        --max-concurrency 3

WHAT TO CHECK AFTER RUNNING:
    1. Terminal shows experiment summary (numeric means may omit categorical scores)
    2. Langfuse > Datasets > FoodPlannerEval shows a new experiment run
    3. Each item has scores for all 9 metrics (3 rule + 6 LLM-judge fields)
    4. Compare LLM judge scores vs rule-based scores for consistency

DESIGN:
    - Imports FoodPlannerTask and rule-based evaluators from EVALUATION/phase3/
      to avoid duplicating the agent runner code.
    - LLM judges use create_llm_as_judge_evaluator() from the EVAL harness.
    - Rubric markdown files live in rubrics/ alongside this script.
    - The user-prompt template passes `final_output` (text) as the Candidate
      Output so the judge sees only the recipe text, not the full output dict.
    - max_concurrency=3 for agent runs; LLM judge calls are sequential within
      each evaluator to avoid compounding rate limit pressure.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# PATH SETUP
#
# Script lives at:  EVALUATION/phase4/run_evaluation.py
# parents[0] = EVALUATION/phase4
# parents[1] = EVALUATION
# parents[2] = /home/coder/eval-agents  ← repo root
# ──────────────────────────────────────────────────────────────

PHASE4_DIR = Path(__file__).resolve().parent
PHASE3_DIR = PHASE4_DIR.parent / "phase3"
EVAL_AGENTS_ROOT = PHASE4_DIR.parents[2]
RUBRICS_DIR = PHASE4_DIR / "rubrics"

for p in [str(PHASE3_DIR), str(EVAL_AGENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

DEFAULT_DATASET_NAME = "FoodPlannerEval"
DEFAULT_EXPERIMENT_NAME = "food_planner_v1_full_eval"
DEFAULT_MAX_CONCURRENCY = 3


# ──────────────────────────────────────────────────────────────
# CUSTOM USER-PROMPT TEMPLATE
#
# The default template passes the entire `output` dict to the judge.
# We override it to pass only the recipe text (`final_output`) so
# the judge focuses on language quality, not the internal tools dict.
# We still include the tools_called list in the Expected Output section
# so the judge has trajectory context when needed (e.g., cfia check).
# ──────────────────────────────────────────────────────────────

FOOD_PLANNER_USER_PROMPT = """\
# User Query (Input)
{input}

# Expected Constraints (Expected Output)
{expected_output}

# Agent Recipe Response (Candidate Output to Evaluate)
{output}
"""


def _recipe_only_output(output: Any) -> str:
    """Extract only the final_output text for LLM judge prompts.

    The judge should see the recipe text, not the internal tools dict.
    For context, we also append the tools_called list.
    """
    if not isinstance(output, dict):
        return str(output)
    final = output.get("final_output", "")
    tools = output.get("tools_called", [])
    return (
        f"Recipe Text:\n{final}\n\n"
        f"Tools Called (in order): {tools}"
    )


def _make_recipe_prompt(output: Any, expected_output: Any, input: Any) -> str:
    """Build the user prompt for LLM judge calls."""
    from aieng.agent_evals.evaluation.graders._utils import serialize_for_prompt
    return FOOD_PLANNER_USER_PROMPT.format(
        input=serialize_for_prompt(input),
        expected_output=serialize_for_prompt(expected_output),
        output=_recipe_only_output(output),
    )


# ──────────────────────────────────────────────────────────────
# LLM JUDGE EVALUATORS
# ──────────────────────────────────────────────────────────────

def _build_llm_judges() -> list:
    """Create all 4 LLM-as-judge evaluators from rubric files."""
    from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
    from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig

    judge_config = LLMRequestConfig(temperature=0.0, retry_max_attempts=3)

    dietary = create_llm_as_judge_evaluator(
        name="dietary_compliance",
        model_config=judge_config,
        prompt_template=FOOD_PLANNER_USER_PROMPT,
        rubric_markdown=RUBRICS_DIR / "dietary_compliance.md",
    )

    time_limit = create_llm_as_judge_evaluator(
        name="time_constraint",
        model_config=judge_config,
        prompt_template=FOOD_PLANNER_USER_PROMPT,
        rubric_markdown=RUBRICS_DIR / "time_constraint.md",
    )

    completeness = create_llm_as_judge_evaluator(
        name="recipe_completeness",
        model_config=judge_config,
        prompt_template=FOOD_PLANNER_USER_PROMPT,
        rubric_markdown=RUBRICS_DIR / "recipe_completeness.md",
    )

    serving = create_llm_as_judge_evaluator(
        name="serving_size",
        model_config=judge_config,
        prompt_template=FOOD_PLANNER_USER_PROMPT,
        rubric_markdown=RUBRICS_DIR / "serving_size.md",
    )

    return [dietary, time_limit, completeness, serving]


# ──────────────────────────────────────────────────────────────
# EVALUATOR WRAPPERS
#
# The LLM judge factory calls `prompt_template.format(input=..., output=...,
# expected_output=...)` directly. We need to intercept the `output` arg and
# pass `_recipe_only_output(output)` instead. We do this via thin wrappers
# around each judge that preprocess the output before calling the judge.
# ──────────────────────────────────────────────────────────────

def _wrap_judge_with_recipe_output(judge_fn):
    """Wrap an LLM judge to substitute the full output dict with recipe text."""
    async def wrapped(
        *,
        input: Any,  # noqa: A002
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # Replace output dict with human-readable recipe text
        recipe_output = _recipe_only_output(output)
        return await judge_fn(
            input=input,
            output=recipe_output,
            expected_output=expected_output,
            metadata=metadata,
            **kwargs,
        )
    wrapped.__name__ = judge_fn.__name__
    return wrapped


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    show_default=True,
    help="Langfuse dataset to evaluate against.",
)
@click.option(
    "--experiment-name",
    default=DEFAULT_EXPERIMENT_NAME,
    show_default=True,
    help="Name for this experiment run in Langfuse.",
)
@click.option(
    "--max-concurrency",
    default=DEFAULT_MAX_CONCURRENCY,
    type=int,
    show_default=True,
    help="Max concurrent agent runs.",
)
def cli(dataset_name: str, experiment_name: str, max_concurrency: int) -> None:
    """Run FoodPlanner full evaluation (rule-based + LLM-as-judge)."""
    asyncio.run(_main(dataset_name, experiment_name, max_concurrency))


async def _main(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int,
) -> None:
    # ── Deferred imports (needs path setup above) ──
    from aieng.agent_evals.evaluation.experiment import run_experiment
    from aieng.agent_evals.async_client_manager import AsyncClientManager

    # Import rule-based evaluators and task from Phase 3
    try:
        from run_evaluation import (  # noqa: PLC0415
            FoodPlannerTask,
            required_tools_evaluator,
            output_present_evaluator,
            shopping_list_evaluator,
        )
    except ImportError as exc:
        logger.error(
            "Could not import from Phase 3. "
            "Ensure EVALUATION/phase3/run_evaluation.py exists in Coder: %s", exc
        )
        sys.exit(1)

    print("\n=== Phase 4: FoodPlanner Full Evaluation (Rule-Based + LLM-as-Judge) ===")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Experiment   : {experiment_name}")
    print(f"  Concurrency  : {max_concurrency}")
    print(f"  Rubrics from : {RUBRICS_DIR}")
    print("")

    # ── 1. Verify rubric files exist ──
    rubric_files = [
        "dietary_compliance.md",
        "time_constraint.md",
        "recipe_completeness.md",
        "serving_size.md",
    ]
    missing_rubrics = [f for f in rubric_files if not (RUBRICS_DIR / f).exists()]
    if missing_rubrics:
        logger.error(
            "Missing rubric files in %s: %s\n"
            "Paste the rubrics/ folder from EVALUATION/phase4/rubrics/ into Coder.",
            RUBRICS_DIR,
            missing_rubrics,
        )
        sys.exit(1)
    logger.info("All 4 rubric files found.")

    # ── 2. Set up task ──
    task = FoodPlannerTask()
    try:
        task.setup()
    except Exception as exc:
        logger.error("Task setup failed: %s", exc)
        sys.exit(1)

    # ── 3. Build evaluator list ──
    raw_llm_judges = _build_llm_judges()
    wrapped_judges = [_wrap_judge_with_recipe_output(j) for j in raw_llm_judges]

    all_evaluators = [
        # Rule-based (fast, deterministic)
        required_tools_evaluator,
        output_present_evaluator,
        shopping_list_evaluator,
        # LLM-as-judge (slow, semantic)
        *wrapped_judges,
    ]

    logger.info(
        "Running experiment with %d evaluators: %s",
        len(all_evaluators),
        [fn.__name__ for fn in all_evaluators],
    )

    # ── 4. Run experiment ──
    try:
        result = run_experiment(
            dataset_name,
            name=experiment_name,
            description=(
                "Phase 4 full evaluation — FoodPlanner agent on 30 ground truth queries. "
                "3 rule-based + 6 LLM-as-judge scores "
                "(dietary_compliance, time_constraint, recipe_completeness_category, "
                "recipe_present, actionable_steps, serving_size)."
            ),
            task=task.run,
            evaluators=all_evaluators,
            max_concurrency=max_concurrency,
            metadata={
                "phase": "4",
                "agent": "FoodPlanner",
                "evaluator_types": "rule_based+llm_judge",
                "rubrics_version": "v2",
            },
        )
    except Exception as exc:
        logger.error("run_experiment failed: %s", exc)
        sys.exit(1)

    # ── 5. Print summary ──
    print("\n" + "=" * 65)
    print("  Experiment complete.")
    print(f"\n{result.format()}")
    print("\n  Score legend:")
    print("    required_tools     → all expected tools called (rule-based)")
    print("    output_present     → agent produced ≥50 char answer (rule-based)")
    print("    shopping_list      → shopping list present when needed (rule-based)")
    print("    dietary_compliance         → recipe respects dietary restrictions (LLM)")
    print("    time_constraint            → recipe fits time limit (LLM)")
    print("    recipe_completeness_category → response shape: complete / incomplete / deferral / … (LLM, string)")
    print("    recipe_present              → output contains a concrete recipe (LLM)")
    print("    actionable_steps            → instructions are clear enough to follow (LLM)")
    print("    serving_size                → recipe serves the correct number (LLM)")
    print("\n  -> Open Langfuse > Datasets > FoodPlannerEval")
    print(f"     to inspect the '{experiment_name}' run.")
    print("  -> Green light: proceed to Phase 5 (analysis + final report).")
    print("=" * 65 + "\n")

    # ── 6. Close client ──
    try:
        await AsyncClientManager.get_instance().close()
    except Exception:
        pass


if __name__ == "__main__":
    cli()
