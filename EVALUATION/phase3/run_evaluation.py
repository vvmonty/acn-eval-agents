"""Phase 3 — Run FoodPlanner Agent on Langfuse Dataset + Basic Evaluators.

PURPOSE:
    Runs the FoodPlanner agent against every item in the 'FoodPlannerEval'
    Langfuse dataset, records actual tool trajectories, and scores each item
    with three rule-based evaluators:

        1. required_tools   — All expected required tools were called
        2. output_present   — Agent produced a non-empty final answer
        3. shopping_list    — Shopping list present when query requires it

    LLM-as-judge scoring (dietary compliance, quality, etc.) is Phase 4.

PASTE TO:
    /home/coder/eval-agents/EVALUATION/phase3/run_evaluation.py

RUN WITH (from /home/coder/eval-agents/):
    uv run --env-file .env python EVALUATION/phase3/run_evaluation.py

    # Override dataset or experiment name:
    uv run --env-file .env python EVALUATION/phase3/run_evaluation.py \\
        --dataset-name FoodPlannerEval \\
        --experiment-name "food_planner_v1_baseline" \\
        --max-concurrency 3

WHAT TO CHECK AFTER RUNNING:
    1. Terminal shows "Experiment complete" with scores summary
    2. Langfuse dashboard > Datasets > FoodPlannerEval shows an experiment run
    3. Each item has scores for required_tools, output_present, shopping_list
    4. Click any item to inspect actual_tools_called vs required_tools in output

DESIGN:
    - Uses run_experiment() from aieng.agent_evals.evaluation.experiment to
      stay consistent with the EVAL-agents-main harness.
    - FoodPlanner is imported from the DESIGN repo (same path setup as Phase 1
      smoke test). All infra (Configs, OpenAI client) comes from EVAL env.
    - Each dataset item gets an isolated SQLiteSession (no context bleed).
    - Tool trajectory is extracted from result.new_items using the raw_item
      function_call type discriminator in the openai-agents SDK.
    - max_concurrency defaults to 3 to avoid Gemini rate limits.
"""

import asyncio
import logging
import os
import sys
import uuid
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
# Script lives at:  /home/coder/eval-agents/EVALUATION/phase3/run_evaluation.py
# parents[0] = .../EVALUATION/phase3
# parents[1] = .../EVALUATION
# parents[2] = /home/coder/eval-agents      ← EVAL repo root
#
# DESIGN repo lives at:
#   /home/coder/eval-agents/DESIGN/
#     agent-bootcamp-accentureone-acnonepm/
#       agent-bootcamp-accentureone-acnonepm/
# ──────────────────────────────────────────────────────────────

EVAL_AGENTS_ROOT = Path(__file__).resolve().parents[2]

DESIGN_ROOT = (
    EVAL_AGENTS_ROOT
    / "DESIGN"
    / "agent-bootcamp-accentureone-acnonepm"
    / "agent-bootcamp-accentureone-acnonepm"
)
FOOD_PLANNER_DIR = DESIGN_ROOT / "src" / "food_planner"

for p in [str(FOOD_PLANNER_DIR), str(DESIGN_ROOT), str(EVAL_AGENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

DEFAULT_DATASET_NAME = "FoodPlannerEval"
DEFAULT_EXPERIMENT_NAME = "food_planner_v1_baseline"
DEFAULT_MAX_CONCURRENCY = 3

SHOPPING_LIST_KEYWORDS = [
    "shopping", "shopping list", "buy", "purchase",
    "grocery", "groceries", "produce", "pantry",
]


# ──────────────────────────────────────────────────────────────
# TRAJECTORY EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_tools_called(run_result: Any) -> list[str]:
    """Extract the ordered list of tool names called during an agent run.

    Iterates over result.new_items and picks out function tool calls
    using the raw_item.type == 'function_call' discriminator from the
    openai-agents SDK.

    Parameters
    ----------
    run_result : agents.RunResult
        The result object returned by agents.Runner.run().

    Returns
    -------
    list[str]
        Tool names in call order (may include duplicates if a tool was
        called more than once).
    """
    tools_called: list[str] = []

    for item in getattr(run_result, "new_items", []):
        raw = getattr(item, "raw_item", None)
        if raw is None:
            continue
        # openai-agents SDK: function calls have raw_item.type == 'function_call'
        if getattr(raw, "type", None) == "function_call":
            tool_name = getattr(raw, "name", "unknown")
            tools_called.append(tool_name)

    return tools_called


# ──────────────────────────────────────────────────────────────
# TASK
# ──────────────────────────────────────────────────────────────

class FoodPlannerTask:
    """Callable task that runs the FoodPlanner agent on one dataset item.

    Instantiated once and reused across all 30 items so the model and
    client objects are created only once.
    """

    def __init__(self):
        # Deferred — set up in async context after imports are verified.
        self._agent_cls = None
        self._instructions = None
        self._agents_module = None
        self._openai_client = None
        self._planner_model = None

    def setup(self):
        """Import and wire up all agent dependencies.

        Called once before run_experiment so import errors surface early.
        """
        import agents as agents_mod
        from openai import AsyncOpenAI
        from aieng.agent_evals.configs import Configs
        from food_agent import FoodPlanner
        from src.prompts import FOOD_PLANNER_INSTRUCTIONS

        # Dummy values so FoodPlanner.__post_init__ doesn't raise on
        # GeminiGroundingWithGoogleSearch construction.
        os.environ.setdefault("WEB_SEARCH_API_KEY", "dummy-not-used")
        os.environ.setdefault("WEB_SEARCH_BASE_URL", "http://localhost:9999")

        configs = Configs()
        openai_client = AsyncOpenAI(
            api_key=configs.openai_api_key.get_secret_value(),
            base_url=configs.openai_base_url,
        )

        self._agents_module = agents_mod
        self._agent_cls = FoodPlanner
        self._instructions = FOOD_PLANNER_INSTRUCTIONS
        self._openai_client = openai_client
        self._planner_model = configs.default_planner_model

        logger.info("FoodPlannerTask ready (model=%s)", self._planner_model)

    async def run(self, *, item: Any, **kwargs: Any) -> dict[str, Any]:
        """Run FoodPlanner on one Langfuse dataset item.

        Parameters
        ----------
        item : DatasetItemClient | LocalExperimentItem
            Langfuse dataset item. item.input is the query string.

        Returns
        -------
        dict with keys:
            final_output     : str   — agent's final answer
            tools_called     : list  — tool names in call order
        """
        agents = self._agents_module

        # Retrieve query text
        query = item["input"] if isinstance(item, dict) else item.input

        # Fresh agent instance per item (stateless between items)
        agent = self._agent_cls(
            name="FoodPlannerEvalAgent",
            instructions=self._instructions,
            model=agents.OpenAIChatCompletionsModel(
                model=self._planner_model,
                openai_client=self._openai_client,
            ),
            model_settings=agents.ModelSettings(parallel_tool_calls=False),
        )

        # Isolated session per item — no context bleed across queries
        session_id = f"eval_phase3_{uuid.uuid4().hex[:12]}"
        session = agents.SQLiteSession(session_id=session_id)

        try:
            result = await agents.Runner.run(
                agent,
                input=query,
                session=session,
                max_turns=30,
            )
        except Exception as exc:
            logger.error("Agent run failed for item '%s': %s", query[:60], exc)
            return {
                "final_output": "",
                "tools_called": [],
                "error": str(exc),
            }

        tools_called = extract_tools_called(result)
        logger.info(
            "Item done. tools=%s | output_len=%d",
            tools_called,
            len(result.final_output or ""),
        )

        return {
            "final_output": result.final_output or "",
            "tools_called": tools_called,
        }


# ──────────────────────────────────────────────────────────────
# EVALUATORS  (rule-based — Phase 3)
# ──────────────────────────────────────────────────────────────

async def required_tools_evaluator(
    *,
    output: dict[str, Any],
    expected_output: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Check that every required tool was actually called.

    Scores 1.0 if all required tools appear in actual tools_called,
    0.0 otherwise. Sets comment listing any missing tools.
    """
    from langfuse.experiment import Evaluation

    required: list[str] = (
        expected_output.get("trajectory", {}).get("required_tools", [])
    )
    actual: list[str] = output.get("tools_called", [])

    missing = [t for t in required if t not in actual]
    passed = len(missing) == 0

    comment = (
        f"All {len(required)} required tools called: {required}"
        if passed
        else f"Missing tools: {missing}. Actual: {actual}"
    )

    return Evaluation(name="required_tools", value=float(passed), comment=comment)


async def output_present_evaluator(
    *,
    output: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Check that the agent produced a non-empty final output."""
    from langfuse.experiment import Evaluation

    final = output.get("final_output", "")
    passed = bool(final and len(final.strip()) > 50)
    comment = (
        f"Output present ({len(final)} chars)"
        if passed
        else f"Output missing or too short ({len(final)} chars)"
    )
    return Evaluation(name="output_present", value=float(passed), comment=comment)


async def shopping_list_evaluator(
    *,
    output: dict[str, Any],
    expected_output: dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Check shopping list is present when the query requires it."""
    from langfuse.experiment import Evaluation

    requires = (
        expected_output.get("constraints", {}).get("requires_shopping_list", False)
    )

    if not requires:
        # Not required → skip scoring (neutral 1.0)
        return Evaluation(
            name="shopping_list",
            value=1.0,
            comment="Shopping list not required for this query — skipped.",
        )

    final_lower = output.get("final_output", "").lower()
    found = any(kw in final_lower for kw in SHOPPING_LIST_KEYWORDS)
    comment = (
        "Shopping list found in output."
        if found
        else f"Shopping list required but not detected. Keywords checked: {SHOPPING_LIST_KEYWORDS}"
    )
    return Evaluation(name="shopping_list", value=float(found), comment=comment)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    show_default=True,
    help="Langfuse dataset to run evaluation against.",
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
    help="Max concurrent agent runs (keep ≤ 5 to avoid rate limits).",
)
def cli(dataset_name: str, experiment_name: str, max_concurrency: int) -> None:
    """Run FoodPlanner evaluation against Langfuse dataset."""
    asyncio.run(_main(dataset_name, experiment_name, max_concurrency))


async def _main(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int,
) -> None:
    from aieng.agent_evals.evaluation.experiment import run_experiment
    from aieng.agent_evals.async_client_manager import AsyncClientManager

    print(f"\n=== Phase 3: FoodPlanner Evaluation ===")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Experiment   : {experiment_name}")
    print(f"  Concurrency  : {max_concurrency}")
    print("")

    # ── 1. Set up task ──
    task = FoodPlannerTask()
    try:
        task.setup()
    except Exception as exc:
        logger.error("Task setup failed: %s", exc)
        sys.exit(1)

    # ── 2. Run experiment ──
    try:
        result = run_experiment(
            dataset_name,
            name=experiment_name,
            description=(
                "Phase 3 batch evaluation — FoodPlanner agent on 30 ground truth queries. "
                "Rule-based evaluators: required_tools, output_present, shopping_list."
            ),
            task=task.run,
            evaluators=[
                required_tools_evaluator,
                output_present_evaluator,
                shopping_list_evaluator,
            ],
            max_concurrency=max_concurrency,
            metadata={
                "phase": "3",
                "agent": "FoodPlanner",
                "evaluator_type": "rule_based",
            },
        )
    except Exception as exc:
        logger.error("run_experiment failed: %s", exc)
        sys.exit(1)

    # ── 3. Print summary ──
    print("\n" + "=" * 60)
    print("  Experiment complete.")
    print(f"\n{result.format()}")
    print("\n  -> Open Langfuse > Datasets > FoodPlannerEval")
    print(f"     to inspect the '{experiment_name}' run.")
    print("  -> Check required_tools, output_present, shopping_list scores.")
    print("  -> Green light: proceed to Phase 4 (LLM-as-judge evaluators).")
    print("=" * 60 + "\n")

    # ── 4. Close client ──
    try:
        await AsyncClientManager.get_instance().close()
    except Exception:
        pass


if __name__ == "__main__":
    cli()
