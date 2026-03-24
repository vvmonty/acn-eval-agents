"""Phase 2 — Upload Food Planner Evaluation Dataset to Langfuse.

PURPOSE:
    Reads food_planner_eval_ground_truth.json, transforms it into the
    flat record format expected by upload_dataset_to_langfuse(), writes
    a temporary JSONL, uploads, then cleans up.

PASTE TO:
    /home/coder/eval-agents/EVALUATION/phase2/langfuse_upload.py

RUN WITH (from /home/coder/eval-agents/):
    uv run --env-file .env python EVALUATION/phase2/langfuse_upload.py

    # Override dataset name if needed:
    uv run --env-file .env python EVALUATION/phase2/langfuse_upload.py \\
        --dataset-name FoodPlannerEval_v2

WHAT TO CHECK AFTER RUNNING:
    1. Terminal shows "Uploaded 30 items to dataset 'FoodPlannerEval'"
    2. Langfuse dashboard > Datasets shows 'FoodPlannerEval' with 30 items
    3. Each item shows the recipe query as input and the constraint/trajectory
       object as expected_output

DESIGN:
    - Reuses aieng.agent_evals.langfuse.upload_dataset_to_langfuse()
      from EVAL-agents-main to stay consistent with other implementations.
    - The ground truth file uses an {_meta, items} wrapper; this script
      extracts `items` and serialises them as a temp JSONL array that
      the upload function accepts.
    - metadata per item includes: id, test_category, max_time_minutes,
      servings, dietary_restrictions — for easy Langfuse filtering.
"""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# PATH SETUP
#
# Script lives at:  /home/coder/eval-agents/EVALUATION/phase2/langfuse_upload.py
# parents[0] = .../EVALUATION/phase2
# parents[1] = .../EVALUATION
# parents[2] = /home/coder/eval-agents  ← repo root (EVAL-agents-main)
# ──────────────────────────────────────────────────────────────

EVAL_AGENTS_ROOT = Path(__file__).resolve().parents[2]

for p in [str(EVAL_AGENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ground truth lives alongside this script
GROUND_TRUTH_PATH = Path(__file__).parent / "food_planner_eval_ground_truth.json"

DEFAULT_DATASET_NAME = "FoodPlannerEval"


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def load_and_transform_ground_truth(path: Path) -> list[dict]:
    """Read the ground truth JSON and return a flat list of upload records.

    Each record has the keys expected by upload_dataset_to_langfuse:
        - input          : str   (the query text)
        - expected_output: dict  (constraints + trajectory + pass_criteria)
        - id             : str   (e.g. "query_1")
        - metadata       : dict  (test_category, max_time_minutes, etc.)

    Parameters
    ----------
    path : Path
        Absolute path to food_planner_eval_ground_truth.json

    Returns
    -------
    list[dict]
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "items" not in raw:
        raise ValueError(f"Ground truth file missing 'items' key: {path}")

    records = []
    for item in raw["items"]:
        constraints = item["expected_output"].get("constraints", {})
        record = {
            "id": item["id"],
            "input": item["input"],
            "expected_output": item["expected_output"],
            "metadata": {
                "test_category": item.get("test_category", "unknown"),
                "max_time_minutes": constraints.get("max_time_minutes"),
                "servings": constraints.get("servings"),
                "dietary_restrictions": constraints.get("dietary_restrictions", []),
                "recipe_category": constraints.get("recipe_category", ""),
                "main_ingredient": constraints.get("main_ingredient", ""),
            },
        }
        records.append(record)

    logger.info("Loaded %d items from %s", len(records), path.name)
    return records


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--ground-truth-path",
    required=False,
    default=str(GROUND_TRUTH_PATH),
    show_default=True,
    help="Path to food_planner_eval_ground_truth.json",
)
@click.option(
    "--dataset-name",
    required=False,
    default=DEFAULT_DATASET_NAME,
    show_default=True,
    help="Name of the Langfuse dataset to create/update.",
)
def cli(ground_truth_path: str, dataset_name: str) -> None:
    """Upload the Food Planner evaluation dataset to Langfuse."""
    asyncio.run(_upload(ground_truth_path, dataset_name))


async def _upload(ground_truth_path: str, dataset_name: str) -> None:
    from aieng.agent_evals.langfuse import upload_dataset_to_langfuse  # noqa: PLC0415

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)

    # ── 1. Load and transform ──
    records = load_and_transform_ground_truth(gt_path)

    # ── 2. Write flat JSON array to a temp file ──
    #    upload_dataset_to_langfuse expects a JSON file that is a list of
    #    records, each with 'input' and 'expected_output' keys.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(records, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    logger.info("Wrote %d records to temp file: %s", len(records), tmp_path)

    # ── 3. Upload via the shared harness function ──
    try:
        await upload_dataset_to_langfuse(tmp_path, dataset_name)
        logger.info("Done. Dataset '%s' is live in Langfuse.", dataset_name)
    finally:
        # Clean up temp file regardless of success/failure
        Path(tmp_path).unlink(missing_ok=True)

    # ── 4. Print summary ──
    categories: dict[str, int] = {}
    for r in records:
        cat = r["metadata"]["test_category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\n" + "=" * 55)
    print(f"  Uploaded {len(records)} items to dataset '{dataset_name}'")
    print("\n  Breakdown by test category:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:<22} {count:>3} items")
    print("\n  -> Open Langfuse > Datasets to verify.")
    print("  -> Green light: proceed to Phase 3.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    cli()
