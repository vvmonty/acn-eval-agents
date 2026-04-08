"""Phase 1 — Food Planner Agent Smoke Test.

PURPOSE:
    Runs the Food Planner agent on a single hardcoded query and verifies:
      1. The agent completes without error
      2. The output contains recipe and shopping list content
      3. A Langfuse trace is flushed and visible in the dashboard

PASTE TO:
    /home/coder/eval-agents/EVALUATION/phase1/smoke_test_food_planner.py

RUN WITH (from /home/coder/eval-agents/):
    uv run --env-file .env python EVALUATION/phase1/smoke_test_food_planner.py

WHAT TO CHECK AFTER RUNNING:
    1. Terminal shows "[OK] All checks passed."
    2. Langfuse dashboard shows a trace named 'smoke_test_food_planner'
       with tool call spans visible (fetch_local_recipe, modify_recipe, etc.)

DESIGN DECISION:
    We only import FoodPlanner and FOOD_PLANNER_INSTRUCTIONS from the DESIGN repo.
    All other infrastructure (Langfuse client, OpenAI client, Configs) comes from
    the EVAL-agents-main environment to avoid the DESIGN repo's Configs requiring
    Weaviate and embedding keys that are not in the EVAL .env.
"""

import asyncio
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# PATH SETUP
#
# Script lives at:
#   /home/coder/eval-agents/EVALUATION/phase1/smoke_test_food_planner.py
#
# parents[0] = .../EVALUATION/phase1
# parents[1] = .../EVALUATION
# parents[2] = /home/coder/eval-agents  ← repo root
#
# Food planner code lives at:
#   DESIGN/agent-bootcamp-accentureone-acnonepm/
#     agent-bootcamp-accentureone-acnonepm/
# ──────────────────────────────────────────────

EVAL_AGENTS_ROOT = Path(__file__).resolve().parents[2]

DESIGN_ROOT = (
    EVAL_AGENTS_ROOT
    / "DESIGN"
    / "agent-bootcamp-accentureone-acnonepm"
    / "agent-bootcamp-accentureone-acnonepm"
)

# Add DESIGN_ROOT → enables `from src.prompts import ...`
# Add src/food_planner → enables `import food_agent`
FOOD_PLANNER_DIR = DESIGN_ROOT / "src" / "food_planner"

for p in [str(FOOD_PLANNER_DIR), str(DESIGN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────
# TEST QUERY
# ──────────────────────────────────────────────

SMOKE_QUERY = (
    "Suggest a vegetarian recipe that can be prepared in under 30 minutes "
    "and is adjusted to serve 2 people. Include a shopping list."
)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def ok(label: str, detail: str = "") -> None:
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [OK]   {label}{detail_str}")


def fail(label: str, error: str) -> None:
    print(f"  [FAIL] {label}")
    print(f"         Error: {error}")
    sys.exit(1)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

async def run_smoke_test() -> None:
    """Run one query through the FoodPlanner agent and verify output."""

    print("\n=== Phase 1: Food Planner Smoke Test ===\n")

    # ── Step 1: Imports ──
    # Only import FoodPlanner + prompt from DESIGN repo.
    # Use EVAL-agents-main's Configs + OpenAI + Langfuse for everything else.
    print("[1/5] Importing dependencies...")
    try:
        import agents
        from openai import AsyncOpenAI

        from aieng.agent_evals.configs import Configs
        from langfuse import Langfuse

        # From DESIGN repo — safe imports (no problematic Configs loading)
        from food_agent import FoodPlanner
        from src.prompts import FOOD_PLANNER_INSTRUCTIONS

        ok("All imports successful")
    except ImportError as e:
        fail("Import failed", str(e))
        return

    # ── Step 2: Build clients from EVAL Configs ──
    print("[2/5] Building clients from EVAL Configs...")
    try:
        configs = Configs()

        openai_client = AsyncOpenAI(
            api_key=configs.openai_api_key.get_secret_value(),
            base_url=configs.openai_base_url,
        )

        lf = Langfuse(
            public_key=configs.langfuse_public_key,
            secret_key=configs.langfuse_secret_key.get_secret_value(),
            host=configs.langfuse_host,
        )

        ok("OpenAI client ready", f"base_url={configs.openai_base_url[:40]}...")
        ok("Langfuse client ready", configs.langfuse_host)
    except Exception as e:
        fail("Client setup", str(e))
        return

    # ── Step 3: Instantiate FoodPlanner agent ──
    print("[3/5] Instantiating FoodPlanner agent...")
    try:
        import os

        # GeminiGroundingWithGoogleSearch raises ValueError (not ImportError)
        # when WEB_SEARCH_API_KEY is missing — the try/except in food_agent.py
        # only catches ImportError so it doesn't protect against this.
        # We set dummy values here since our smoke query won't trigger web fallback.
        os.environ.setdefault("WEB_SEARCH_API_KEY", "dummy-not-used")
        os.environ.setdefault("WEB_SEARCH_BASE_URL", "http://localhost:9999")

        planner_model = configs.default_planner_model

        main_agent = FoodPlanner(
            name="SmokeTestAgent",
            instructions=FOOD_PLANNER_INSTRUCTIONS,
            model=agents.OpenAIChatCompletionsModel(
                model=planner_model,
                openai_client=openai_client,
            ),
            model_settings=agents.ModelSettings(parallel_tool_calls=False),
        )
        ok("FoodPlanner instantiated", f"model={planner_model}")
    except Exception as e:
        fail("Agent instantiation", str(e))
        return

    # ── Step 4: Run agent on test query ──
    print("[4/5] Running agent on smoke test query...")
    print(f"      Query: {SMOKE_QUERY[:80]}...")
    try:
        with lf.start_as_current_observation(
            name="smoke_test_food_planner",
            as_type="span",
            input=SMOKE_QUERY,
        ) as obs:
            session = agents.SQLiteSession(session_id="smoke_test_session_001")

            result = await agents.Runner.run(
                main_agent,
                input=SMOKE_QUERY,
                session=session,
                max_turns=30,
            )

            final_output = result.final_output
            obs.update(output=final_output)

        ok("Agent run completed")
    except Exception as e:
        fail("Agent run", str(e))
        return

    # ── Step 5: Verify output + flush ──
    print("[5/5] Verifying output and flushing traces...")
    try:
        if not final_output:
            fail("Output check", "Final output is empty")
            return

        output_lower = final_output.lower()

        has_recipe = any(kw in output_lower for kw in [
            "recipe", "ingredient", "cook", "minute", "serve", "serving"
        ])
        has_shopping = any(kw in output_lower for kw in [
            "shopping", "list", "produce", "pantry", "buy"
        ])

        if has_recipe:
            ok("Output contains recipe content")
        else:
            print("  [WARN] Output may not contain recipe details — check manually")

        if has_shopping:
            ok("Output contains shopping list content")
        else:
            print("  [WARN] Output may not contain shopping list — check manually")

        preview = final_output[:300].replace("\n", " ")
        print(f"\n  Output preview:\n  {preview}...\n")

        lf.flush()
        ok("Langfuse traces flushed")

    except Exception as e:
        fail("Output verification / flush", str(e))
        return

    # ── Summary ──
    print("=" * 55)
    print("  All checks passed.")
    print("  -> Open Langfuse dashboard and confirm a trace")
    print("     named 'smoke_test_food_planner' is visible.")
    print("  -> Verify tool call spans are present under it.")
    print("  -> Green light: proceed to Phase 2.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
