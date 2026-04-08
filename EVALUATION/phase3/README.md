# Phase 3 — Batch Agent Evaluation

## Files

| File | Purpose | Run from |
|------|---------|----------|
| `run_evaluation.py` | Run FoodPlanner on all 30 dataset items + basic evaluators | Root of `eval-agents` repo |

---

## What This Phase Does

1. Fetches all 30 items from the `FoodPlannerEval` Langfuse dataset (created in Phase 2)
2. Runs the FoodPlanner agent on each query (max 3 concurrent to avoid Gemini rate limits)
3. Extracts the **actual tool trajectory** from each run — which tools were called, in what order
4. Scores each item with **3 rule-based evaluators**:

| Evaluator | What it checks | Score |
|---|---|---|
| `required_tools` | All expected required tools were actually called | 0 or 1 |
| `output_present` | Agent produced a non-empty final answer (>50 chars) | 0 or 1 |
| `shopping_list` | Shopping list present when query explicitly requests it | 0 or 1 |

5. Sends all results + scores to Langfuse as an experiment run

---

## How to Run in Coder

```bash
# From /home/coder/eval-agents/
uv run --env-file .env python EVALUATION/phase3/run_evaluation.py

# Optional overrides:
uv run --env-file .env python EVALUATION/phase3/run_evaluation.py \
    --dataset-name FoodPlannerEval \
    --experiment-name "food_planner_v1_baseline" \
    --max-concurrency 3
```

**Note:** With 30 queries and max_concurrency=3, expect ~30–60 minutes of runtime
depending on Gemini response times. The progress is visible in the terminal.

---

## Expected Output

```
=== Phase 3: FoodPlanner Evaluation ===
  Dataset      : FoodPlannerEval
  Experiment   : food_planner_v1_baseline
  Concurrency  : 3

...agent runs...

============================================================
  Experiment complete.

  Experiment: food_planner_v1_baseline
  Items: 30 / 30 completed

  Score Summary:
    required_tools   : X.XX / 1.0
    output_present   : X.XX / 1.0
    shopping_list    : X.XX / 1.0

  -> Open Langfuse > Datasets > FoodPlannerEval
     to inspect the 'food_planner_v1_baseline' run.
  -> Check required_tools, output_present, shopping_list scores.
  -> Green light: proceed to Phase 4 (LLM-as-judge evaluators).
============================================================
```

---

## What to Verify in Langfuse

1. **Datasets > FoodPlannerEval** shows a new experiment run named `food_planner_v1_baseline`
2. Each item has 3 scores: `required_tools`, `output_present`, `shopping_list`
3. Click a `tight_time` item (e.g. `query_86`) — check if `search_web` appears in `tools_called`
4. Click a `dietary_multiple` item (e.g. `query_32`) — confirm `modify_recipe` in `tools_called`
5. Click a `baseline` item (e.g. `query_85`) — check that all 3 required tools were called

---

## What Gets Stored in Langfuse Per Item

```json
{
  "output": {
    "final_output": "Here is a recipe for ...",
    "tools_called": [
      "get_local_recipe_type",
      "fetch_local_recipe",
      "check_cfia_recalls",
      "modify_recipe",
      "prepare_shopping_list"
    ]
  },
  "scores": {
    "required_tools": 1.0,
    "output_present": 1.0,
    "shopping_list": 1.0
  }
}
```

---

## Phase 4 Preview

Phase 4 adds **LLM-as-judge evaluators** that run on top of this experiment's outputs:
- `dietary_compliance` — does the recipe respect all stated dietary restrictions?
- `time_constraint` — does the recipe stay within the stated time limit?
- `recipe_quality` — is the recipe coherent, complete, and practical?
- `cfia_safety` — was the CFIA recall check actually performed and reflected?
