# Phase 2 — Ground Truth Construction & Dataset Upload

## Files

| File | Purpose | Run from |
|------|---------|----------|
| `food_planner_eval_ground_truth.json` | 30 annotated evaluation queries | (data file — no run needed) |
| `langfuse_upload.py` | Upload the ground truth to Langfuse as a dataset | Root of `eval-agents` repo |

---

## Ground Truth: 30 Selected Queries

Queries were selected from `data/culinary_agent_queries_200_with_ids.csv` to
cover all major failure modes and edge cases of the Food Planner agent.

| Category | Count | What it tests |
|---|---|---|
| `baseline` | 5 | Core tool chain (no restrictions) |
| `dietary_single` | 6 | `modify_recipe` for one constraint |
| `dietary_multiple` | 7 | Stacked restriction handling |
| `tight_time` | 5 | ≤ 25 min — likely `search_web` fallback |
| `large_serving` | 2 | Recipe scaling (≥ 10 people) |
| `special_diet` | 5 | Halal / keto / low-sodium / high-fiber / low-carb |

**Total: 30 items**

---

## Expected Tool Trajectory (per query)

The agent MUST always call:
1. `get_local_recipe_type` → classify the request
2. `fetch_local_recipe` → look up local DB
3. `check_cfia_recalls` → food safety check

Additional tools triggered by query characteristics:
- `modify_recipe` → dietary restrictions present **OR** serving size requires scaling
- `prepare_shopping_list` → query explicitly requests a shopping list
- `search_web` → only when `fetch_local_recipe` returns `NO_MATCH`

---

## How to Run in Coder

```bash
# From /home/coder/eval-agents/
uv run --env-file .env python EVALUATION/phase2/langfuse_upload.py

# Override dataset name:
uv run --env-file .env python EVALUATION/phase2/langfuse_upload.py \
    --dataset-name FoodPlannerEval_v2
```

---

## Expected Output

```
=======================================================
  Uploaded 30 items to dataset 'FoodPlannerEval'

  Breakdown by test category:
    baseline               5 items
    dietary_multiple       7 items
    dietary_single         6 items
    large_serving          2 items
    special_diet           5 items
    tight_time             5 items

  -> Open Langfuse > Datasets to verify.
  -> Green light: proceed to Phase 3.
=======================================================
```

---

## What to Verify in Langfuse

1. Dataset **FoodPlannerEval** exists with exactly **30 items**
2. Each item shows the query text as **Input**
3. Each item's **Expected Output** contains `constraints`, `trajectory`, and `pass_criteria`
4. Click a `tight_time` item (e.g. `query_29`) — confirm `trajectory.optional_tools` includes `search_web`
5. Click a `dietary_multiple` item (e.g. `query_32`) — confirm `trajectory.required_tools` includes `modify_recipe`

---

## Ground Truth Schema

```json
{
  "id": "query_X",
  "test_category": "dietary_single | dietary_multiple | baseline | tight_time | large_serving | special_diet",
  "input": "Recipe request (...) Please suggest...",
  "expected_output": {
    "constraints": {
      "recipe_category": "...",
      "cooking_method": "...",
      "main_ingredient": "...",
      "max_time_minutes": 50,
      "servings": 4,
      "dietary_restrictions": ["gluten-free"],
      "requires_shopping_list": false
    },
    "trajectory": {
      "required_tools": ["get_local_recipe_type", "fetch_local_recipe", "check_cfia_recalls", "modify_recipe"],
      "optional_tools": ["prepare_shopping_list", "search_web"],
      "notes": "..."
    },
    "pass_criteria": {
      "recipe_present": true,
      "respects_time_constraint": true,
      "correct_servings": true,
      "respects_dietary_restrictions": true,
      "cfia_check_performed": true,
      "shopping_list_present": false,
      "output_includes": ["..."],
      "output_must_not_include": ["..."]
    }
  }
}
```
