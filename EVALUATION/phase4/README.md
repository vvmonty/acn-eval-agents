# Phase 4 — Full Evaluation: Rule-Based + LLM-as-Judge

## Files

```
phase4/
├── run_evaluation.py         ← full experiment: 3 rule-based + 6 LLM-judge scores
├── rubrics/
│   ├── dietary_compliance.md   ← does recipe respect dietary restrictions?
│   ├── time_constraint.md      ← does recipe fit within time limit?
│   ├── recipe_completeness.md  ← recipe shape category + present + actionable steps
│   └── serving_size.md         ← does recipe serve correct number of people?
└── README.md
```

---

## Evaluators

### Rule-Based (from Phase 3 — fast, deterministic)

| Name | What it checks | Source |
|---|---|---|
| `required_tools` | All expected tools appear in tools_called | Phase 3 |
| `output_present` | Agent produced ≥50 char answer | Phase 3 |
| `shopping_list` | Shopping list present when query requests it | Phase 3 |

### LLM-as-Judge (Phase 4 — semantic, uses evaluator model)

| Name | Rubric | Metrics emitted |
|---|---|---|
| `dietary_compliance` | dietary_compliance.md | `dietary_compliance` |
| `time_constraint` | time_constraint.md | `time_constraint` |
| `recipe_completeness` | recipe_completeness.md | `recipe_completeness_category`, `recipe_present`, `actionable_steps` |
| `serving_size` | serving_size.md | `serving_size` |

**Total: 9 scores per item** (3 rule-based + 6 LLM-judge metric fields). The
`recipe_completeness_category` value is a **string** (closed enum); Langfuse or
terminal summaries may show numeric averages only for 0/1 metrics.

---

## How to Run in Coder

**Prerequisites:** Phase 3's `run_evaluation.py` must exist at
`EVALUATION/phase3/run_evaluation.py` (Phase 4 imports from it).

```bash
# From /home/coder/eval-agents/
uv run --env-file .env python EVALUATION/phase4/run_evaluation.py

# Overrides:
uv run --env-file .env python EVALUATION/phase4/run_evaluation.py \
    --dataset-name FoodPlannerEval \
    --experiment-name "food_planner_v1_full_eval" \
    --max-concurrency 3
```

**Runtime estimate:** ~45–90 min (30 agent runs + 30 × 4 LLM judge calls)

---

## Expected Output

```
=== Phase 4: FoodPlanner Full Evaluation (Rule-Based + LLM-as-Judge) ===
  Dataset      : FoodPlannerEval
  Experiment   : food_planner_v1_full_eval
  Concurrency  : 3
...

=================================================================
  Experiment complete.

  Experiment: food_planner_v1_full_eval
  Items: 30 / 30 completed

  Score Summary:
    required_tools      : X.XX / 1.0
    output_present      : X.XX / 1.0
    shopping_list       : X.XX / 1.0
    dietary_compliance  : X.XX / 1.0
    time_constraint     : X.XX / 1.0
    recipe_present      : X.XX / 1.0
    actionable_steps    : X.XX / 1.0
    serving_size        : X.XX / 1.0

  -> Open Langfuse > Datasets > FoodPlannerEval
     to inspect the 'food_planner_v1_full_eval' run.
  -> Green light: proceed to Phase 5 (analysis + final report).
=================================================================
```

(`recipe_completeness_category` may not appear as a numeric mean in this summary;
inspect per-item scores in Langfuse.)

---

## What to Verify in Langfuse

1. New experiment run `food_planner_v1_full_eval` appears under **FoodPlannerEval**
2. Each of the 30 items has **9 scores** visible (including categorical category string)
3. Click a `dietary_multiple` item (e.g. `query_32` — gluten-free + no mushrooms):
   - `dietary_compliance` should be **1** if agent respected both restrictions
4. Click a `tight_time` item (e.g. `query_86` — 20 min slow-cooker):
   - `time_constraint` may be **0** if agent couldn't find a 20-min slow-cooker recipe
5. LLM judge comments in Langfuse explain each score decision

---

## Rubric Design Notes

Each rubric:
- Is a focused markdown file with a single responsibility
- Emits **binary metrics (0 or 1)** where noted, except `recipe_completeness_category` (string enum)
- Includes a mapping table for common restriction/scenario types
- Instructs the judge to give benefit-of-the-doubt to avoid false failures
- Uses concise 1-sentence comments to explain each score

The `recipe_completeness` rubric emits **3 metrics**: a categorical
`recipe_completeness_category` (e.g. `complete_recipe`, `deferral_no_recipe`,
`constraint_conflict_clarification`) plus binary `recipe_present` and
`actionable_steps`. All other rubrics emit **1 metric** each.
