# Phase 5 — Analysis & Report Generation

## Files

| File | Purpose |
|------|---------|
| `analyze_results.py` | Fetch Phase 4 scores from Langfuse → CSV + markdown report |
| `run_level_aggregator.py` | Run-level evaluator to plug into Phase 4's `run_experiment()` |
| `food_planner_eval_ground_truth.json` | Copy from `phase2/` (needed for test_category metadata) |

---

## Two Ways to Get Aggregates

### Option A — Standalone fetch (recommended)

Run after Phase 4 completes. Fetches stored scores from Langfuse.

```bash
# Copy ground truth from phase2 first:
cp EVALUATION/phase2/food_planner_eval_ground_truth.json EVALUATION/phase5/

# Then run:
uv run --env-file .env python EVALUATION/phase5/analyze_results.py \
    --run-name food_planner_v1_full_eval \
    --output-dir EVALUATION/phase5/output
```

**Produces:**
```
EVALUATION/phase5/output/
├── results_detailed.csv    ← 30 rows × (metadata + 8 metric scores)
├── results_summary.csv     ← pass rates grouped by test_category
└── analysis_report.md      ← full analysis with tables + observations
```

### Option B — Inline run-level evaluator (integrated with Phase 4)

Add to Phase 4's `run_evaluation.py` before calling `run_experiment`:

```python
import sys
sys.path.insert(0, "EVALUATION/phase5")
from run_level_aggregator import food_planner_run_level_grader

result = run_experiment(
    dataset_name,
    ...
    run_evaluators=[food_planner_run_level_grader],  # ← ADD THIS
)
```

Run-level scores then appear in Langfuse under the experiment run's
"Run Scores" section — no extra script needed.

---

## Expected Console Output (Option A)

```
=== Phase 5: Analysis ===
  Dataset  : FoodPlannerEval
  Run      : food_planner_v1_full_eval
  Output   : EVALUATION/phase5/output

=================================================================
  RESULTS SUMMARY

  Rule-based metrics:
    required_tools         [████████░░]  80% (24/30)
    output_present         [██████████]  100% (30/30)
    shopping_list          [██████████]  100% (8/8)

  LLM-judge metrics:
    dietary_compliance     [████████░░]  83% (25/30)
    time_constraint        [███████░░░]  70% (21/30)
    recipe_present         [█████████░]  93% (28/30)
    actionable_steps       [████████░░]  87% (26/30)
    serving_size           [████████░░]  80% (24/30)

  Items with ≥2 LLM-judge failures: 3

  Output files:
    .../results_detailed.csv
    .../results_summary.csv
    .../analysis_report.md

  -> Review analysis_report.md for full analysis.
  -> Green light: proceed to Phase 6 (final report + slides).
=================================================================
```

---

## Analysis Report Structure (`analysis_report.md`)

1. **Overall Score Summary** — pass rate table for all 8 metrics
2. **Pass Rates by Test Category** — matrix of category × metric
3. **Failure Analysis** — items with ≥2 LLM-judge failures (with ✓/✗ per metric)
4. **Key Observations** — auto-generated insights:
   - Weakest/strongest metric
   - Hardest/easiest test category
   - Tight-time query performance
   - Dietary compliance rate
   - Tool trajectory issues
5. **Items by Test Category** — detailed per-item score table

---

## What to Look for in the Report

| Finding | Implication |
|---|---|
| `time_constraint` low on `tight_time` items | Agent struggles with ≤25 min recipes — `search_web` fallback not always finding fast recipes |
| `dietary_compliance` low on `dietary_multiple` | Stacked restrictions not fully handled by `modify_recipe` |
| `required_tools` < 100% | Agent sometimes skips CFIA check or `modify_recipe` — trajectory issues |
| `serving_size` low on `large_serving` | Scaling logic in `modify_recipe` may have bugs for ≥10 servings |
| `recipe_present` < 90% | Agent sometimes responds with clarifying questions instead of a recipe |
