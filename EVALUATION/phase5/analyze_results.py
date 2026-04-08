"""Phase 5 — Analysis: Fetch Langfuse Results and Generate Report.

PURPOSE:
    Pulls experiment scores from Langfuse (written by Phase 4), joins them
    with ground truth metadata (test_category, constraints), and produces:

        results_detailed.csv    — one row per item, all 8 metric scores
        results_summary.csv     — pass rates grouped by test_category
        analysis_report.md      — full markdown analysis report

PASTE TO:
    /home/coder/eval-agents/EVALUATION/phase5/analyze_results.py

    Also paste the ground truth alongside it:
    /home/coder/eval-agents/EVALUATION/phase5/food_planner_eval_ground_truth.json
    (copy from EVALUATION/phase2/food_planner_eval_ground_truth.json)

RUN WITH (from /home/coder/eval-agents/):
    uv run --env-file .env python EVALUATION/phase5/analyze_results.py

    # Overrides:
    uv run --env-file .env python EVALUATION/phase5/analyze_results.py \\
        --dataset-name FoodPlannerEval \\
        --run-name food_planner_v1_full_eval \\
        --output-dir EVALUATION/phase5/output

WHAT TO CHECK AFTER RUNNING:
    1. Terminal prints full score table by category
    2. output/ folder contains 3 files: detailed CSV, summary CSV, markdown report
    3. Review analysis_report.md for failure patterns and recommendations

DESIGN:
    Phase 5 does NOT re-run the agent. It reads from Langfuse using:
        - lf.get_dataset_run()           → get run items (each with a trace_id)
        - lf.api.trace.get(trace_id)     → get trace with .scores list
    Scores are joined with local ground truth JSON for test_category metadata.
    pandas is used for aggregation; no other heavy dependencies needed.
"""

import json
import logging
import sys
from datetime import datetime
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
# ──────────────────────────────────────────────────────────────

PHASE5_DIR = Path(__file__).resolve().parent
EVAL_AGENTS_ROOT = PHASE5_DIR.parents[2]

for p in [str(EVAL_AGENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ground truth lives alongside this script (copied from phase2/)
GROUND_TRUTH_PATH = PHASE5_DIR / "food_planner_eval_ground_truth.json"

DEFAULT_DATASET_NAME = "FoodPlannerEval"
DEFAULT_RUN_NAME = "food_planner_v1_full_eval"
DEFAULT_OUTPUT_DIR = PHASE5_DIR / "output"

# All metrics emitted across Phase 3 (rule-based) and Phase 4 (LLM-judge)
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

# Metrics that require dietary restrictions to be meaningful
DIET_METRICS = ["dietary_compliance"]
# Metrics only meaningful when shopping list is required
SHOPPING_METRICS = ["shopping_list"]


# ──────────────────────────────────────────────────────────────
# GROUND TRUTH LOADER
# ──────────────────────────────────────────────────────────────

def load_ground_truth(path: Path) -> dict[str, dict]:
    """Load ground truth JSON and return dict keyed by query_id."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    result: dict[str, dict] = {}
    for item in raw.get("items", []):
        qid = item["id"]
        constraints = item["expected_output"].get("constraints", {})
        result[qid] = {
            "id": qid,
            "test_category": item.get("test_category", "unknown"),
            "max_time_minutes": constraints.get("max_time_minutes"),
            "servings": constraints.get("servings"),
            "dietary_restrictions": constraints.get("dietary_restrictions", []),
            "requires_shopping_list": constraints.get("requires_shopping_list", False),
            "recipe_category": constraints.get("recipe_category", ""),
            "main_ingredient": constraints.get("main_ingredient", ""),
        }
    logger.info("Loaded %d ground truth items from %s", len(result), path.name)
    return result


# ──────────────────────────────────────────────────────────────
# LANGFUSE FETCHER
# ──────────────────────────────────────────────────────────────

def fetch_run_scores(dataset_name: str, run_name: str) -> list[dict]:
    """Fetch all item scores for a Langfuse dataset run.

    Returns a list of dicts, each with:
        item_id   : str  (e.g. "query_1")
        trace_id  : str
        <metric>  : float | None  (one key per score)

    Uses the Langfuse Python SDK v3 pattern seen in the EVAL-agents-main codebase.
    """
    from aieng.agent_evals.configs import Configs
    from langfuse import Langfuse

    configs = Configs()
    lf = Langfuse(
        public_key=configs.langfuse_public_key,
        secret_key=configs.langfuse_secret_key.get_secret_value(),
        host=configs.langfuse_host,
    )

    logger.info("Fetching dataset run '%s' / '%s'...", dataset_name, run_name)

    # ── Fetch run items ──
    try:
        run = lf.get_dataset_run(dataset_name=dataset_name, run_name=run_name)
        run_items = run.dataset_run_items
    except AttributeError:
        # Fallback: some SDK versions expose .items instead
        run = lf.get_dataset_run(dataset_name=dataset_name, run_name=run_name)
        run_items = getattr(run, "items", []) or []

    logger.info("Found %d run items.", len(run_items))

    # ── For each item, fetch trace scores ──
    rows: list[dict] = []
    for run_item in run_items:
        trace_id = getattr(run_item, "trace_id", None)
        item_id = _extract_item_id(run_item)

        scores: dict[str, Any] = {}
        if trace_id:
            try:
                trace = lf.api.trace.get(trace_id)
                for score in (getattr(trace, "scores", None) or []):
                    name = getattr(score, "name", None)
                    value = getattr(score, "value", None)
                    if name:
                        scores[name] = value
            except Exception as exc:
                logger.warning("Could not fetch trace %s: %s", trace_id, exc)

        row: dict = {"item_id": item_id, "trace_id": trace_id or ""}
        row.update(scores)
        rows.append(row)

    lf.flush()
    return rows


def _extract_item_id(run_item: Any) -> str:
    """Extract the item ID from a Langfuse run item.

    The item ID is stored in metadata.id by our upload script.
    """
    # Try metadata.id first (set by our langfuse_upload.py in phase2)
    meta = getattr(run_item, "metadata", None) or {}
    if isinstance(meta, dict) and meta.get("id"):
        return str(meta["id"])

    # Fallback: use source_id, dataset_item_id, or id field
    for attr in ("source_id", "dataset_item_id", "id"):
        val = getattr(run_item, attr, None)
        if val:
            return str(val)

    return "unknown"


# ──────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────

def build_dataframe(
    scores_rows: list[dict],
    ground_truth: dict[str, dict],
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Join scores with ground truth metadata into a DataFrame."""
    import pandas as pd

    rows = []
    for score_row in scores_rows:
        item_id = score_row.get("item_id", "unknown")
        gt = ground_truth.get(item_id, {})

        row: dict[str, Any] = {
            "item_id": item_id,
            "test_category": gt.get("test_category", "unknown"),
            "max_time_minutes": gt.get("max_time_minutes"),
            "servings": gt.get("servings"),
            "n_dietary_restrictions": len(gt.get("dietary_restrictions", [])),
            "requires_shopping_list": gt.get("requires_shopping_list", False),
            "recipe_category": gt.get("recipe_category", ""),
            "main_ingredient": gt.get("main_ingredient", ""),
            "trace_id": score_row.get("trace_id", ""),
        }

        for metric in ALL_METRICS:
            val = score_row.get(metric)
            row[metric] = float(val) if val is not None else None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("item_id").reset_index(drop=True)
    return df


def compute_category_summary(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    """Compute per-category pass rates for each metric."""
    import pandas as pd

    records = []
    categories = sorted(df["test_category"].unique())

    for cat in categories:
        sub = df[df["test_category"] == cat]
        row: dict[str, Any] = {"test_category": cat, "n_items": len(sub)}
        for metric in ALL_METRICS:
            valid = sub[metric].dropna()
            row[f"{metric}_pass_rate"] = round(valid.mean(), 3) if len(valid) > 0 else None
            row[f"{metric}_n_valid"] = len(valid)
        records.append(row)

    # Add overall row
    overall: dict[str, Any] = {"test_category": "OVERALL", "n_items": len(df)}
    for metric in ALL_METRICS:
        valid = df[metric].dropna()
        overall[f"{metric}_pass_rate"] = round(valid.mean(), 3) if len(valid) > 0 else None
        overall[f"{metric}_n_valid"] = len(valid)
    records.append(overall)

    return pd.DataFrame(records)


def find_failures(df: "pd.DataFrame", threshold: int = 2) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Return items that failed ≥ threshold LLM-judge metrics."""
    llm_metrics = [
        "dietary_compliance", "time_constraint",
        "recipe_present", "actionable_steps",
        "recipe_present_strict", "actionable_steps_strict",
        "recipe_present_lenient", "actionable_steps_lenient",
        "serving_size",
    ]
    df = df.copy()
    df["n_llm_failures"] = df[llm_metrics].apply(
        lambda row: sum(1 for m in llm_metrics if row.get(m) == 0.0), axis=1
    )
    failed = df[df["n_llm_failures"] >= threshold][
        ["item_id", "test_category", "n_llm_failures"] + llm_metrics
    ].sort_values("n_llm_failures", ascending=False)
    return failed


# ──────────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────────

def generate_markdown_report(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    summary: "pd.DataFrame",  # type: ignore[name-defined]
    failures: "pd.DataFrame",  # type: ignore[name-defined]
    dataset_name: str,
    run_name: str,
) -> str:
    """Generate a markdown analysis report from DataFrames."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_items = len(df)

    lines = [
        f"# Food Planner Agent — Evaluation Analysis Report",
        f"",
        f"**Dataset:** `{dataset_name}`  ",
        f"**Experiment Run:** `{run_name}`  ",
        f"**Generated:** {now}  ",
        f"**Total Items:** {n_items}",
        f"",
        f"---",
        f"",
        f"## 1. Overall Score Summary",
        f"",
    ]

    # Overall pass rates table
    overall = summary[summary["test_category"] == "OVERALL"].iloc[0]
    lines.append("| Metric | Type | Pass Rate | Valid Items |")
    lines.append("|---|---|---|---|")
    rule_metrics = ["required_tools", "output_present", "shopping_list"]
    llm_metrics = [
        "dietary_compliance", "time_constraint",
        "recipe_present", "actionable_steps",
        "recipe_present_strict", "actionable_steps_strict",
        "recipe_present_lenient", "actionable_steps_lenient",
        "serving_size",
    ]
    for m in rule_metrics:
        pr = overall.get(f"{m}_pass_rate")
        nv = int(overall.get(f"{m}_n_valid", 0))
        pr_str = f"{pr:.1%}" if pr is not None else "—"
        lines.append(f"| `{m}` | Rule-based | {pr_str} | {nv}/{n_items} |")
    for m in llm_metrics:
        pr = overall.get(f"{m}_pass_rate")
        nv = int(overall.get(f"{m}_n_valid", 0))
        pr_str = f"{pr:.1%}" if pr is not None else "—"
        lines.append(f"| `{m}` | LLM-judge | {pr_str} | {nv}/{n_items} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Pass Rates by Test Category",
        "",
    ]

    # Category breakdown table
    cat_cols = [c for c in summary.columns if c not in ("n_items",) and not c.endswith("_n_valid")]
    header_metrics = ALL_METRICS
    header = "| Category | N |" + "".join(f" {m[:8]} |" for m in header_metrics)
    sep = "|---|---|" + "".join("---|" for _ in header_metrics)
    lines.append(header)
    lines.append(sep)

    for _, row in summary[summary["test_category"] != "OVERALL"].iterrows():
        cat = row["test_category"]
        n = int(row["n_items"])
        vals = []
        for m in header_metrics:
            pr = row.get(f"{m}_pass_rate")
            vals.append(f"{pr:.0%}" if pr is not None else "—")
        lines.append(f"| {cat} | {n} |" + "".join(f" {v} |" for v in vals))

    # OVERALL row
    ov = summary[summary["test_category"] == "OVERALL"].iloc[0]
    vals = []
    for m in header_metrics:
        pr = ov.get(f"{m}_pass_rate")
        vals.append(f"**{pr:.0%}**" if pr is not None else "—")
    lines.append(f"| **OVERALL** | **{n_items}** |" + "".join(f" {v} |" for v in vals))

    lines += [
        "",
        "---",
        "",
        "## 3. Failure Analysis",
        "",
    ]

    if len(failures) == 0:
        lines.append("No items failed ≥ 2 LLM-judge metrics. Agent performed well across the board.")
    else:
        lines.append(f"**{len(failures)} items** failed ≥ 2 LLM-judge metrics:\n")
        lines.append("| Item | Category | # Failures | dietary | time | recipe | steps | recipe_s | steps_s | recipe_l | steps_l | serving |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in failures.iterrows():
            def fmt(v):
                if v is None: return "—"
                return "✓" if v == 1.0 else "✗"
            lines.append(
                f"| {row['item_id']} | {row['test_category']} | {int(row['n_llm_failures'])} |"
                f" {fmt(row.get('dietary_compliance'))} |"
                f" {fmt(row.get('time_constraint'))} |"
                f" {fmt(row.get('recipe_present'))} |"
                f" {fmt(row.get('actionable_steps'))} |"
                f" {fmt(row.get('recipe_present_strict'))} |"
                f" {fmt(row.get('actionable_steps_strict'))} |"
                f" {fmt(row.get('recipe_present_lenient'))} |"
                f" {fmt(row.get('actionable_steps_lenient'))} |"
                f" {fmt(row.get('serving_size'))} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 4. Key Observations",
        "",
    ]

    # Auto-generate observations
    observations = _generate_observations(df, summary)
    for obs in observations:
        lines.append(f"- {obs}")

    lines += [
        "",
        "---",
        "",
        "## 5. Items by Test Category",
        "",
    ]

    for cat in sorted(df["test_category"].unique()):
        sub = df[df["test_category"] == cat][["item_id", "main_ingredient", "max_time_minutes",
                                              "servings"] + ALL_METRICS]
        lines.append(f"### {cat} ({len(sub)} items)")
        lines.append("")
        lines.append("| Item | Ingredient | Time | Srv |" +
                     "".join(f" {m[:6]} |" for m in ALL_METRICS))
        lines.append("|---|---|---|---|" + "---|" * len(ALL_METRICS))
        for _, row in sub.iterrows():
            def s(v):
                if v is None: return "—"
                return "1" if v == 1.0 else "0"
            lines.append(
                f"| {row['item_id']} | {str(row['main_ingredient'])[:16]} |"
                f" {row['max_time_minutes']}m | {row['servings']} |"
                + "".join(f" {s(row.get(m))} |" for m in ALL_METRICS)
            )
        lines.append("")

    lines += [
        "---",
        "",
        "*Generated by EVAL team Phase 5 analysis script.*",
    ]

    return "\n".join(lines)


def _generate_observations(df, summary) -> list[str]:
    """Auto-generate key observations from the analysis data."""
    obs = []
    overall = summary[summary["test_category"] == "OVERALL"].iloc[0]

    # Lowest scoring metric
    metric_rates = {}
    for m in ALL_METRICS:
        pr = overall.get(f"{m}_pass_rate")
        if pr is not None:
            metric_rates[m] = pr
    if metric_rates:
        worst = min(metric_rates, key=metric_rates.get)
        best = max(metric_rates, key=metric_rates.get)
        obs.append(
            f"**Weakest metric**: `{worst}` ({metric_rates[worst]:.0%} pass rate). "
            f"**Strongest**: `{best}` ({metric_rates[best]:.0%})."
        )

    # Category with most failures
    cat_summary = summary[summary["test_category"] != "OVERALL"]
    if not cat_summary.empty:
        # Average pass rate across all LLM metrics per category
        llm_cols = [f"{m}_pass_rate" for m in
                    ["dietary_compliance", "time_constraint",
                     "recipe_present", "actionable_steps",
                     "recipe_present_strict", "actionable_steps_strict",
                     "recipe_present_lenient", "actionable_steps_lenient",
                     "serving_size"]]
        valid_cols = [c for c in llm_cols if c in cat_summary.columns]
        if valid_cols:
            cat_summary = cat_summary.copy()
            cat_summary["avg_llm"] = cat_summary[valid_cols].mean(axis=1)
            worst_cat = cat_summary.loc[cat_summary["avg_llm"].idxmin(), "test_category"]
            best_cat = cat_summary.loc[cat_summary["avg_llm"].idxmax(), "test_category"]
            obs.append(
                f"**Hardest test category**: `{worst_cat}` (lowest avg LLM-judge pass rate). "
                f"**Easiest**: `{best_cat}`."
            )

    # Time constraint check
    tight_time = df[df["test_category"] == "tight_time"]
    if len(tight_time) > 0:
        tc = tight_time["time_constraint"].dropna()
        if len(tc) > 0:
            rate = tc.mean()
            obs.append(
                f"**Tight-time queries** (≤25 min): `time_constraint` pass rate = {rate:.0%} "
                f"({int(tc.sum())}/{len(tc)} passed). "
                + ("Agent often exceeds tight limits." if rate < 0.6 else "Agent handles tight limits well.")
            )

    # Dietary compliance check
    diet_items = df[df["n_dietary_restrictions"] > 0]
    if len(diet_items) > 0:
        dc = diet_items["dietary_compliance"].dropna()
        if len(dc) > 0:
            rate = dc.mean()
            obs.append(
                f"**Dietary restriction queries**: `dietary_compliance` pass rate = {rate:.0%} "
                f"({int(dc.sum())}/{len(dc)} passed)."
            )

    # Required tools check
    rt = df["required_tools"].dropna()
    if len(rt) > 0 and rt.mean() < 1.0:
        missed = int((rt == 0).sum())
        obs.append(
            f"**Tool trajectory**: {missed} item(s) missed at least one required tool call. "
            "Review those items in Langfuse for skipped CFIA checks or missing modify_recipe calls."
        )

    return obs


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

@click.command()
@click.option("--dataset-name", default=DEFAULT_DATASET_NAME, show_default=True)
@click.option("--run-name", default=DEFAULT_RUN_NAME, show_default=True)
@click.option(
    "--ground-truth-path",
    default=str(GROUND_TRUTH_PATH),
    show_default=True,
    help="Path to food_planner_eval_ground_truth.json",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory for CSV and markdown output files.",
)
def cli(
    dataset_name: str,
    run_name: str,
    ground_truth_path: str,
    output_dir: str,
) -> None:
    """Fetch Phase 4 results from Langfuse and generate analysis report."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required: uv add pandas")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.error(
            "Ground truth not found at %s\n"
            "Copy EVALUATION/phase2/food_planner_eval_ground_truth.json here.", gt_path
        )
        sys.exit(1)

    print(f"\n=== Phase 5: Analysis ===")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Run      : {run_name}")
    print(f"  Output   : {out_dir}")
    print("")

    # ── 1. Load ground truth metadata ──
    ground_truth = load_ground_truth(gt_path)

    # ── 2. Fetch Langfuse scores ──
    try:
        scores_rows = fetch_run_scores(dataset_name, run_name)
    except Exception as exc:
        logger.error("Failed to fetch Langfuse data: %s", exc)
        sys.exit(1)

    if not scores_rows:
        logger.error("No run items found for '%s' / '%s'.", dataset_name, run_name)
        sys.exit(1)

    # ── 3. Build DataFrames ──
    df = build_dataframe(scores_rows, ground_truth)
    summary = compute_category_summary(df)
    failures = find_failures(df, threshold=2)

    # ── 4. Save CSV outputs ──
    detailed_path = out_dir / "results_detailed.csv"
    summary_path = out_dir / "results_summary.csv"
    df.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Saved: %s", detailed_path)
    logger.info("Saved: %s", summary_path)

    # ── 5. Generate and save markdown report ──
    report_md = generate_markdown_report(df, summary, failures, dataset_name, run_name)
    report_path = out_dir / "analysis_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved: %s", report_path)

    # ── 6. Print console summary ──
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    overall = summary[summary["test_category"] == "OVERALL"].iloc[0]
    rule_metrics = ["required_tools", "output_present", "shopping_list"]
    llm_metrics = [
        "dietary_compliance", "time_constraint",
        "recipe_present", "actionable_steps",
        "recipe_present_strict", "actionable_steps_strict",
        "recipe_present_lenient", "actionable_steps_lenient",
        "serving_size",
    ]

    print("\n  Rule-based metrics:")
    for m in rule_metrics:
        pr = overall.get(f"{m}_pass_rate")
        nv = int(overall.get(f"{m}_n_valid", 0))
        bar = _bar(pr)
        print(f"    {m:<22} {bar}  {pr:.0%} ({nv}/{len(df)})" if pr is not None else f"    {m:<22} —")

    print("\n  LLM-judge metrics:")
    for m in llm_metrics:
        pr = overall.get(f"{m}_pass_rate")
        nv = int(overall.get(f"{m}_n_valid", 0))
        bar = _bar(pr)
        print(f"    {m:<22} {bar}  {pr:.0%} ({nv}/{len(df)})" if pr is not None else f"    {m:<22} —")

    print(f"\n  Items with ≥2 LLM-judge failures: {len(failures)}")

    print("\n  Output files:")
    print(f"    {detailed_path}")
    print(f"    {summary_path}")
    print(f"    {report_path}")

    print("\n  -> Review analysis_report.md for full analysis.")
    print("  -> Green light: proceed to Phase 6 (final report + slides).")
    print("=" * 65 + "\n")


def _bar(rate: float | None, width: int = 10) -> str:
    """Simple ASCII progress bar."""
    if rate is None:
        return "[" + "?" * width + "]"
    filled = round(rate * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


if __name__ == "__main__":
    cli()
