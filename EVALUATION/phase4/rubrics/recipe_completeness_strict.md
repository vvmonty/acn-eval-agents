# Rubric: recipe_completeness_strict

You are evaluating whether a Food Planner agent response contains a highly complete,
professional-grade recipe that a home cook can execute flawlessly without any guesswork.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
response text.

## Scoring Instructions

You must emit **exactly two metrics**:

### 1. recipe_present_strict

Does the output contain a fully concrete recipe meeting all structural professional standards?

| Scenario | Value |
|---|---|
| Output includes named dish, exact yield/servings, exhaustive ingredient list with precise measurements, and detailed, ordered cooking steps | **1** (pass) |
| Output misses any of the following: yield, precise ingredient measurements, or step-by-step instructions | **0** (fail) |
| Output asks clarifying questions or provides only high-level guidance | **0** (fail) |

### 2. actionable_steps_strict

Are the instructions meticulously clear, lacking any ambiguity?

| Scenario | Value |
|---|---|
| EVERY relevant step includes precise quantities, exact temperatures, specific timing cues, AND physical indicators of doneness | **1** (pass) |
| Steps are missing exact times/temperatures where applicable, or leave any guesswork to the cook (e.g., "cook until done") | **0** (fail) |

**Strictness:** A recipe MUST be perfect. No omissions are acceptable. Score 0 if any structural element or exact measurement is missing.

Use binary values only (0 or 1). Include a one-sentence comment per metric.