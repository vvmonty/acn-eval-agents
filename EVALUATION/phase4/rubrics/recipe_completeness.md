# Rubric: recipe_completeness

You are evaluating whether a Food Planner agent response contains a complete,
usable recipe that a home cook can actually follow.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
response text.

## Scoring Instructions

You must emit **exactly two metrics**:

### 1. recipe_present

Does the output contain an actual concrete recipe (not just generic advice)?

| Scenario | Value |
|---|---|
| Output includes named dish, ingredient list, and at least 3 cooking steps | **1** (pass) |
| Output is only generic advice, a discussion, or a single-sentence suggestion with no recipe | **0** (fail) |
| Output asks clarifying questions instead of providing a recipe | **0** (fail) |

### 2. actionable_steps

Are the instructions clear and specific enough to follow without additional research?

| Scenario | Value |
|---|---|
| Steps include concrete quantities, temperatures, or timing cues | **1** (pass) |
| Steps are so vague that a cook could not follow them (e.g., "cook the chicken" with no method or time) | **0** (fail) |

**Leniency:** A recipe does not need to be perfect — minor omissions are acceptable.
Only score 0 if a key structural element is clearly missing.

Use binary values only (0 or 1). Include a one-sentence comment per metric.
