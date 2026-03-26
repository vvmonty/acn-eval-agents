# Rubric: recipe_completeness

You are evaluating whether a Food Planner agent response contains a complete,
usable recipe that a home cook can actually follow, and **what kind of response**
it is when there is no full recipe.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
response text.

## Scoring Instructions

You must emit **exactly three metrics**:

### 1. recipe_completeness_category

Classify the output using **exactly one** of the following strings (copy the
value character-for-character):

| Category | When to use |
|---|---|
| `complete_recipe` | Named dish, ingredient list, and **at least 3 distinct cooking steps** are all present. |
| `incomplete_recipe` | Recipe-like content but a **key structural piece is missing** (e.g. fewer than 3 steps, no ingredient list, dish named but no workable procedure). |
| `deferral_no_recipe` | Explains that no suitable recipe was found, mentions a "closest match," promises a **web search or next step**, or similar—**without** supplying a full recipe in this message. |
| `constraint_conflict_clarification` | User request is **logically contradictory or impossible as stated**; agent explains the conflict and asks the user to **choose how to proceed** (e.g. vegetarian + chicken). |
| `clarifying_questions` | Asks for **missing** preferences or details **without** framing an explicit logical contradiction between stated constraints. |
| `non_recipe` | Generic advice, off-topic prose, or anything that does not fit the categories above. |

**Disambiguation:** If the agent both names another dish and gives a full recipe for *that* dish in the same message, use `complete_recipe` or `incomplete_recipe` based on **that** recipe’s structure—not `deferral_no_recipe`. Use `deferral_no_recipe` only when there is **no** such complete recipe body.

For metric `recipe_completeness_category`, set `value` to the string above (not 0/1). Include a one-sentence comment.

### 2. recipe_present

Does the output contain an actual concrete recipe (not just generic advice)?

| Scenario | Value |
|---|---|
| Category is `complete_recipe` | **1** (pass) |
| Any other category | **0** (fail) |

### 3. actionable_steps

Are the instructions clear and specific enough to follow without additional research?

| Scenario | Value |
|---|---|
| Category is `complete_recipe` **and** steps include concrete quantities, temperatures, or timing cues where needed | **1** (pass) |
| Category is `incomplete_recipe` **and** the steps that exist are concrete enough to follow | **1** (pass) |
| Steps are missing or so vague a cook could not follow them | **0** (fail) |
| Category is `deferral_no_recipe`, `constraint_conflict_clarification`, `clarifying_questions`, or `non_recipe` | **0** (fail) |

**Leniency:** A recipe does not need to be perfect — minor omissions are acceptable.
Only score `actionable_steps` 0 when steps are clearly unusable or absent.

For `recipe_present` and `actionable_steps`, use **binary values only** (0 or 1).
Include a one-sentence comment per metric.

## Worked examples

- **Deferral:** The agent says it could not find "salmon tacos" meeting stovetop and
  no-mushroom constraints in time, names "Air Fryer Lobster Tails" as closest match,
  and says it will search the web—**no** ingredients or numbered steps in the message.
  → `recipe_completeness_category` = `deferral_no_recipe`; `recipe_present` = **0**;
  `actionable_steps` = **0**.

- **Conflict clarification:** The user asks for a grill recipe with chicken and rice but
  also specifies vegetarian. The agent apologizes, explains chicken is not vegetarian,
  and asks whether to do a vegetarian grill recipe **or** chicken with rice.
  → `recipe_completeness_category` = `constraint_conflict_clarification`;
  `recipe_present` = **0**; `actionable_steps` = **0**.
