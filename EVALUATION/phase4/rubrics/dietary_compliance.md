# Rubric: dietary_compliance

You are evaluating whether a Food Planner recipe output respects ALL dietary
restrictions explicitly stated in the user's query.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
response, which may include a recipe text and/or a list of tools called.
The "Expected Output" contains a `constraints.dietary_restrictions` list.

## Scoring Instructions

You must emit **exactly one metric**:

### 1. dietary_compliance

Examine every ingredient, cooking method, and preparation note in the recipe text.
Cross-reference them against the dietary restrictions listed in the Input.

| Scenario | Value |
|---|---|
| No dietary restrictions stated in Input | **1** (not applicable) |
| Recipe contains NO ingredient or method that violates a stated restriction | **1** (pass) |
| Recipe contains at least one clear violation of a stated restriction | **0** (fail) |

**Common restriction mappings to check:**
- `vegan` → no meat, poultry, fish, dairy, eggs, honey
- `vegetarian` → no meat, poultry, fish; dairy and eggs are allowed
- `gluten-free` → no wheat flour, regular pasta, bread crumbs, soy sauce (unless tamari), barley, rye
- `dairy-free` → no butter, milk, cream, cheese, yogurt, sour cream
- `halal` → no pork, pork-derived ingredients; alcohol; non-halal-certified meat
- `kosher` → no pork, shellfish; no mixing of meat and dairy
- `low-carb` / `keto-friendly` → no pasta, bread, rice (unless cauliflower rice), sugar, potatoes
- `low-sodium` → no added table salt in large amounts; no high-sodium condiments without low-sodium alternative
- `high-fiber` → recipe should emphasize fiber-rich ingredients (beans, whole grains, vegetables)
- `no [ingredient]` → that ingredient must not appear in the recipe

**Important:** Only flag a violation if it is **clear and unambiguous**. Give benefit
of the doubt when substitutions are mentioned (e.g., "use dairy-free cheese").

Use binary values only (0 or 1). Include a one-sentence comment citing the specific
violation found, or confirming compliance.
