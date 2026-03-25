# Rubric: serving_size

You are evaluating whether a Food Planner recipe is scaled to serve the correct
number of people as stated in the user's query.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
recipe response. The "Expected Output" contains `constraints.servings`.

## Scoring Instructions

You must emit **exactly one metric**:

### 1. serving_size

Extract the requested serving count from the Input (e.g., "serve 6 people" → 6).
Then check if the Candidate Output mentions the correct serving size.

| Scenario | Value |
|---|---|
| Output explicitly states the correct serving count (e.g., "serves 6", "for 6 people") | **1** (pass) |
| Output does not mention serving count but ingredient quantities suggest the correct scale | **1** (pass — reasonable inference) |
| Output explicitly states a **different** serving count from what was requested | **0** (fail) |
| Output clearly uses quantities for a very different scale (e.g., recipe for 2 when 12 were requested, with no scaling note) | **0** (fail) |

**Notes:**
- "Makes 12 servings" for a 10-person request is acceptable (close enough).
- Only flag as fail if the mismatch is clear and unambiguous.
- If the agent mentioned scaling (e.g., "scaled from 4 to 10 servings"), score 1.

Use binary values only (0 or 1). Include a one-sentence comment citing the
serving size found in the output vs. the requested count.
