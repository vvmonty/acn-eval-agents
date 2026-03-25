# Rubric: time_constraint

You are evaluating whether a Food Planner recipe respects the total time limit
stated in the user's query.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
recipe response. The "Expected Output" contains `constraints.max_time_minutes`.

## Scoring Instructions

You must emit **exactly one metric**:

### 1. time_constraint

Extract the time limit from the Input (e.g., "under 30 minutes" → 30 min).
Then examine the Candidate Output for any stated total time, prep time, or
cook time.

| Scenario | Value |
|---|---|
| Candidate Output explicitly states a total time ≤ the limit | **1** (pass) |
| Candidate Output does not mention time, but described steps are clearly feasible within the limit | **1** (pass) |
| Candidate Output states a total time that **exceeds** the limit | **0** (fail) |
| Candidate Output describes steps that clearly cannot be completed within the limit (e.g., "marinate overnight" for a 30-minute query) | **0** (fail) |

**Notes:**
- "Prep: 10 min, Cook: 20 min → Total: 30 min" passes a 30-minute limit.
- "Bake for 45 minutes" fails a 30-minute limit.
- If both prep and cook times are given and their sum exceeds the limit, score 0.
- Give benefit of the doubt on ambiguous phrasing unless there is clear evidence of violation.

Use binary values only (0 or 1). Include a one-sentence comment citing the
specific time values found, or explaining the failure reason.
