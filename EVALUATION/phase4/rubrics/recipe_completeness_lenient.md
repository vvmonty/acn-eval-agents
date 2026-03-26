# Rubric: recipe_completeness_lenient

You are casually evaluating whether a Food Planner agent response contains a general idea or suggestion of a recipe that a home cook can roughly follow.

The "Input" is the user's recipe query. The "Candidate Output" is the agent's
response text.

## Scoring Instructions

You must emit **exactly two metrics**:

### 1. recipe_present_lenient

Does the output contain any semblance of a concrete recipe or dish suggestion?

| Scenario | Value |
|---|---|
| Output mentions at least a dish idea and some basic ingredients or steps | **1** (pass) |
| Output is only generic chat, completely unrelated, or just asks clarifying questions with zero recipe suggestions | **0** (fail) |

### 2. actionable_steps_lenient

Are there any instructions or general guidance that could vaguely be helpful to a cook?

| Scenario | Value |
|---|---|
| Steps give a general idea of how to prepare the dish (even if broad, like "toss the salad") | **1** (pass) |
| There are absolutely no instructions provided whatsoever | **0** (fail) |

**Leniency:** A recipe can be very loose. Extremely minor omissions are completely fine. A bare-bones list of steps or ingredients is enough to pass.

Use binary values only (0 or 1). Include a one-sentence comment per metric.