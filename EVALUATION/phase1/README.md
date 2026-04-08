# Phase 1 — Setup & Environment Verification

## Files

| File | Purpose | Run from |
|------|---------|----------|
| `test_connections.py` | Verify all API keys and services load correctly | Root of `EVAL-agents-main` repo |
| `smoke_test_food_planner.py` | Run Food Planner on 1 query, confirm trace in Langfuse | Root of food planner repo |

## How to Run in Coder

### test_connections.py
```bash
# From the EVAL-agents-main root directory
uv run --env-file .env python execution/phase1/test_connections.py
```

### smoke_test_food_planner.py
```bash
# From /home/coder/eval-agents/
uv run --env-file .env python execution/phase1/smoke_test_food_planner.py
```

## Expected Output

### test_connections.py
```
[CHECK] Loading Configs...          OK
[CHECK] Gemini API key present...   OK  (model: gemini-2.5-pro)
[CHECK] Langfuse public key...      OK  (pk-lf-...)
[CHECK] Langfuse secret key...      OK
[CHECK] Langfuse host...            OK  (https://us.cloud.langfuse.com)
[CHECK] Langfuse client connects... OK
[CHECK] Sending test trace...       OK  (trace_id: ...)

All checks passed. Ready to proceed to Phase 2.
```

### smoke_test_food_planner.py
```
[SMOKE] Running Food Planner on test query...
[SMOKE] Query: Suggest a vegetarian recipe under 30 minutes for 2 people.
[SMOKE] Agent finished. Final output received: YES
[SMOKE] Output preview: ...
[SMOKE] Langfuse trace flushed.

Check your Langfuse dashboard for a trace named 'smoke_test_food_planner'.
Green light: proceed to Phase 2.
```
