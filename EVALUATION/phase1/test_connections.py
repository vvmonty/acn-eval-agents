"""Phase 1 — Connection & Environment Verification Script.

PURPOSE:
    Verifies that all required API keys and services are correctly configured
    in the Coder workspace .env file before any evaluation work begins.
    Sends one test trace to Langfuse to confirm write access.

PASTE TO:
    <eval-agents-main-root>/execution/phase1/test_connections.py

RUN WITH:
    uv run --env-file .env python execution/phase1/test_connections.py

EXPECTED RESULT:
    All checks print OK. A test trace appears in the Langfuse dashboard.
    If any check fails, fix the .env entry before proceeding to Phase 2.
"""

import sys


# ──────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────

def ok(label: str, detail: str = "") -> None:
    """Print a green OK line."""
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [OK]   {label}{detail_str}")


def fail(label: str, error: str) -> None:
    """Print a red FAIL line and exit."""
    print(f"  [FAIL] {label}")
    print(f"         Error: {error}")
    sys.exit(1)


# ──────────────────────────────────────────────
# CHECK 1 — Load Configs
# ──────────────────────────────────────────────

print("\n=== Phase 1: Connection & Environment Checks ===\n")

print("[1/6] Loading Configs from environment...")
try:
    from aieng.agent_evals.configs import Configs
    configs = Configs()
    ok("Configs loaded")
except Exception as e:
    fail("Configs failed to load", str(e))


# ──────────────────────────────────────────────
# CHECK 2 — Gemini / OpenAI API Key
# ──────────────────────────────────────────────

print("[2/6] Checking Gemini API key...")
try:
    key_val = configs.openai_api_key.get_secret_value()
    if not key_val or key_val in ("...", ""):
        fail("Gemini API key", "Key is empty or placeholder '...'")
    ok(
        "Gemini API key present",
        f"planner={configs.default_planner_model}  "
        f"worker={configs.default_worker_model}  "
        f"evaluator={configs.default_evaluator_model}",
    )
except Exception as e:
    fail("Gemini API key check", str(e))


# ──────────────────────────────────────────────
# CHECK 3 — Langfuse Keys
# ──────────────────────────────────────────────

print("[3/6] Checking Langfuse keys...")
try:
    pub = configs.langfuse_public_key
    sec = configs.langfuse_secret_key

    if not pub:
        fail("Langfuse public key", "Not set — add LANGFUSE_PUBLIC_KEY to .env")
    if not pub.startswith("pk-lf-"):
        fail("Langfuse public key format", f"Must start with 'pk-lf-', got: {pub[:10]}...")

    if not sec:
        fail("Langfuse secret key", "Not set — add LANGFUSE_SECRET_KEY to .env")
    sec_val = sec.get_secret_value()
    if not sec_val.startswith("sk-lf-"):
        fail("Langfuse secret key format", "Must start with 'sk-lf-'")

    ok("Langfuse public key", pub[:20] + "...")
    ok("Langfuse secret key", "sk-lf-****")
    ok("Langfuse host", configs.langfuse_host)
except Exception as e:
    fail("Langfuse key check", str(e))


# ──────────────────────────────────────────────
# CHECK 4 — Langfuse Client Instantiation
# ──────────────────────────────────────────────

print("[4/6] Instantiating Langfuse client...")
try:
    from langfuse import Langfuse

    lf = Langfuse(
        public_key=configs.langfuse_public_key,
        secret_key=configs.langfuse_secret_key.get_secret_value(),
        host=configs.langfuse_host,
    )
    ok("Langfuse client instantiated")
except Exception as e:
    fail("Langfuse client instantiation", str(e))


# ──────────────────────────────────────────────
# CHECK 5 — Send a Test Trace to Langfuse
# Uses Langfuse v3 API: start_as_current_observation()
# (lf.trace() was removed in v3)
# ──────────────────────────────────────────────

print("[5/6] Sending test trace to Langfuse...")
try:
    with lf.start_as_current_observation(
        name="phase1_connection_test",
        as_type="span",
        input={"message": "Phase 1 connection test from EVAL team"},
    ) as obs:
        obs.update(
            output={"status": "ok"},
            metadata={"phase": "1", "check": "connection_test"},
        )
        trace_id = lf.get_current_trace_id()

    lf.flush()
    ok("Test trace sent", f"trace_id={trace_id}")
except Exception as e:
    fail("Sending test trace to Langfuse", str(e))


# ──────────────────────────────────────────────
# CHECK 6 — Google API Key for ADK
# ──────────────────────────────────────────────

print("[6/6] Checking Google API key (for ADK / google-genai)...")
try:
    google_key = configs.google_api_key.get_secret_value()
    if not google_key or google_key in ("...", ""):
        fail("Google API key", "Not set — add GOOGLE_API_KEY or GEMINI_API_KEY to .env")
    ok("Google API key present", "used by google-adk")
except Exception as e:
    fail("Google API key check", str(e))


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("  All 6 checks passed.")
print("  -> Open Langfuse dashboard and confirm the")
print("     'phase1_connection_test' trace is visible.")
print("  -> Green light: proceed to Phase 2.")
print("=" * 50 + "\n")
