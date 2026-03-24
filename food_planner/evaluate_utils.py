from functools import wraps
from src.utils.langfuse.shared_client import langfuse_client


# --- Decorator for auto-tracing tools ---
def traced_tool(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with langfuse_client.start_as_current_observation(
                name=name,
                input={"args": args, "kwargs": kwargs}
            ) as obs:
                try:
                    result = func(*args, **kwargs)
                    obs.output = result
                    return result
                except Exception as e:
                    obs.error = str(e)
                    raise
        return wrapper
    return decorator
