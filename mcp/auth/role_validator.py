from functools import wraps
from typing import Callable

ROLE_HIERARCHY = {
    "SYSTEM_INTEGRATION": 10,
    "COMPLIANCE_ANALYST": 20,
    "COMPLIANCE_MANAGER": 30,
    "MODEL_RISK": 25,
    "INTERNAL_AUDIT": 15,
}


def require_role(allowed_roles: list[str]) -> Callable:
    """Decorator — raises PermissionError if caller role not in allowed_roles."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, caller_role: str = "COMPLIANCE_ANALYST", **kwargs):
            if caller_role not in allowed_roles:
                raise PermissionError(
                    f"Role '{caller_role}' not permitted for tool '{fn.__name__}'. "
                    f"Required: {allowed_roles}"
                )
            return fn(*args, caller_role=caller_role, **kwargs)
        return wrapper
    return decorator
