import re

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"<\|.*?\|>",
    r"DROP TABLE",
    r"SELECT \*",
    r"--",
]


def sanitize_string(value: str) -> str:
    """Strip prompt injection and SQL injection attempts from MCP tool inputs."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f"Potentially unsafe input detected: '{pattern}'")
    return value.strip()


def sanitize_params(params: dict) -> dict:
    return {
        k: sanitize_string(v) if isinstance(v, str) else v
        for k, v in params.items()
    }
