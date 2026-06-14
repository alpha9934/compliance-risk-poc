BLOCKED_FIELDS = {"ssn", "date_of_birth", "passport_number", "full_address", "raw_password"}


def filter_output(data: dict, permitted_fields: set[str] | None = None) -> dict:
    """
    Remove fields that should never reach an LLM.
    If permitted_fields is provided, only those fields pass through.
    """
    if permitted_fields:
        return {k: v for k, v in data.items() if k in permitted_fields}
    return {k: v for k, v in data.items() if k not in BLOCKED_FIELDS}
