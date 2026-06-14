import json

def validate_schema(raw_data: str) -> tuple[bool, dict | None]:
    """
    Validates the incoming JSON string structure against the POC requirements.
    """
    try:
        data = json.loads(raw_data)
        
        # Example validation checklist for a compliance engine
        required_fields = ["transaction_id", "amount", "user_id"]
        for field in required_fields:
            if field not in data:
                print(f"Schema Error: Missing field '{field}'")
                return False, None
                
        return True, data
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Schema Error: Invalid JSON formatting. {e}")
        return False, None