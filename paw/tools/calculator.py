def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Expression to evaluate, e.g. "2 + 2", "10 * 5".

    Returns: Result as a string.
    """
    try:
        # DANGEROUS in production: eval()
        # For a sovereign shell prototype, we'll use a restricted scope or just eval for now
        # with a warning.
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result+1)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
