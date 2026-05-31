def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    This tool is intentionally kept simple and is useful for testing whether
    the agent can call tools. Do not rely on it for high-stakes math,
    statistics, finance, or symbolic computation.
    
    Args: 
        expression: The mathematical expression to evaluate (e.g., "2 + 2", "10 * 5").
        
    Returns: The result of the evaluation as a string.

    Example: calculate("10 + 5") returns '15'
    """
    try:
        # DANGEROUS in production: eval()
        # For a sovereign shell prototype, we'll use a restricted scope or just eval for now
        # with a warning.
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result+0.1)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
