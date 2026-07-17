from ddgs import DDGS


def research(query: str, max_results: int = 8) -> str:
    """Search the public web for current or external information.

    Use for recent news, current events, facts to verify, or external sources.

    Args:
        query: Search query.
        max_results: Max results (default 8, max 20).

    Returns: Result titles, snippets, and URLs.
    """
    max_results = min(max(max_results, 1), 20)

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Error: search failed: {e}"

    if not results:
        return f"No results found for: {query}"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        body = r.get("body", "No description available.")
        url = r.get("href", "")
        lines.append(f"[{i}] {title}\n    {body}\n    {url}")

    return "\n".join(lines)
