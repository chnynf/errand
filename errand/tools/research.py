from ddgs import DDGS


def research(query: str, max_results: int = 8) -> str:
    """
    Search the public web for current or external information.

    Use when the user asks about:
    - recent news, current events, or things that may have changed
    - facts you are uncertain about and need to verify
    - external sources for a topic (papers, vendor docs, blog posts)

    Do NOT use for:
    - reading configured local files or the knowledge base (use read_file instead)
    - local project files
    - simple chit-chat that doesn't need fresh information

    Args:
        query: Search query (e.g., "latest advances in quantum computing").
        max_results: Maximum number of results (default 8, max 20).

    Returns: Formatted list of result titles, snippets, and source URLs.

    Example: research("Python 3.13 new features") returns a summary of top results
    """
    max_results = min(max(max_results, 1), 20)

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return f"No results found for: {query}"

    lines = [f"Research results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        body = r.get("body", "No description available.")
        url = r.get("href", "")
        lines.append(f"[{i}] {title}\n    {body}\n    Source: {url}\n")

    lines.append(
        "---\n"
        f"Found {len(results)} results. "
        "Use the sources above to form a comprehensive answer."
    )
    return "\n".join(lines)
