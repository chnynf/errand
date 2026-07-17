import os
import requests

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _get_headers():
    token = os.getenv("NOTION_TOKEN")
    if not token:
        raise EnvironmentError("NOTION_TOKEN not found in environment variables.")
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _find_database_id(headers: dict, name: str) -> str:
    resp = requests.post(
        f"{NOTION_API_BASE}/search",
        headers=headers,
        json={
            "query": name,
            "filter": {"property": "object", "value": "database"},
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Notion search failed ({resp.status_code}): {resp.text}")

    for result in resp.json().get("results", []):
        title_parts = result.get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_parts).strip()
        if title.lower() == name.lower():
            return result["id"]

    raise RuntimeError(f"No Notion database found with name '{name}'.")


def _extract_title(properties: dict) -> str:
    for prop_val in properties.values():
        if prop_val.get("type") == "title":
            parts = prop_val.get("title", [])
            if parts:
                return "".join(t.get("plain_text", "") for t in parts)
    return "Untitled"


def _extract_status(properties: dict, prop_name: str = "Status") -> str:
    status_prop = properties.get(prop_name, {})
    if status_prop.get("type") == "status" and status_prop.get("status"):
        return status_prop["status"].get("name", "Unknown")
    return "Unknown"


def _extract_page_content(headers: dict, page_id: str) -> str:
    try:
        resp = requests.get(
            f"{NOTION_API_BASE}/blocks/{page_id}/children",
            headers=headers,
            params={"page_size": 100}
        )
        if resp.status_code != 200:
            return ""
        
        blocks = resp.json().get("results", [])
        content_lines = []
        for block in blocks:
            block_type = block.get("type")
            if not block_type:
                continue
                
            block_data = block.get(block_type, {})
            rich_text = block_data.get("rich_text", [])
            
            text_content = "".join(t.get("plain_text", "") for t in rich_text)
            if text_content.strip():
                content_lines.append(text_content.strip())
                
        return " | ".join(content_lines)
    except Exception:
        return ""


def get_notion_tasks(status_filter: str = "To Do,Doing") -> str:
    """Fetch tasks from the Notion "Task List" database.

    Use when the user asks about their Notion tasks, todo list, or items by status.

    Args:
        status_filter: Comma-separated status values (e.g. "To Do,Doing").

    Returns: Task names and context grouped by status.
    """
    try:
        headers = _get_headers()
    except EnvironmentError as e:
        return f"Error: {e}"

    try:
        database_id = _find_database_id(headers, "Task List")
    except RuntimeError as e:
        return f"Error: {e}"

    statuses = [s.strip() for s in status_filter.split(",")]

    filter_conditions = [
        {"property": "Status", "status": {"equals": s}} for s in statuses
    ]
    query_filter = (
        {"or": filter_conditions}
        if len(filter_conditions) > 1
        else filter_conditions[0]
    )

    try:
        resp = requests.post(
            f"{NOTION_API_BASE}/databases/{database_id}/query",
            headers=headers,
            json={"filter": query_filter, "page_size": 100},
        )
        if resp.status_code != 200:
            return f"Error querying Notion database ({resp.status_code}): {resp.text}"

        pages = resp.json().get("results", [])
    except Exception as e:
        return f"Error querying Notion: {e}"

    tasks_by_status = {}
    for page in pages:
        props = page.get("properties", {})
        title = _extract_title(props)
        status = _extract_status(props)
        context = _extract_page_content(headers, page.get("id"))
        
        task_info = {"title": title, "context": context}
        tasks_by_status.setdefault(status, []).append(task_info)

    if not tasks_by_status:
        return "No tasks found with the specified statuses."

    lines = []
    for status in statuses:
        tasks = tasks_by_status.get(status, [])
        if tasks:
            lines.append(f"[{status}] ({len(tasks)} tasks)")
            for task in tasks:
                lines.append(f"  - {task['title']}")
                if task['context']:
                    lines.append(f"    Context: {task['context']}")
            lines.append("")

    return "\n".join(lines).strip() if lines else "No tasks found."
