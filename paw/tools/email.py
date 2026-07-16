import mimetypes
import os
import re
import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText
from pathlib import Path


_CONTACTS_FILE = Path(__file__).resolve().parent.parent / "contacts.md"

_SMTP_DEFAULTS = {
    "host": "smtp.gmail.com",
    "port": 587,
}


def _load_contacts() -> dict[str, str]:
    """Parse contacts.md and return a {name_lower: email} mapping."""
    if not _CONTACTS_FILE.exists():
        return {}

    contacts = {}
    text = _CONTACTS_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        name, _, email = line[2:].partition(":")
        email = email.strip()
        if "@" in email:
            contacts[name.strip().lower()] = email
    return contacts


def _get_smtp_config() -> dict:
    address = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")
    if not address or not password:
        raise EnvironmentError(
            "EMAIL_ADDRESS and EMAIL_PASSWORD must be set in environment variables."
        )
    return {
        "address": address,
        "password": password,
        "host": os.getenv("EMAIL_SMTP_HOST", _SMTP_DEFAULTS["host"]),
        "port": int(os.getenv("EMAIL_SMTP_PORT", _SMTP_DEFAULTS["port"])),
    }


def _resolve_attachments(attachments: str) -> tuple[list[Path], list[str]]:
    """Split a comma/newline-separated path string into (existing, missing).

    Returns the resolved paths that exist as files and a list of the raw
    entries that could not be found.
    """
    entries = [p.strip() for p in re.split(r"[,\n]", attachments) if p.strip()]
    resolved: list[Path] = []
    missing: list[str] = []
    for entry in entries:
        path = Path(entry).expanduser()
        if path.is_file():
            resolved.append(path)
        else:
            missing.append(entry)
    return resolved, missing


def send_email(to: str, subject: str, body: str, attachments: str = "") -> str:
    """Send an email via SMTP, optionally with file attachments.

    Use when the user explicitly asks to send an email (not draft).
    Resolve ambiguous recipients with list_contacts first.

    Args:
        to: Recipient name (from contacts.md) or email address.
        subject: Subject line.
        body: Body text. Always open with a brief identification line, e.g.
            "Hi, I'm Yunfei's AI assistant." — unless the user explicitly asks
            you not to identify yourself.
        attachments: Optional file path, or several paths separated by commas
            or newlines. Each must be an existing file. Leave empty for none.

    Returns: Success or error message.
    """
    if "@" in to:
        recipient_email = to
    else:
        contacts = _load_contacts()
        recipient_email = contacts.get(to.lower())
        if not recipient_email:
            available = ", ".join(
                name.title() for name in sorted(contacts.keys())
            )
            return (
                f"Contact '{to}' not found. "
                f"Available contacts: {available or 'none'}"
            )

    files, missing = _resolve_attachments(attachments)
    if missing:
        return f"Attachment(s) not found: {', '.join(missing)}"

    try:
        cfg = _get_smtp_config()
    except EnvironmentError as e:
        return str(e)

    if files:
        msg = EmailMessage()
        msg.set_content(body)
        for path in files:
            ctype, _ = mimetypes.guess_type(path.name)
            maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
            msg.add_attachment(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype or "octet-stream",
                filename=path.name,
            )
    else:
        msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = cfg["address"]
    msg["To"] = recipient_email
    msg["Subject"] = subject

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.starttls()
            server.login(cfg["address"], cfg["password"])
            server.send_message(msg)
    except Exception as e:
        return f"Failed to send email: {e}"

    result = f"Email sent to {recipient_email}. Subject: {subject!r}. Body: {len(body)} chars."
    if files:
        result += f" Attachments: {', '.join(p.name for p in files)}."
    return result


def list_contacts() -> str:
    """List saved email contacts.

    Use when resolving a recipient for email or when the user asks for available contacts.

    Returns: Contact names and email addresses.
    """
    contacts = _load_contacts()
    if not contacts:
        return "No contacts found in contacts.md."

    lines = [f"Contacts ({len(contacts)}):"]
    for name, email in sorted(contacts.items()):
        lines.append(f"  - {name.title()}: {email}")
    return "\n".join(lines)


def add_contact(name: str, email: str) -> str:
    """Add a new saved email contact.

    Use only when the user explicitly asks to save a contact.

    Args:
        name: Display name.
        email: Email address.

    Returns: Confirmation or error.
    """
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return f"Invalid email address: {email}"

    contacts = _load_contacts()
    if name.lower() in contacts:
        return f"Contact '{name}' already exists with email {contacts[name.lower()]}."

    with open(_CONTACTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {name}: {email}\n")

    return f"Contact '{name}' ({email}) added successfully."
