import os
import re
import smtplib
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


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via SMTP.

    Use only when the user explicitly asks to send an email (not draft).
    Resolve ambiguous recipients with list_contacts first.

    Args:
        to: Recipient name (from contacts.md) or email address.
        subject: Subject line.
        body: Body text.

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

    try:
        cfg = _get_smtp_config()
    except EnvironmentError as e:
        return str(e)

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = cfg["address"]
    msg["To"] = recipient_email
    msg["Subject"] = subject

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.starttls()
            server.login(cfg["address"], cfg["password"])
            server.send_message(msg)
        return f"Email sent successfully to {recipient_email}."
    except Exception as e:
        return f"Failed to send email: {e}"


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
