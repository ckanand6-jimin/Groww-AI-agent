"""Report rendering — Phase 5.

Transforms a PulseReport into Google Docs batch-update requests and
Gmail email HTML/text bodies. Pure functions with no MCP dependency.

Previews can be written to local files for stakeholder review before
Workspace integration in Phase 6.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

from pulse.models.models import PulseReport
from pulse.render.docs import (
    build_batch_update_requests,
    build_doc_content,
    build_heading_text,
)
from pulse.render.email import (
    build_email_body_html,
    build_email_body_text,
    build_email_payload,
    build_subject,
)


def render(
    report: PulseReport,
    *,
    doc_url: str = "",
    recipients: list[str] | None = None,
) -> dict:
    """Render a PulseReport into Doc and Email payloads.

    Args:
        report: The complete PulseReport to render.
        doc_url: Google Doc URL for email deep link (optional).
        recipients: Email recipient list (optional).

    Returns:
        A dict with:
            heading_text: Deterministic H2 heading for idempotency.
            doc_batch_update: Google Docs batchUpdate request body.
            doc_content: Structured content dict for preview/snapshot.
            email_payload: Complete email payload (subject, html, text, recipients).
    """
    return {
        "heading_text": build_heading_text(report),
        "doc_batch_update": build_batch_update_requests(report),
        "doc_content": build_doc_content(report),
        "email_payload": build_email_payload(
            report, recipients=recipients, doc_url=doc_url,
        ),
    }


def preview(
    report: PulseReport,
    output_dir: str,
    *,
    doc_url: str = "",
    recipients: list[str] | None = None,
) -> Tuple[str, str]:
    """Write preview files to disk for stakeholder review.

    Writes two files:
        preview.doc.json  — Google Docs batch-update payload + structured content
        preview.email.html — HTML email preview

    Args:
        report: The complete PulseReport to render.
        output_dir: Directory to write preview files to.
        doc_url: Google Doc URL for email deep link (optional).
        recipients: Email recipient list (optional).

    Returns:
        Tuple of (doc_preview_path, email_preview_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    rendered = render(report, doc_url=doc_url, recipients=recipients)

    # --- Doc preview (JSON) ---
    doc_preview = {
        "heading_text": rendered["heading_text"],
        "doc_content": rendered["doc_content"],
        "batch_update_requests": rendered["doc_batch_update"],
    }
    doc_path = os.path.join(output_dir, "preview.doc.json")
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(doc_preview, f, indent=2, ensure_ascii=False)

    # --- Email preview (HTML) ---
    email_html = build_email_body_html(report, doc_url=doc_url)
    email_path = os.path.join(output_dir, "preview.email.html")
    with open(email_path, "w", encoding="utf-8") as f:
        f.write(email_html)

    return doc_path, email_path


__all__ = [
    "build_batch_update_requests",
    "build_doc_content",
    "build_email_body_html",
    "build_email_body_text",
    "build_email_payload",
    "build_heading_text",
    "build_subject",
    "preview",
    "render",
]
