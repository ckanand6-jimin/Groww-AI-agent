"""Gmail email teaser builder — Phase 5.

Transforms a PulseReport into email subject + HTML/text body suitable for
Gmail MCP's create_draft or send_email tools.

Architecture reference: architecture.md §5.3
"""

from __future__ import annotations

from pulse.models.models import PulseReport


def build_subject(report: PulseReport) -> str:
    """Build the email subject line.

    Format: Groww Weekly Review Pulse — Week {ISO_WEEK}

    Example: Groww Weekly Review Pulse — Week 2026-W23
    """
    product_display = report.product.title()
    return f"{product_display} Weekly Review Pulse — Week {report.iso_week}"


def _build_theme_teasers(report: PulseReport, max_teasers: int = 5) -> list[str]:
    """Build short theme teaser bullets for the email body."""
    teasers: list[str] = []
    for theme in report.themes[:max_teasers]:
        teasers.append(f"{theme.name}: {theme.summary}")
    return teasers


def _build_doc_link(doc_url: str, heading_text: str) -> str:
    """Build a deep link to the Doc section.

    If a heading anchor/bookmark ID is known, append it as a fragment.
    Otherwise, just link to the document.
    """
    if not doc_url:
        return "[Document link not yet configured]"
    # For now, link to the doc. Deep linking with heading anchors
    # will be added in Phase 6 when we integrate with Docs MCP.
    return doc_url


def build_email_body_text(
    report: PulseReport,
    doc_url: str = "",
) -> str:
    """Build the plain-text email body.

    Args:
        report: The PulseReport to render.
        doc_url: URL to the Google Doc (optional; placeholder if empty).
    """
    teasers = _build_theme_teasers(report)
    doc_link = _build_doc_link(doc_url, "")

    lines: list[str] = []
    lines.append(f"{report.product.title()} — Week {report.iso_week}")
    lines.append(f"Period: {report.period.start_date} to {report.period.end_date}")
    lines.append(f"Reviews analyzed: {report.stats.reviews_clustered}")
    lines.append("")
    lines.append("Top themes this week:")
    for teaser in teasers:
        lines.append(f"  • {teaser}")
    lines.append("")
    lines.append(f"Read full report: {doc_link}")
    lines.append("")
    lines.append(
        f"Generated at {report.generated_at} | "
        f"Pulse Agent v0.1.0"
    )

    return "\n".join(lines)


def build_email_body_html(
    report: PulseReport,
    doc_url: str = "",
) -> str:
    """Build the HTML email body.

    Args:
        report: The PulseReport to render.
        doc_url: URL to the Google Doc (optional; placeholder if empty).
    """
    teasers = _build_theme_teasers(report)
    doc_link = _build_doc_link(doc_url, "")
    product_display = report.product.title()

    html_parts: list[str] = []
    html_parts.append(
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        '<body style="font-family: Arial, sans-serif; max-width: 600px; '
        'margin: 0 auto; padding: 20px; color: #333;">'
    )

    # Header
    html_parts.append(
        f'<h2 style="color: #1a73e8; margin-bottom: 4px;">'
        f'{product_display} — Week {report.iso_week}'
        f'</h2>'
    )
    html_parts.append(
        f'<p style="color: #666; font-size: 13px; margin-top: 0;">'
        f'Period: {report.period.start_date} to {report.period.end_date} | '
        f'{report.stats.reviews_clustered} reviews analyzed'
        f'</p>'
    )

    # Theme teasers
    html_parts.append(
        '<h3 style="color: #333; margin-bottom: 8px;">Top themes</h3>'
        '<ul style="padding-left: 20px; line-height: 1.6;">'
    )
    for teaser in teasers:
        html_parts.append(f'<li>{_html_escape(teaser)}</li>')
    html_parts.append('</ul>')

    # CTA button
    if doc_url:
        html_parts.append(
            f'<p style="margin-top: 24px;">'
            f'<a href="{doc_url}" '
            f'style="background-color: #1a73e8; color: white; '
            f'padding: 10px 20px; text-decoration: none; '
            f'border-radius: 4px; display: inline-block;">'
            f'Read full report &rarr;'
            f'</a>'
            f'</p>'
        )
    else:
        html_parts.append(
            f'<p style="margin-top: 24px; color: #666; font-style: italic;">'
            f'{doc_link}'
            f'</p>'
        )

    # Footer
    html_parts.append(
        '<hr style="border: none; border-top: 1px solid #eee; margin-top: 24px;">'
        f'<p style="color: #999; font-size: 11px;">'
        f'Generated at {report.generated_at} | '
        f'Pulse Agent v0.1.0'
        f'</p>'
    )

    html_parts.append('</body></html>')
    return "\n".join(html_parts)


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_email_payload(
    report: PulseReport,
    *,
    recipients: list[str] | None = None,
    doc_url: str = "",
) -> dict:
    """Build the complete email payload for Gmail MCP.

    Args:
        report: The PulseReport to render.
        recipients: List of email addresses (optional; for snapshot tests).
        doc_url: URL to the Google Doc.

    Returns:
        A dict with subject, body_text, body_html, and recipients keys,
        suitable for passing to a Gmail MCP tool.
    """
    return {
        "subject": build_subject(report),
        "body_text": build_email_body_text(report, doc_url=doc_url),
        "body_html": build_email_body_html(report, doc_url=doc_url),
        "recipients": recipients or [],
    }
