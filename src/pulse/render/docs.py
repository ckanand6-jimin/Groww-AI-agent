"""Google Docs section builder — Phase 5.

Transforms a PulseReport into a Google Docs batch-update requests payload
suitable for the Docs API `batchUpdate` method (or a MCP tool wrapping it).

Architecture reference: architecture.md §5.2
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pulse.models.models import PulseReport


def build_heading_text(report: PulseReport) -> str:
    """Build the deterministic H2 heading for idempotency.

    Format: {ProductTitle} — Week {ISO_WEEK} ({start_date} – {end_date})

    Example: Groww — Week 2026-W23 (2026-03-31 – 2026-06-08)
    """
    product_display = report.product.title()
    return (
        f"{product_display} — Week {report.iso_week} "
        f"({report.period.start_date} – {report.period.end_date})"
    )


def _build_intro_paragraph(report: PulseReport) -> str:
    """Build the introductory paragraph below the H2 heading."""
    return (
        f"Period covered: {report.period.start_date} to {report.period.end_date} "
        f"({report.period.window_weeks}-week rolling window). "
        f"Reviews fetched: {report.stats.total_reviews_fetched}, "
        f"after deduplication: {report.stats.reviews_after_dedupe}, "
        f"clustered into {report.stats.clusters_found} themes, "
        f"top {report.stats.top_themes_selected} selected for this report. "
        f"Generated at {report.generated_at}."
    )


def _build_themes_section(report: PulseReport) -> List[str]:
    """Build bullet points for the 'Top themes' section."""
    bullets: List[str] = []
    for theme in report.themes:
        bullets.append(
            f"{theme.name} — {theme.summary} "
            f"(size: {theme.cluster_size}, avg rating: {theme.avg_rating:.1f})"
        )
    return bullets


def _build_quotes_section(report: PulseReport) -> List[str]:
    """Build bullet points for the 'Real user quotes' section."""
    bullets: List[str] = []
    for theme in report.themes:
        for quote in theme.quotes:
            bullets.append(f'"{quote}"')
    return bullets


def _build_actions_section(report: PulseReport) -> List[str]:
    """Build bullet points for the 'Action ideas' section."""
    bullets: List[str] = []
    for theme in report.themes:
        for action in theme.action_ideas:
            bullets.append(f"{action.title}: {action.rationale}")
    return bullets


def _build_audience_section(report: PulseReport) -> List[str]:
    """Build the 'Who this helps' narrative section."""
    notes = report.audience_notes
    return [
        f"Product team: {notes.product}",
        f"Support team: {notes.support}",
        f"Leadership: {notes.leadership}",
    ]


def build_doc_content(report: PulseReport) -> dict:
    """Build the complete Doc section content as a structured dict.

    Returns a dict with:
        heading_text: The H2 heading string (used for idempotency find_heading).
        sections: Ordered list of (heading, body_blocks) tuples representing
                  the section structure.
    """
    return {
        "heading_text": build_heading_text(report),
        "sections": [
            {
                "heading_level": None,  # intro paragraph (no heading)
                "heading": None,
                "body_type": "paragraph",
                "body": _build_intro_paragraph(report),
            },
            {
                "heading_level": 3,
                "heading": "Top themes",
                "body_type": "bullets",
                "bullets": _build_themes_section(report),
            },
            {
                "heading_level": 3,
                "heading": "Real user quotes",
                "body_type": "bullets",
                "bullets": _build_quotes_section(report),
            },
            {
                "heading_level": 3,
                "heading": "Action ideas",
                "body_type": "bullets",
                "bullets": _build_actions_section(report),
            },
            {
                "heading_level": 3,
                "heading": "Who this helps",
                "body_type": "bullets",
                "bullets": _build_audience_section(report),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Google Docs batch-update request builder
# ---------------------------------------------------------------------------


def _json_str(text: str) -> str:
    """JSON-escape a string for embedding in batch-update text."""
    return json.dumps(text)


def build_batch_update_requests(report: PulseReport) -> dict:
    """Build a Google Docs API batchUpdate request body.

    Produces a payload that appends the full weekly section to the end
    of the document, with proper heading styles and bullet formatting.

    The returned dict can be passed directly to the Docs API or to a
    Docs MCP tool that wraps batchUpdate.

    Returns:
        A dict with 'requests' key containing the ordered list of
        Google Docs API Request objects.
    """
    requests: List[Dict[str, Any]] = []
    heading_text = build_heading_text(report)
    intro = _build_intro_paragraph(report)

    # --- Build the full text content with structural markers ---
    # We insert all text at once using structural markers, then apply
    # paragraph styles and bullet formatting.

    lines: List[str] = []

    # H2: heading
    lines.append(heading_text)

    # Intro paragraph
    lines.append(intro)

    # H3: Top themes
    lines.append("Top themes")
    for bullet in _build_themes_section(report):
        lines.append(f"• {bullet}")

    # H3: Real user quotes
    lines.append("Real user quotes")
    for bullet in _build_quotes_section(report):
        lines.append(f"• {bullet}")

    # H3: Action ideas
    lines.append("Action ideas")
    for bullet in _build_actions_section(report):
        lines.append(f"• {bullet}")

    # H3: Who this helps
    lines.append("Who this helps")
    for bullet in _build_audience_section(report):
        lines.append(f"• {bullet}")

    # Join lines with newlines.
    full_text = "\n".join(lines) + "\n"

    # --- Request 1: Insert all text at end of document ---
    requests.append({
        "insertText": {
            "location": {"index": 1},
            "text": full_text,
        }
    })

    # --- Apply paragraph styles ---
    # After inserting text, we need to apply NamedStyles to the paragraphs.
    # Paragraphs are indexed by their startIndex in the inserted text.
    # We track the current index as we insert.
    #
    # For simplicity, we use a single insertText then apply paragraph style
    # updates. Each paragraph is separated by a newline.

    # Calculate paragraph start indices (0-based after insertion at index 1).
    # index 1 = start of first paragraph (heading)
    current_idx = 1  # start index of inserted text

    para_starts: list[tuple[int, str]] = []  # (startIndex, namedStyleType)

    # H2 heading
    para_starts.append((current_idx, "HEADING_2"))
    current_idx += len(heading_text) + 1  # +1 for newline

    # Intro paragraph → NORMAL_TEXT
    para_starts.append((current_idx, "NORMAL_TEXT"))
    current_idx += len(intro) + 1

    # H3: Top themes → HEADING_3
    para_starts.append((current_idx, "HEADING_3"))
    current_idx += len("Top themes") + 1

    # Bullet items → NORMAL_TEXT (we'll convert to bullets separately)
    for bullet in _build_themes_section(report):
        line = f"• {bullet}"
        para_starts.append((current_idx, "NORMAL_TEXT"))
        current_idx += len(line) + 1

    # H3: Real user quotes → HEADING_3
    para_starts.append((current_idx, "HEADING_3"))
    current_idx += len("Real user quotes") + 1

    for bullet in _build_quotes_section(report):
        line = f"• {bullet}"
        para_starts.append((current_idx, "NORMAL_TEXT"))
        current_idx += len(line) + 1

    # H3: Action ideas → HEADING_3
    para_starts.append((current_idx, "HEADING_3"))
    current_idx += len("Action ideas") + 1

    for bullet in _build_actions_section(report):
        line = f"• {bullet}"
        para_starts.append((current_idx, "NORMAL_TEXT"))
        current_idx += len(line) + 1

    # H3: Who this helps → HEADING_3
    para_starts.append((current_idx, "HEADING_3"))
    current_idx += len("Who this helps") + 1

    for bullet in _build_audience_section(report):
        line = f"• {bullet}"
        para_starts.append((current_idx, "NORMAL_TEXT"))
        current_idx += len(line) + 1

    # Request 2+: Apply paragraph styles.
    for start_idx, style_type in para_starts:
        requests.append({
            "updateParagraphStyle": {
                "range": {
                    "startIndex": start_idx,
                    "endIndex": start_idx + 1,
                },
                "paragraphStyle": {
                    "namedStyleType": style_type,
                },
                "fields": "namedStyleType",
            }
        })

    return {"requests": requests}
