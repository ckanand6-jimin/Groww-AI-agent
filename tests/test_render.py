"""Snapshot tests for Phase 5 report rendering.

Tests doc batch-update request structure and email HTML/text output
using a fixture PulseReport matching the architecture sample.
"""

from __future__ import annotations

import json
import os

import pytest

from pulse.models.models import (
    ActionIdea,
    AudienceNotes,
    PulseReport,
    PulseReportPeriod,
    PulseReportStats,
    Theme,
)
from pulse.render import preview, render
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


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report() -> PulseReport:
    """A realistic PulseReport matching the architecture sample in context.md."""
    return PulseReport(
        product="groww",
        iso_week="2026-W23",
        period=PulseReportPeriod(
            start_date="2026-03-31",
            end_date="2026-06-08",
            window_weeks=10,
        ),
        stats=PulseReportStats(
            total_reviews_fetched=1240,
            reviews_after_dedupe=1180,
            reviews_clustered=1100,
            clusters_found=18,
            top_themes_selected=3,
        ),
        themes=[
            Theme(
                rank=1,
                name="App performance & bugs",
                summary="Lag, crashes during trading hours; login/session timeouts.",
                cluster_size=142,
                avg_rating=2.1,
                quotes=[
                    "The app freezes exactly when the market opens, very frustrating.",
                    "Login keeps failing during peak hours.",
                ],
                action_ideas=[
                    ActionIdea(
                        title="Stabilize peak-time performance",
                        rationale="Scale infra during market hours; improve crash visibility.",
                    ),
                ],
            ),
            Theme(
                rank=2,
                name="Customer support friction",
                summary="Slow responses; unresolved tickets.",
                cluster_size=98,
                avg_rating=1.8,
                quotes=[
                    "Support takes days to reply and doesn't solve the issue.",
                    "Chat support is useless, no one responds.",
                ],
                action_ideas=[
                    ActionIdea(
                        title="Improve support SLA visibility",
                        rationale="Expected response time in-app; ticket status tracking.",
                    ),
                ],
            ),
            Theme(
                rank=3,
                name="UX & feature gaps",
                summary="Confusing navigation for portfolio insights; missing advanced analytics.",
                cluster_size=76,
                avg_rating=2.5,
                quotes=[
                    "Good for beginners but lacks detailed analysis tools.",
                    "Portfolio view is hard to find and understand.",
                ],
                action_ideas=[
                    ActionIdea(
                        title="Enhance power-user features",
                        rationale="Advanced portfolio analytics; clearer investments navigation.",
                    ),
                ],
            ),
        ],
        audience_notes=AudienceNotes(
            product="Focus on app stability during trading hours and UX improvements for portfolio navigation.",
            support="Prepare for increased ticket volume around login issues and brokerage queries; improve SLA visibility.",
            leadership="User sentiment is declining due to performance and support gaps; addressing top 3 themes could improve retention by 15-20%.",
        ),
        generated_at="2026-06-08T06:30:00+05:30",
    )


# ---------------------------------------------------------------------------
# Heading convention (5.1)
# ---------------------------------------------------------------------------


class TestHeadingConvention:
    def test_heading_format(self, sample_report: PulseReport):
        heading = build_heading_text(sample_report)
        assert heading.startswith("Groww")
        assert "Week 2026-W23" in heading
        assert "2026-03-31" in heading
        assert "2026-06-08" in heading
        # Must be deterministic
        heading2 = build_heading_text(sample_report)
        assert heading == heading2

    def test_heading_contains_iso_week(self, sample_report: PulseReport):
        heading = build_heading_text(sample_report)
        assert "2026-W23" in heading


# ---------------------------------------------------------------------------
# Doc content structure (5.2)
# ---------------------------------------------------------------------------


class TestDocContent:
    def test_doc_content_has_all_sections(self, sample_report: PulseReport):
        content = build_doc_content(sample_report)
        sections = content["sections"]
        assert len(sections) == 5  # intro + 4 H3 sections

        section_headings = [s["heading"] for s in sections]
        assert None in section_headings  # intro has no heading
        assert "Top themes" in section_headings
        assert "Real user quotes" in section_headings
        assert "Action ideas" in section_headings
        assert "Who this helps" in section_headings

    def test_doc_content_heading_text_present(self, sample_report: PulseReport):
        content = build_doc_content(sample_report)
        assert "heading_text" in content
        assert "Groww" in content["heading_text"]

    def test_themes_section_has_all_themes(self, sample_report: PulseReport):
        content = build_doc_content(sample_report)
        themes_section = next(s for s in content["sections"] if s["heading"] == "Top themes")
        bullets = themes_section["bullets"]
        assert len(bullets) == 3  # 3 themes

    def test_quotes_section_has_all_quotes(self, sample_report: PulseReport):
        content = build_doc_content(sample_report)
        quotes_section = next(s for s in content["sections"] if s["heading"] == "Real user quotes")
        bullets = quotes_section["bullets"]
        assert len(bullets) == 6  # 2 quotes × 3 themes


# ---------------------------------------------------------------------------
# Doc batch-update requests (5.2)
# ---------------------------------------------------------------------------


class TestBatchUpdateRequests:
    def test_requests_not_empty(self, sample_report: PulseReport):
        payload = build_batch_update_requests(sample_report)
        assert "requests" in payload
        assert len(payload["requests"]) > 0

    def test_first_request_is_insert_text(self, sample_report: PulseReport):
        payload = build_batch_update_requests(sample_report)
        first = payload["requests"][0]
        assert "insertText" in first
        assert "text" in first["insertText"]

    def test_inserted_text_contains_heading(self, sample_report: PulseReport):
        payload = build_batch_update_requests(sample_report)
        text = payload["requests"][0]["insertText"]["text"]
        assert "Groww" in text
        assert "2026-W23" in text

    def test_inserted_text_contains_all_themes(self, sample_report: PulseReport):
        payload = build_batch_update_requests(sample_report)
        text = payload["requests"][0]["insertText"]["text"]
        for theme in sample_report.themes:
            assert theme.name in text

    def test_has_heading_style_requests(self, sample_report: PulseReport):
        payload = build_batch_update_requests(sample_report)
        style_requests = [
            r for r in payload["requests"] if "updateParagraphStyle" in r
        ]
        assert len(style_requests) > 0
        # Should have at least one HEADING_2 and HEADING_3
        named_styles = [
            r["updateParagraphStyle"]["paragraphStyle"]["namedStyleType"]
            for r in style_requests
        ]
        assert "HEADING_2" in named_styles
        assert "HEADING_3" in named_styles

    def test_batch_update_is_valid_json(self, sample_report: PulseReport):
        payload = build_batch_update_requests(sample_report)
        # Should be serializable.
        serialized = json.dumps(payload, ensure_ascii=False)
        assert len(serialized) > 0
        # And deserializable.
        roundtripped = json.loads(serialized)
        assert roundtripped == payload


# ---------------------------------------------------------------------------
# Email subject (5.3)
# ---------------------------------------------------------------------------


class TestEmailSubject:
    def test_subject_format(self, sample_report: PulseReport):
        subject = build_subject(sample_report)
        assert "Groww Weekly Review Pulse" in subject
        assert "2026-W23" in subject

    def test_subject_is_deterministic(self, sample_report: PulseReport):
        s1 = build_subject(sample_report)
        s2 = build_subject(sample_report)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Email body text (5.3)
# ---------------------------------------------------------------------------


class TestEmailBodyText:
    def test_body_contains_theme_teasers(self, sample_report: PulseReport):
        body = build_email_body_text(sample_report)
        for theme in sample_report.themes:
            assert theme.name in body

    def test_body_contains_period(self, sample_report: PulseReport):
        body = build_email_body_text(sample_report)
        assert "2026-03-31" in body
        assert "2026-06-08" in body

    def test_body_has_read_full_report_link(self, sample_report: PulseReport):
        body = build_email_body_text(sample_report, doc_url="https://docs.google.com/doc/abc")
        assert "Read full report" in body
        assert "https://docs.google.com/doc/abc" in body

    def test_body_placeholder_when_no_doc_url(self, sample_report: PulseReport):
        body = build_email_body_text(sample_report)
        assert "[Document link not yet configured]" in body

    def test_body_does_not_duplicate_full_report(self, sample_report: PulseReport):
        """Email must be a teaser, not a full report duplicate (exit criterion)."""
        body = build_email_body_text(sample_report)
        # Should contain theme teasers (name + summary)
        assert "Top themes this week:" in body
        # Body should NOT contain verbatim action rationales (those are full-report detail)
        assert "Scale infra during market hours" not in body
        assert "Advanced portfolio analytics" not in body
        # Teaser should be concise: each theme gets one line
        teaser_count = body.count("  \u2022")
        assert teaser_count == 3  # exactly 3 theme teaser bullets


# ---------------------------------------------------------------------------
# Email body HTML (5.3)
# ---------------------------------------------------------------------------


class TestEmailBodyHtml:
    def test_html_is_valid_structure(self, sample_report: PulseReport):
        html = build_email_body_html(sample_report)
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "<body" in html
        assert "</body>" in html

    def test_html_contains_theme_teasers(self, sample_report: PulseReport):
        html = build_email_body_html(sample_report)
        for theme in sample_report.themes:
            # Theme names with & are HTML-escaped, check for escaped form.
            escaped_name = (
                theme.name.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            assert escaped_name in html, f"Expected '{escaped_name}' in HTML"

    def test_html_has_cta_when_doc_url_provided(self, sample_report: PulseReport):
        html = build_email_body_html(sample_report, doc_url="https://docs.google.com/doc/abc")
        assert "Read full report" in html
        assert "https://docs.google.com/doc/abc" in html

    def test_html_escapes_special_chars(self):
        """HTML output must escape <, >, & in theme text."""
        report = PulseReport(
            product="groww",
            iso_week="2026-W23",
            period=PulseReportPeriod(start_date="2026-01-01", end_date="2026-06-01", window_weeks=10),
            stats=PulseReportStats(
                total_reviews_fetched=1, reviews_after_dedupe=1,
                reviews_clustered=1, clusters_found=1, top_themes_selected=1,
            ),
            themes=[
                Theme(
                    rank=1, name="Bug <script>alert('xss')</script>",
                    summary="Test & verification of <html> escaping.",
                    cluster_size=10, avg_rating=2.0,
                    quotes=["Safe quote."],
                    action_ideas=[ActionIdea(title="Fix", rationale="Because & reasons.")],
                ),
            ],
            audience_notes=AudienceNotes(product="p", support="s", leadership="l"),
            generated_at="2026-06-08T06:30:00+05:30",
        )
        html = build_email_body_html(report)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&amp;" in html

    def test_html_has_footer(self, sample_report: PulseReport):
        html = build_email_body_html(sample_report)
        assert "Pulse Agent v0.1.0" in html


# ---------------------------------------------------------------------------
# Email payload (5.3)
# ---------------------------------------------------------------------------


class TestEmailPayload:
    def test_payload_has_all_keys(self, sample_report: PulseReport):
        payload = build_email_payload(
            sample_report,
            recipients=["product@example.com"],
            doc_url="https://docs.google.com/doc/abc",
        )
        assert "subject" in payload
        assert "body_text" in payload
        assert "body_html" in payload
        assert "recipients" in payload
        assert payload["recipients"] == ["product@example.com"]

    def test_payload_default_recipients(self, sample_report: PulseReport):
        payload = build_email_payload(sample_report)
        assert payload["recipients"] == []


# ---------------------------------------------------------------------------
# Render integration (5.2 + 5.3 combined)
# ---------------------------------------------------------------------------


class TestRenderIntegration:
    def test_render_returns_all_sections(self, sample_report: PulseReport):
        result = render(sample_report, doc_url="https://docs.google.com/doc/abc")
        assert "heading_text" in result
        assert "doc_batch_update" in result
        assert "doc_content" in result
        assert "email_payload" in result

    def test_render_heading_matches_doc_and_email(self, sample_report: PulseReport):
        result = render(sample_report)
        heading = result["heading_text"]
        # Email subject should reference same week.
        assert "2026-W23" in result["email_payload"]["subject"]
        assert heading.startswith("Groww")


# ---------------------------------------------------------------------------
# Preview file writer (5.6)
# ---------------------------------------------------------------------------


class TestPreview:
    def test_preview_writes_files(self, sample_report: PulseReport, tmp_path):
        doc_path, email_path = preview(sample_report, str(tmp_path))

        assert os.path.exists(doc_path)
        assert os.path.exists(email_path)
        assert doc_path.endswith("preview.doc.json")
        assert email_path.endswith("preview.email.html")

    def test_preview_doc_is_valid_json(self, sample_report: PulseReport, tmp_path):
        doc_path, _ = preview(sample_report, str(tmp_path))
        with open(doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "heading_text" in data
        assert "doc_content" in data
        assert "batch_update_requests" in data

    def test_preview_email_is_html(self, sample_report: PulseReport, tmp_path):
        _, email_path = preview(sample_report, str(tmp_path))
        with open(email_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "Groww" in content



