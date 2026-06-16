"""LLM summarization — Phase 4.

Summarizes ranked clusters into a PulseReport using Groq llama-3.3-70b-versatile.
Includes quote validation against source snippets and fallback retry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from groq import Groq

from pulse.models.models import (
    ActionIdea,
    AudienceNotes,
    Cluster,
    PulseReport,
    PulseReportPeriod,
    PulseReportStats,
    Theme,
)
from pulse.summarize.prompts import SYSTEM_PROMPT, build_per_cluster_prompt

logger = logging.getLogger(__name__)

# Groq model.
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
# Rate limits (Groq free tier).
RPM_LIMIT = 30
TPM_LIMIT = 12_000
# Minimum seconds between LLM calls to stay under RPM.
_MIN_CALL_INTERVAL_S = max(60.0 / RPM_LIMIT, 2.0)
# Max retries per cluster on quote validation failure.
MAX_QUOTE_RETRIES = 2
# Snippet truncation: max characters per cluster prompt's snippet block.
MAX_SNIPPET_CHARS = 2_000


def _now_ist_iso() -> str:
    """Return current datetime in IST as ISO-8601 string."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).isoformat()


def _compact_json(obj: dict | list) -> str:
    """Compact JSON string without newlines for logging."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Quote validation
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def validate_quote(quote: str, source_texts: list[str]) -> bool:
    """Return True if *quote* appears verbatim (after whitespace normalization)
    as a substring of any source text.

    Args:
        quote: Claimed verbatim quote from the LLM.
        source_texts: Original review snippet texts to validate against.
    """
    needle = _normalize_whitespace(quote.lower())
    if not needle:
        return False
    return any(needle in _normalize_whitespace(t.lower()) for t in source_texts)


def _all_quotes_valid(quotes: list[str], source_texts: list[str]) -> bool:
    return all(validate_quote(q, source_texts) for q in quotes)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it to your Groq API key."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Per-cluster summarization
# ---------------------------------------------------------------------------


def _call_llm(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_LLM_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Tuple[dict, int, int]:
    """Call the Groq LLM and return (parsed_json, prompt_tokens, completion_tokens).

    Raises ValueError on JSON parse failure.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content or ""
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    # Strip markdown code fences if present.
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        content = re.sub(r"^```(?:json)?\s*", "", content)
        # Remove closing fence
        content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.error("LLM returned unparseable JSON: %s", content[:500])
        raise ValueError(f"Failed to parse LLM JSON response: {content[:200]}")

    return parsed, prompt_tokens, completion_tokens


def _summarize_cluster(
    client: Groq,
    cluster: Cluster,
    all_source_texts: list[str],
    model: str,
) -> Tuple[Optional[Theme], int, int]:
    """Summarize a single cluster.  Returns (theme_or_None, prompt_tokens, completion_tokens)."""
    rank = cluster.rank
    cluster_size = cluster.cluster_size
    avg_rating = cluster.avg_rating
    earliest = cluster.earliest_date.strftime("%Y-%m-%d") if cluster.earliest_date else "N/A"
    latest = cluster.latest_date.strftime("%Y-%m-%d") if cluster.latest_date else "N/A"

    # Build snippets string, truncating to budget.
    snippet_lines = []
    char_count = 0
    for snip in cluster.representative_snippets:
        if char_count + len(snip) > MAX_SNIPPET_CHARS:
            break
        snippet_lines.append(f"- \"{snip}\"")
        char_count += len(snip) + 4

    snippets_str = "\n".join(snippet_lines)

    user_prompt = build_per_cluster_prompt(
        rank=rank,
        cluster_size=cluster_size,
        avg_rating=avg_rating,
        earliest_date=earliest,
        latest_date=latest,
        snippets=snippets_str,
    )

    # Source texts for quote validation (use the same snippets).
    source_texts = list(cluster.representative_snippets)

    prompt_tokens_total = 0
    completion_tokens_total = 0

    for attempt in range(1 + MAX_QUOTE_RETRIES):
        if attempt > 0:
            logger.info(
                "Cluster #%d: retry %d/%d due to quote validation failure.",
                rank, attempt, MAX_QUOTE_RETRIES,
            )
            # Fallback prompt: restrict to snippets only.
            fallback = (
                user_prompt
                + "\n\nIMPORTANT: In your previous response, at least one quote was NOT "
                "a verbatim substring of the snippets above.  Re-read the snippets and "
                "output ONLY quotes that appear EXACTLY as substrings in the snippets. "
                "Do NOT modify, paraphrase, or truncate them."
            )
            prompt = fallback
        else:
            prompt = user_prompt

        try:
            parsed, pt, ct = _call_llm(client, SYSTEM_PROMPT, prompt, model=model)
            prompt_tokens_total += pt
            completion_tokens_total += ct

            quotes: list[str] = parsed.get("quotes", [])
            if not isinstance(quotes, list):
                quotes = []

            theme_name = parsed.get("theme_name", f"Theme #{rank}")
            theme_summary = parsed.get("theme_summary", "")

            action_ideas_raw = parsed.get("action_ideas", [])
            if not isinstance(action_ideas_raw, list):
                action_ideas_raw = []

            action_ideas = [
                ActionIdea(
                    title=a.get("title", ""),
                    rationale=a.get("rationale", ""),
                )
                for a in action_ideas_raw[:2]
                if isinstance(a, dict)
            ]

            # Validate quotes.
            if not quotes or not _all_quotes_valid(quotes, source_texts):
                if attempt < MAX_QUOTE_RETRIES:
                    continue
                else:
                    logger.warning(
                        "Cluster #%d: quote validation failed after %d retries. "
                        "Dropping unvalidated quotes.",
                        rank, MAX_QUOTE_RETRIES,
                    )
                    quotes = [q for q in quotes if validate_quote(q, source_texts)]

            return Theme(
                rank=rank,
                name=theme_name,
                summary=theme_summary,
                cluster_size=cluster_size,
                avg_rating=avg_rating,
                quotes=quotes,
                action_ideas=action_ideas,
            ), prompt_tokens_total, completion_tokens_total

        except Exception as exc:
            logger.exception(
                "Cluster #%d, attempt %d failed",
                rank,
                attempt + 1,
            )
            if attempt < MAX_QUOTE_RETRIES:
                time.sleep(2)
                continue
            return None, prompt_tokens_total, completion_tokens_total

    return None, prompt_tokens_total, completion_tokens_total


# ---------------------------------------------------------------------------
# Audience notes
# ---------------------------------------------------------------------------


def _generate_audience_notes(
    client: Groq,
    themes: list[Theme],
    model: str,
) -> Tuple[AudienceNotes, int, int]:
    """Generate audience-specific notes (product, support, leadership)."""
    theme_summaries = "\n".join(
        f"- {t.name}: {t.summary}" for t in themes
    )

    prompt = f"""Based on these app review themes, write short (1-2 sentence) notes for three audiences.

THEMES:
{theme_summaries}

Return ONLY a JSON object:
{{
  "product": "What the product team should focus on",
  "support": "What customer support should prepare for",
  "leadership": "High-level business impact insight"
}}
"""

    parsed, pt, ct = _call_llm(client, SYSTEM_PROMPT, prompt, model=model)
    return (
        AudienceNotes(
            product=parsed.get("product", ""),
            support=parsed.get("support", ""),
            leadership=parsed.get("leadership", ""),
        ),
        pt,
        ct,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def summarize(
    clusters: List[Cluster],
    *,
    model: str = DEFAULT_LLM_MODEL,
    product: str = "groww",
    iso_week: str = "2026-W23",
    start_date: str = "",
    end_date: str = "",
    window_weeks: int = 10,
    total_reviews_fetched: int = 0,
    reviews_after_dedupe: int = 0,
    reviews_clustered: int = 0,
    clusters_found: int = 0,
) -> Tuple[PulseReport, dict]:
    """Summarize top clusters into a PulseReport using Groq LLM.

    Args:
        clusters: Ranked Cluster objects from Phase 3 (top-K already selected).
        model: Groq model ID.
        product: Product identifier.
        iso_week: ISO week string (e.g. "2026-W23").
        start_date: Period start date string.
        end_date: Period end date string.
        window_weeks: Review lookback window in weeks.
        total_reviews_fetched: Total reviews fetched by MCP.
        reviews_after_dedupe: Reviews after deduplication.
        reviews_clustered: Reviews assigned to non-noise clusters.
        clusters_found: Total clusters discovered by HDBSCAN.

    Returns:
        PulseReport: Complete report with themes and audience notes.
        token_usage dict: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}.
    """
    if not clusters:
        raise ValueError("No clusters provided for summarization.")

    client = _get_groq_client()

    themes: list[Theme] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, cluster in enumerate(clusters):
        logger.info(
            "Summarizing cluster #%d/%d (size=%d, avg_rating=%.1f) …",
            i + 1, len(clusters), cluster.cluster_size, cluster.avg_rating,
        )

        theme_result, p_tok, c_tok = _summarize_cluster(
            client=client,
            cluster=cluster,
            all_source_texts=list(cluster.representative_snippets),
            model=model,
        )

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        if theme_result:
            themes.append(theme_result)
            logger.info(
                "  → theme='%s', quotes=%d, actions=%d",
                theme_result.name,
                len(theme_result.quotes),
                len(theme_result.action_ideas),
            )
        else:
            logger.warning("Cluster #%d summarization failed — skipping.", cluster.rank)

        # Rate-limit: pause between calls to stay under RPM.
        if i < len(clusters) - 1:
            time.sleep(_MIN_CALL_INTERVAL_S)

    # Generate audience notes.
    logger.info("Generating audience notes …")
    audience_notes, apt, act = _generate_audience_notes(client, themes, model)
    total_prompt_tokens += apt
    total_completion_tokens += act

    # If any themes were dropped, renumber ranks.
    for idx, theme in enumerate(themes):
        theme.rank = idx + 1

    report = PulseReport(
        product=product,
        iso_week=iso_week,
        period=PulseReportPeriod(
            start_date=start_date,
            end_date=end_date,
            window_weeks=window_weeks,
        ),
        stats=PulseReportStats(
            total_reviews_fetched=total_reviews_fetched,
            reviews_after_dedupe=reviews_after_dedupe,
            reviews_clustered=reviews_clustered,
            clusters_found=clusters_found,
            top_themes_selected=len(themes),
        ),
        themes=themes,
        audience_notes=audience_notes,
        generated_at=_now_ist_iso(),
    )

    token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }

    logger.info(
        "Summarization complete: %d themes, token_usage=%s",
        len(themes), _compact_json(token_usage),
    )

    return report, token_usage

