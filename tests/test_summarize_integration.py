"""Phase 4 integration test — run full pipeline and validate summarization output.

Requires GROQ_API_KEY env var set to a valid Groq API key.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure we can import from pulse.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse.analysis import analyze
from pulse.ingest.adapter import ingest_from_cache
from pulse.summarize import summarize, validate_quote


def _get_cache_dir() -> str:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(test_dir))
    return os.path.join(repo_root, "data", "cache", "groww", "2026-06-09")


def main():
    cache_dir = _get_cache_dir()
    if not os.path.isdir(cache_dir):
        print(f"ERROR: Cache directory not found: {cache_dir}")
        return 1

    print("=" * 70)
    print("PHASE 4 INTEGRATION TEST")
    print("=" * 70)

    # ---- 1. Ingest ----
    print("\n[1/3] Ingesting reviews from cache …")
    reviews, ingest_stats = ingest_from_cache(cache_dir, "2026-W23", "groww")
    print(f"  Ingested: {len(reviews)} reviews (from {ingest_stats['fetched']} raw)")

    # ---- 2. Analyze (Phase 3) ----
    print("\n[2/3] Running Phase 3 clustering …")
    clusters, analysis_stats = analyze(reviews, top_k=5)
    print(f"  Clusters found: {analysis_stats['clusters_found']}")
    print(f"  Reviews clustered: {analysis_stats['reviews_clustered']}")
    print(f"  Top-K returned: {len(clusters)}")
    for c in clusters:
        print(f"    #{c.rank}: cluster_id={c.cluster_id}, size={c.cluster_size}, "
              f"avg_rating={c.avg_rating:.1f}, score={c.rank_score:.4f}")

    # ---- 3. Summarize (Phase 4) ----
    print("\n[3/3] Running Phase 4 LLM summarization (Groq llama-3.3-70b-versatile) …")
    report, token_usage = summarize(
        clusters,
        model="llama-3.3-70b-versatile",
        product="groww",
        iso_week="2026-W23",
        start_date="2026-03-31",
        end_date="2026-06-08",
        window_weeks=10,
        total_reviews_fetched=ingest_stats["fetched"],
        reviews_after_dedupe=len(reviews),
        reviews_clustered=analysis_stats["reviews_clustered"],
        clusters_found=analysis_stats["clusters_found"],
    )

    # ---- Print PulseReport ----
    print("\n" + "=" * 70)
    print("PULSE REPORT")
    print("=" * 70)

    report_dict = json.loads(report.model_dump_json())
    print(json.dumps(report_dict, indent=2, ensure_ascii=False))

    # ---- Validate quotes ----
    print("\n" + "=" * 70)
    print("QUOTE VALIDATION")
    print("=" * 70)

    all_valid = True
    for theme in report.themes:
        # Find the corresponding cluster to get source snippets.
        cluster = next((c for c in clusters if c.rank == theme.rank), None)
        source_snippets = list(cluster.representative_snippets) if cluster else []

        print(f"\n#{theme.rank} {theme.name}:")
        for i, quote in enumerate(theme.quotes):
            valid = validate_quote(quote, source_snippets)
            status = "[PASS]" if valid else "[FAIL]"
            if not valid:
                all_valid = False
            print(f"  [{status}] Q{i+1}: \"{quote[:100]}{'...' if len(quote) > 100 else ''}\"")
        for action in theme.action_ideas:
            print(f"  [ACTION ] {action.title}: {action.rationale[:80]}...")

    # ---- Token usage ----
    print("\n" + "=" * 70)
    print("TOKEN USAGE")
    print("=" * 70)
    print(f"  Prompt tokens:     {token_usage['prompt_tokens']}")
    print(f"  Completion tokens: {token_usage['completion_tokens']}")
    print(f"  Total tokens:      {token_usage['total_tokens']}")
    print(f"  (RPM limit: {30}, TPM limit: {12_000}, TPD limit: {100_000})")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Themes generated:      {len(report.themes)}")
    print(f"  All quotes validated:  {'YES' if all_valid else 'NO — see failures above'}")
    print(f"  Token usage:           {token_usage['total_tokens']} tokens")
    print(f"  Audience notes:        product={bool(report.audience_notes.product)}, "
          f"support={bool(report.audience_notes.support)}, "
          f"leadership={bool(report.audience_notes.leadership)}")

    if all_valid:
        print("\n[PASS] Phase 4 integration test PASSED")
        return 0
    else:
        print("\n[FAIL] Phase 4 integration test FAILED — some quotes not validated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
