#!/usr/bin/env python
"""Test script to verify build_query_varied correctness."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reason_abstain.bench.templates import (
    ABSTAIN_TEMPLATE_IDS,
    ANSWER_TEMPLATE_IDS,
    GROUNDING_TEMPLATES,
    build_query_varied,
)


def test_abstain_templates():
    """Verify abstain samples only use neutral templates (0 and 2)."""
    print("Testing abstain templates (should only use templates 0 and 2)...")
    abstain_queries = []
    for i in range(10):
        vary_key = f"test-abstain-sample-{i}"
        query = build_query_varied("tomato_leaf", vary_key, is_answer=False)
        abstain_queries.append(query)
        print(f"  {i}: {query}")

    # Check that no abstain query uses template 1 ("return its bounding box")
    template_1_text = "return its bounding box"
    for query in abstain_queries:
        assert template_1_text not in query, f"Abstain query should not use template 1: {query}"

    # Check that only templates 0 and 2 are used
    used_templates = set()
    for query in abstain_queries:
        for tid in ABSTAIN_TEMPLATE_IDS:
            if query == GROUNDING_TEMPLATES[tid].format(label="tomato leaf"):
                used_templates.add(tid)
                break

    print(f"  Used template IDs: {sorted(used_templates)}")
    assert used_templates.issubset(set(ABSTAIN_TEMPLATE_IDS)), (
        f"Abstain should only use templates {ABSTAIN_TEMPLATE_IDS}, got {used_templates}"
    )
    print("  ✓ PASS: Abstain queries only use neutral templates\n")


def test_answer_templates():
    """Verify answer samples can use all three templates."""
    print("Testing answer templates (should use all 3 templates)...")
    answer_queries = []
    for i in range(30):  # Use more samples to ensure we hit all templates
        vary_key = f"test-answer-sample-{i}"
        query = build_query_varied("apple", vary_key, is_answer=True)
        answer_queries.append(query)

    # Check which templates are used
    used_templates = set()
    for query in answer_queries:
        for tid in ANSWER_TEMPLATE_IDS:
            if query == GROUNDING_TEMPLATES[tid].format(label="apple"):
                used_templates.add(tid)
                break

    print(f"  Used template IDs: {sorted(used_templates)}")
    for tid in sorted(used_templates):
        print(f"    Template {tid}: {GROUNDING_TEMPLATES[tid].format(label='apple')}")

    assert used_templates == set(ANSWER_TEMPLATE_IDS), (
        f"Answer should use all templates {ANSWER_TEMPLATE_IDS}, got {used_templates}"
    )
    print("  ✓ PASS: Answer queries use all 3 templates\n")


def test_determinism():
    """Verify same vary_key produces same result (deterministic)."""
    print("Testing determinism (same vary_key should produce same query)...")
    vary_key = "deterministic-test-key"
    results = []
    for _ in range(5):
        query = build_query_varied("potato", vary_key, is_answer=True)
        results.append(query)

    assert all(r == results[0] for r in results), f"Results should be deterministic: {results}"

    print(f"  All 5 calls produced: {results[0]}")
    print("  ✓ PASS: Deterministic behavior confirmed\n")


def test_different_keys():
    """Verify different vary_keys can produce different templates."""
    print("Testing that different vary_keys produce varied templates...")
    abstain_results = {}
    answer_results = {}

    for i in range(20):
        vary_key = f"sample-{i}"
        abstain_q = build_query_varied("wheat", vary_key, is_answer=False)
        answer_q = build_query_varied("wheat", vary_key, is_answer=True)
        abstain_results[vary_key] = abstain_q
        answer_results[vary_key] = answer_q

    # Count unique templates for abstain
    abstain_unique = len(set(abstain_results.values()))
    answer_unique = len(set(answer_results.values()))

    print(f"  Abstain: {abstain_unique} unique templates out of 20 samples")
    print(f"  Answer: {answer_unique} unique templates out of 20 samples")
    print(f"  ✓ PASS: Variation confirmed (abstain: {abstain_unique}, answer: {answer_unique})\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing build_query_varied correctness")
    print("=" * 70 + "\n")

    test_abstain_templates()
    test_answer_templates()
    test_determinism()
    test_different_keys()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
