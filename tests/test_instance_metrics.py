#!/usr/bin/env python3
"""
Quick test to verify instance metrics functions work correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lc_metrics import (
    build_corpus_statistics,
    calculate_instance_atp,
    calculate_instance_delta_p_backward,
    calculate_instance_boundary_entropy
)

# Sample constructions
test_constructions = [
    {
        'aux_chain': ['is', 'being', 'analyzed'],
        'pattern': 'DEF_NP:pres-be:Ø',
        'following_word': 'carefully'
    },
    {
        'aux_chain': ['has', 'been', 'analyzed'],
        'pattern': 'DEF_NP:perf-be:Ø',
        'following_word': 'thoroughly'
    },
    {
        'aux_chain': ['will', 'be', 'analyzed'],
        'pattern': 'DEF_NP:modal-be:Ø',
        'following_word': 'soon'
    },
    {
        'aux_chain': ['has', 'been', 'analyzed'],
        'pattern': 'DEF_NP:perf-be:Ø',
        'following_word': 'carefully'
    },
]

print("=" * 60)
print("Testing Instance Metrics Functions")
print("=" * 60)

# Test 1: Build corpus statistics
print("\n1. Building corpus statistics...")
corpus_stats = build_corpus_statistics(test_constructions)

print(f"   ✓ Unigrams: {len(corpus_stats['unigram_counts'])}")
print(f"   ✓ Bigrams: {len(corpus_stats['bigram_counts'])}")
print(f"   ✓ Total words: {corpus_stats['total_words']}")
print(f"   ✓ Patterns with followers: {len(corpus_stats['follower_counts'])}")

# Test 2: Calculate instance metrics
print("\n2. Calculating instance metrics for first construction...")
const = test_constructions[0]

atp = calculate_instance_atp(const, corpus_stats)
dpb = calculate_instance_delta_p_backward(const, corpus_stats)
hr = calculate_instance_boundary_entropy(const, corpus_stats)

print(f"   ✓ ATP: {atp:.4f}")
print(f"   ✓ ΔP_backward: {dpb:.4f}")
print(f"   ✓ H_r: {hr:.4f}")

# Test 3: Check thresholds
print("\n3. Checking thresholds (ATP ≥ 0.10, ΔP ≥ 0.10, H_r ≤ 2.8)...")
passes_atp = atp >= 0.10
passes_dpb = dpb >= 0.10
passes_hr = hr <= 2.8

print(f"   ATP ≥ 0.10: {'✓ PASS' if passes_atp else '✗ FAIL'}")
print(f"   ΔP ≥ 0.10: {'✓ PASS' if passes_dpb else '✗ FAIL'}")
print(f"   H_r ≤ 2.8: {'✓ PASS' if passes_hr else '✗ FAIL'}")

if passes_atp and passes_dpb and passes_hr:
    print("\n✅ Construction would be accepted as formulaic!")
else:
    print("\n❌ Construction would be filtered out.")

print("\n" + "=" * 60)
print("✓ All tests completed successfully!")
print("=" * 60)
