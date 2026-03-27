"""
Statistical validation tests for schema pipeline.

Includes:
1. Permutation testing (TAM×COMP associations)
2. Ablation testing (filter necessity)

Usage:
    from validation_tests import (
        permutation_test_tam_comp,
        ablation_test_filters,
        ablation_test_dual_lane,
        print_validation_report
    )

    # Run tests
    perm_results = permutation_test_tam_comp(formulaic_instances)
    ablation_results = ablation_test_filters(valid_constructions)
    lane_results = ablation_test_dual_lane(metrics)

    # Print report
    print_validation_report(perm_results, ablation_results, lane_results)
"""

import random
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


def permutation_test_tam_comp(
    constructions: List[Dict],
    n_iterations: int = 1000,
    random_seed: int = 42
) -> Dict:
    """
    Test if TAM×COMP associations are stronger than random.

    Null Hypothesis (H0): TAM and COMP are independent (no association)
    Alternative Hypothesis (H1): TAM and COMP show constructional preferences

    Method:
        1. Calculate real schema count from actual data
        2. Shuffle TAM assignments n_iterations times
        3. Re-run dual-lane acceptance for each permutation
        4. Compare real count to permuted distribution
        5. p-value = proportion of permutations ≥ real count

    Args:
        constructions: List of formulaic instances (post instance-prefilters)
        n_iterations: Number of permutations (default: 1000)
        random_seed: Random seed for reproducibility

    Returns:
        Dict with:
        - real_schema_count: Actual schemas from data
        - permuted_counts: List of schema counts from permutations
        - mean_permuted: Mean of permuted counts
        - std_permuted: Standard deviation of permuted counts
        - p_value: Proportion of permutations ≥ real count
        - significant: True if p < 0.05

    Example:
        >>> results = permutation_test_tam_comp(formulaic_instances)
        >>> print(f"Real: {results['real_schema_count']}, p={results['p_value']}")
        Real: 32, p=0.001
    """
    from lc_metrics import calculate_lc_metrics
    from lc_metrics import apply_dual_lane_acceptance

    random.seed(random_seed)

    # Calculate real schema count
    print("  Calculating real schema count...")
    real_metrics = calculate_lc_metrics(constructions)
    real_accepted, _ = apply_dual_lane_acceptance(real_metrics, mode='production')
    real_count = len(real_accepted)

    print(f"  Real schemas: {real_count}")
    print(f"  Running {n_iterations} permutations...")

    # Run permutations
    permuted_counts = []

    for i in range(n_iterations):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i + 1}/{n_iterations} ({(i + 1)/n_iterations*100:.1f}%)")

        # Shuffle TAM values (break TAM-COMP associations)
        shuffled = [c.copy() for c in constructions]
        tam_values = [c['tam'] for c in shuffled]
        random.shuffle(tam_values)

        for const, new_tam in zip(shuffled, tam_values):
            const['tam'] = new_tam
            # Update pattern key to reflect new TAM
            const['pattern'] = f"{new_tam},{const['comp']}"

        # Re-calculate metrics and acceptance
        perm_metrics = calculate_lc_metrics(shuffled)
        perm_accepted, _ = apply_dual_lane_acceptance(perm_metrics, mode='production')
        permuted_counts.append(len(perm_accepted))

    # Calculate statistics
    mean_perm = np.mean(permuted_counts)
    std_perm = np.std(permuted_counts)
    p_value = sum(c >= real_count for c in permuted_counts) / n_iterations

    print(f"  Permuted mean: {mean_perm:.1f} (SD: {std_perm:.1f})")
    print(f"  p-value: {p_value:.4f}")

    return {
        'real_schema_count': real_count,
        'permuted_counts': permuted_counts,
        'mean_permuted': mean_perm,
        'std_permuted': std_perm,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def ablation_test_filters(
    constructions: List[Dict]
) -> Dict:
    """
    Test contribution of each instance filter.

    Removes one filter at a time and measures impact on:
    - Number of formulaic instances (post-prefilter)
    - Number of accepted schemas (post-dual-lane)

    Validates:
    - Each filter contributes meaningfully
    - No filter is redundant
    - H_r is expected to be the major filter

    Args:
        constructions: All valid constructions (post basic filters, pre instance-prefilters)

    Returns:
        Dict with results for each ablation condition:
        - full_pipeline: Baseline with all filters
        - no_atp: ATP filter removed
        - no_dpb: ΔP_backward filter removed
        - no_hr: Boundary entropy (H_r) filter removed

    Example:
        >>> results = ablation_test_filters(valid_constructions)
        >>> baseline = results['full_pipeline']['schema_count']
        >>> no_hr = results['no_hr']['schema_count']
        >>> print(f"H_r impact: {no_hr - baseline} schemas")
        H_r impact: 58 schemas
    """
    from lc_metrics import calculate_lc_metrics
    from lc_metrics import apply_dual_lane_acceptance

    results = {}

    # Baseline: Full pipeline (all three filters)
    print("  Testing full pipeline (baseline)...")
    full_filtered = [
        c for c in constructions
        if c.get('instance_atp', 0) >= 0.10
        and c.get('instance_dpb', 0) >= 0.10
        and c.get('instance_hr', float('inf')) <= 2.8
    ]
    full_metrics = calculate_lc_metrics(full_filtered)
    full_accepted, _ = apply_dual_lane_acceptance(full_metrics, mode='production')

    results['full_pipeline'] = {
        'formulaic_count': len(full_filtered),
        'schema_count': len(full_accepted),
        'description': 'All filters applied (baseline)'
    }

    # Ablation 1: Remove ATP filter
    print("  Testing without ATP filter...")
    no_atp = [
        c for c in constructions
        # Skip ATP ≥ 0.10
        if c.get('instance_dpb', 0) >= 0.10
        and c.get('instance_hr', float('inf')) <= 2.8
    ]
    metrics = calculate_lc_metrics(no_atp)
    accepted, _ = apply_dual_lane_acceptance(metrics, mode='production')

    results['no_atp'] = {
        'formulaic_count': len(no_atp),
        'schema_count': len(accepted),
        'description': 'Removed ATP ≥ 0.10 filter',
        'change_formulaic': len(no_atp) - len(full_filtered),
        'change_schemas': len(accepted) - len(full_accepted)
    }

    # Ablation 2: Remove ΔP_backward filter
    print("  Testing without ΔP_backward filter...")
    no_dpb = [
        c for c in constructions
        if c.get('instance_atp', 0) >= 0.10
        # Skip ΔP ≥ 0.10
        and c.get('instance_hr', float('inf')) <= 2.8
    ]
    metrics = calculate_lc_metrics(no_dpb)
    accepted, _ = apply_dual_lane_acceptance(metrics, mode='production')

    results['no_dpb'] = {
        'formulaic_count': len(no_dpb),
        'schema_count': len(accepted),
        'description': 'Removed ΔP_backward ≥ 0.10 filter',
        'change_formulaic': len(no_dpb) - len(full_filtered),
        'change_schemas': len(accepted) - len(full_accepted)
    }

    # Ablation 3: Remove boundary entropy (H_r) filter
    print("  Testing without H_r filter...")
    no_hr = [
        c for c in constructions
        if c.get('instance_atp', 0) >= 0.10
        and c.get('instance_dpb', 0) >= 0.10
        # Skip H_r ≤ 2.8
    ]
    metrics = calculate_lc_metrics(no_hr)
    accepted, _ = apply_dual_lane_acceptance(metrics, mode='production')

    results['no_hr'] = {
        'formulaic_count': len(no_hr),
        'schema_count': len(accepted),
        'description': 'Removed H_r ≤ 2.8 filter',
        'change_formulaic': len(no_hr) - len(full_filtered),
        'change_schemas': len(accepted) - len(full_accepted)
    }

    return results


def ablation_test_dual_lane(
    metrics: Dict[str, Dict]
) -> Dict:
    """
    Test contribution of each dual-lane criterion.

    Compares:
    - Full dual-lane (NPMI OR IG) - baseline
    - NPMI lane only
    - IG lane only
    - Both required (NPMI AND IG)

    Validates:
    - OR logic is appropriate (vs AND)
    - Relative contribution of each lane
    - Whether one lane dominates

    Args:
        metrics: Schema-level metrics from calculate_lc_metrics()

    Returns:
        Dict with results for each lane configuration

    Example:
        >>> results = ablation_test_dual_lane(lc_metrics)
        >>> print(f"NPMI only: {results['npmi_only']['schema_count']}")
        >>> print(f"IG only: {results['ig_only']['schema_count']}")
        NPMI only: 17
        IG only: 32
    """
    results = {}

    # Baseline: Full dual-lane (OR logic)
    print("  Testing full dual-lane (NPMI OR IG)...")
    full_accepted = {}
    for pattern, m in metrics.items():
        npmi_pass = m['npmi'] >= 0.05 and m['g_squared'] >= 3.84
        ig_pass = m['ig'] >= 0.01 and m['g_squared'] >= 1.0

        if npmi_pass or ig_pass:
            full_accepted[pattern] = m

    results['full_dual_lane'] = {
        'schema_count': len(full_accepted),
        'description': 'NPMI OR IG (both lanes, OR logic)'
    }

    # NPMI lane only
    print("  Testing NPMI lane only...")
    npmi_only = {
        p: m for p, m in metrics.items()
        if m['npmi'] >= 0.05 and m['g_squared'] >= 3.84
    }

    results['npmi_only'] = {
        'schema_count': len(npmi_only),
        'description': 'NPMI lane only (constructional association)',
        'change': len(npmi_only) - len(full_accepted)
    }

    # IG lane only
    print("  Testing IG lane only...")
    ig_only = {
        p: m for p, m in metrics.items()
        if m['ig'] >= 0.01 and m['g_squared'] >= 1.0
    }

    results['ig_only'] = {
        'schema_count': len(ig_only),
        'description': 'IG lane only (lexical productivity)',
        'change': len(ig_only) - len(full_accepted)
    }

    # AND logic (both lanes required)
    print("  Testing AND logic (both required)...")
    both_required = {
        p: m for p, m in metrics.items()
        if (m['npmi'] >= 0.05 and m['g_squared'] >= 3.84)
        and (m['ig'] >= 0.01 and m['g_squared'] >= 1.0)
    }

    results['both_required'] = {
        'schema_count': len(both_required),
        'description': 'NPMI AND IG (both required, AND logic)',
        'change': len(both_required) - len(full_accepted)
    }

    return results


def print_validation_report(
    permutation_results: Dict,
    ablation_filter_results: Dict,
    ablation_lane_results: Dict
):
    """
    Print comprehensive validation report.

    Displays:
    1. Permutation test results (statistical significance)
    2. Ablation test results (filter contributions)
    3. Dual-lane ablation results (lane contributions)

    Args:
        permutation_results: Output from permutation_test_tam_comp()
        ablation_filter_results: Output from ablation_test_filters()
        ablation_lane_results: Output from ablation_test_dual_lane()
    """
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION REPORT")
    print("="*70)

    # Permutation test
    print("\n📊 PERMUTATION TEST (TAM×COMP Associations)")
    print("-"*70)
    print(f"Null Hypothesis: TAM and COMP are independent (no association)")
    print(f"Alternative: TAM and COMP show constructional preferences")
    print()
    print(f"Real schemas: {permutation_results['real_schema_count']}")
    print(f"Permuted mean: {permutation_results['mean_permuted']:.1f} "
          f"(SD: {permutation_results['std_permuted']:.1f})")
    print(f"p-value: {permutation_results['p_value']:.4f}")
    print()

    if permutation_results['significant']:
        print("✅ SIGNIFICANT: Real associations are stronger than random (p < 0.05)")
        print("   Interpretation: TAM×COMP patterns are linguistically meaningful")
    else:
        print("❌ NOT SIGNIFICANT: Associations could be random (p ≥ 0.05)")
        print("   Interpretation: TAM×COMP patterns may be statistical accidents")

    # Ablation tests - Instance filters
    print("\n" + "="*70)
    print("🔬 ABLATION TEST: Instance Filter Contributions")
    print("-"*70)
    print("Question: Does each filter contribute meaningfully?")
    print()

    baseline = ablation_filter_results['full_pipeline']
    print(f"Baseline (all filters): {baseline['formulaic_count']:,} formulaic → "
          f"{baseline['schema_count']} schemas")
    print()

    for key in ['no_atp', 'no_dpb', 'no_hr']:
        result = ablation_filter_results[key]
        form_change = result.get('change_formulaic', 0)
        schema_change = result.get('change_schemas', 0)
        form_pct = (form_change / baseline['formulaic_count'] * 100) if baseline['formulaic_count'] > 0 else 0
        schema_pct = (schema_change / baseline['schema_count'] * 100) if baseline['schema_count'] > 0 else 0

        print(f"{result['description']}:")
        print(f"  Formulaic: {result['formulaic_count']:,} ({form_change:+,} / {form_pct:+.1f}%)")
        print(f"  Schemas: {result['schema_count']} ({schema_change:+d} / {schema_pct:+.1f}%)")

        # Interpretation
        if abs(form_change) < 50:
            print(f"  → Minor impact: filter has small effect")
        elif abs(form_change) < 200:
            print(f"  → Moderate impact: filter contributes meaningfully")
        else:
            print(f"  → Major impact: filter is critical for quality")
        print()

    # Ablation tests - Dual-lane
    print("="*70)
    print("🔬 ABLATION TEST: Dual-Lane Contributions")
    print("-"*70)
    print("Question: Which lane contributes more schemas?")
    print()

    for key in ['full_dual_lane', 'npmi_only', 'ig_only', 'both_required']:
        result = ablation_lane_results[key]
        change = result.get('change', 0)
        change_str = f" ({change:+d})" if 'change' in result else ""

        print(f"{result['description']}: {result['schema_count']} schemas{change_str}")

    print()

    # Interpretation
    npmi_count = ablation_lane_results['npmi_only']['schema_count']
    ig_count = ablation_lane_results['ig_only']['schema_count']
    full_count = ablation_lane_results['full_dual_lane']['schema_count']

    if ig_count > npmi_count * 1.5:
        print("✅ IG lane is PRIMARY validator (contributes most schemas)")
    elif npmi_count > ig_count * 1.5:
        print("✅ NPMI lane is PRIMARY validator (contributes most schemas)")
    else:
        print("✅ Both lanes contribute comparably (balanced validation)")

    if full_count > max(npmi_count, ig_count):
        print("✅ OR logic is appropriate (captures both types of formulaicity)")
    else:
        print("⚠️  OR logic may be too permissive (consider AND logic)")

    print("="*70)
