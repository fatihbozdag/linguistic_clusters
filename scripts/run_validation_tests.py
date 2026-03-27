#!/usr/bin/env python3
"""
Statistical Validation Tests for ConstructionMiner Pipeline

Runs validation tests on completed pipeline results:
1. Permutation testing (TAM×COMP associations)
2. Ablation testing (filter contributions)
3. Dual-lane ablation (lane contributions)

Usage:
    python run_validation_tests.py

Requirements:
    - Pipeline must be run first (run_full_corpus_analysis.py)
    - Results must exist in analysis_results/

Output:
    - Console: Validation report
    - File: validation_results.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Import modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from validation_tests import (
    permutation_test_tam_comp,
    ablation_test_filters,
    ablation_test_dual_lane,
    print_validation_report
)


def load_pipeline_results(results_dir: Path):
    """
    Load pipeline results from CSV files.

    Args:
        results_dir: Path to analysis_results directory

    Returns:
        Tuple of (valid_constructions, formulaic_instances, lc_metrics)
    """
    import csv

    print("Loading pipeline results...")

    # Load valid constructions
    constructions_file = results_dir / 'constructions_filtered.csv'
    if not constructions_file.exists():
        raise FileNotFoundError(
            f"Pipeline results not found: {constructions_file}\n"
            f"Please run run_full_corpus_analysis.py first!"
        )

    valid_constructions = []
    with open(constructions_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['instance_atp'] = float(row.get('instance_atp', 0))
            row['instance_dpb'] = float(row.get('instance_dpb', 0))
            row['instance_hr'] = float(row.get('instance_hr', float('inf')))
            valid_constructions.append(row)

    print(f"  ✓ Loaded {len(valid_constructions):,} valid constructions")

    # Load formulaic instances
    formulaic_file = results_dir / 'constructions_formulaic.csv'
    formulaic_instances = []
    with open(formulaic_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['instance_atp'] = float(row.get('instance_atp', 0))
            row['instance_dpb'] = float(row.get('instance_dpb', 0))
            row['instance_hr'] = float(row.get('instance_hr', float('inf')))
            formulaic_instances.append(row)

    print(f"  ✓ Loaded {len(formulaic_instances):,} formulaic instances")

    # Load LC metrics
    metrics_file = results_dir / 'patterns_all_metrics.csv'
    lc_metrics = {}
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = row['pattern']
            lc_metrics[pattern] = {
                'n_tokens': int(row['n_tokens']),
                'n_docs': int(row['n_docs']),
                'atp': float(row['atp']),
                'delta_p': float(row['delta_p']),
                'ig': float(row['ig']),
                'npmi': float(row['npmi']),
                'dispersion': float(row['dispersion']),
                'g_squared': float(row['g_squared']),
                'p_value': float(row['p_value'])
            }

    print(f"  ✓ Loaded {len(lc_metrics):,} patterns with metrics")

    return valid_constructions, formulaic_instances, lc_metrics


def main():
    """Run all validation tests."""
    # Configuration
    results_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean/analysis_results')

    print("="*70)
    print("CONSTRUCTIONMINER STATISTICAL VALIDATION TESTS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load pipeline results
    try:
        valid_constructions, formulaic_instances, lc_metrics = load_pipeline_results(results_dir)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    print()

    # Run permutation test
    print("="*70)
    print("TEST 1: Permutation Test (TAM×COMP Associations)")
    print("="*70)
    print("Question: Are TAM-COMP associations real or random?")
    print("Method: Shuffle TAM assignments 1,000 times, re-run dual-lane")
    print()

    perm_results = permutation_test_tam_comp(
        formulaic_instances,
        n_iterations=1000,
        random_seed=42
    )

    print()

    # Run ablation tests - instance filters
    print("="*70)
    print("TEST 2: Ablation Test (Instance Filters)")
    print("="*70)
    print("Question: Does each instance filter contribute meaningfully?")
    print("Method: Remove one filter at a time, measure impact")
    print()

    ablation_filter_results = ablation_test_filters(valid_constructions)

    print()

    # Run ablation tests - dual-lane
    print("="*70)
    print("TEST 3: Ablation Test (Dual-Lane)")
    print("="*70)
    print("Question: Which lane contributes more schemas?")
    print("Method: Test NPMI-only, IG-only, both, AND logic")
    print()

    ablation_lane_results = ablation_test_dual_lane(lc_metrics)

    # Print comprehensive report
    print_validation_report(
        perm_results,
        ablation_filter_results,
        ablation_lane_results
    )

    # Export validation results
    validation_file = results_dir / 'validation_results.json'
    with open(validation_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'permutation_test': {
                'real_schema_count': int(perm_results['real_schema_count']),
                'mean_permuted': float(perm_results['mean_permuted']),
                'std_permuted': float(perm_results['std_permuted']),
                'p_value': float(perm_results['p_value']),
                'significant': bool(perm_results['significant'])
            },
            'ablation_filters': ablation_filter_results,
            'ablation_dual_lane': ablation_lane_results
        }, f, indent=2)

    print(f"\n✓ Validation results saved to: {validation_file}")
    print()
    print("="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
