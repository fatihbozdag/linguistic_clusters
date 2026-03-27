#!/usr/bin/env python3
"""
Sensitivity Analysis for Threshold Parameters

Tests the stability of schema acceptance across different threshold values.
Addresses CLaLT reviewer concern: "Thresholds appear arbitrary. No sensitivity
analysis or empirical justification is provided."

This script:
1. Varies each threshold parameter while holding others constant
2. Records which schemas are accepted at each threshold level
3. Identifies "stable" schemas that appear across multiple thresholds
4. Generates visualization and report

Usage:
    python scripts/sensitivity_analysis.py
"""

import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from datetime import datetime
from collections import defaultdict
import itertools

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lc_metrics import (
    calculate_lc_metrics,
    apply_dual_lane_acceptance,
    build_corpus_statistics,
    calculate_instance_atp,
    calculate_instance_delta_p_backward,
    calculate_instance_boundary_entropy
)


# ============================================================================
# Parameter Grids
# ============================================================================

# Instance-level prefilter thresholds (Layer 1)
ATP_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]
DPB_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]
HR_THRESHOLDS = [2.0, 2.5, 2.8, 3.0, 3.5]

# Schema-level thresholds (Layer 3 - Dual Lane)
NPMI_THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10]
H_SLOT_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0]  # Slot entropy thresholds
G2_THRESHOLDS = [1.0, 2.0, 3.84, 5.0, 7.0]

# Default values (current production settings)
DEFAULTS = {
    'atp': 0.10,
    'dpb': 0.10,
    'hr': 2.8,
    'npmi': 0.05,
    'h_slot': 1.5,  # Slot entropy threshold
    'g2_npmi': 3.84,
    'g2_h_slot': 1.0
}


def apply_instance_prefilters(
    constructions: List[Dict],
    atp_thresh: float = 0.10,
    dpb_thresh: float = 0.10,
    hr_thresh: float = 2.8
) -> List[Dict]:
    """Apply instance-level prefilters with custom thresholds."""
    formulaic = []
    for const in constructions:
        atp = const.get('instance_atp', 0.0)
        dpb = const.get('instance_dpb', 0.0)
        hr = const.get('instance_hr', float('inf'))

        if atp >= atp_thresh and dpb >= dpb_thresh and hr <= hr_thresh:
            formulaic.append(const)

    return formulaic


def apply_dual_lane_g2(
    metrics: Dict[str, Dict],
    npmi_thresh: float = 0.05,
    h_slot_thresh: float = 1.5,
    g2_npmi_thresh: float = 3.84,
    g2_h_slot_thresh: float = 1.0
) -> Set[str]:
    """
    Apply dual-lane acceptance with G²-based significance.

    Uses G² thresholds rather than FDR correction because:
    - TAM×COMP patterns are not independent (shared structural dependencies)
    - FDR assumes independent tests, which over-penalizes
    - Permutation testing validates schemas more appropriately

    Args:
        metrics: Dict mapping pattern → metrics dict
        npmi_thresh: Minimum NPMI for Lane 1
        h_slot_thresh: Minimum H_slot (Slot Entropy) for Lane 2
        g2_npmi_thresh: G² threshold for NPMI lane (3.84 = p<0.05)
        g2_h_slot_thresh: G² threshold for H_slot lane (1.0, more lenient)

    Returns:
        Set of accepted pattern names
    """
    accepted = set()

    for pattern, m in metrics.items():
        npmi = m.get('npmi', 0)
        h_slot = m.get('h_slot', 0)
        g2 = m.get('g_squared', 0)

        # Lane 1: NPMI (Constructional Association)
        npmi_pass = npmi >= npmi_thresh and g2 >= g2_npmi_thresh

        # Lane 2: H_slot (Lexical Productivity) with NPMI floor
        h_slot_pass = h_slot >= h_slot_thresh and g2 >= g2_h_slot_thresh and npmi >= 0

        # OR logic
        if npmi_pass or h_slot_pass:
            accepted.add(pattern)

    return accepted


def run_single_parameter_sensitivity(
    constructions: List[Dict],
    parameter: str,
    values: List[float],
    verbose: bool = True
) -> Dict:
    """
    Run sensitivity analysis for a single parameter.

    Args:
        constructions: List of constructions with instance metrics
        parameter: Parameter name ('atp', 'dpb', 'hr', 'npmi', 'h_slot')
        values: List of threshold values to test
        verbose: Print progress

    Returns:
        Dict with results for each threshold value
    """
    results = {
        'parameter': parameter,
        'values': values,
        'n_accepted': [],
        'schemas_accepted': [],
        'stable_schemas': set()
    }

    # Determine if this is instance-level or schema-level parameter
    instance_params = {'atp', 'dpb', 'hr'}

    if verbose:
        print(f"\nTesting {parameter} thresholds: {values}")

    for thresh in values:
        # Set thresholds
        if parameter in instance_params:
            # Instance-level: vary one, keep others at default
            atp_t = thresh if parameter == 'atp' else DEFAULTS['atp']
            dpb_t = thresh if parameter == 'dpb' else DEFAULTS['dpb']
            hr_t = thresh if parameter == 'hr' else DEFAULTS['hr']

            formulaic = apply_instance_prefilters(
                constructions, atp_t, dpb_t, hr_t
            )

            if len(formulaic) == 0:
                results['n_accepted'].append(0)
                results['schemas_accepted'].append(set())
                continue

            # Calculate schema metrics
            metrics = calculate_lc_metrics(formulaic)

            # Apply G²-based dual-lane acceptance
            accepted = apply_dual_lane_g2(
                metrics,
                DEFAULTS['npmi'],
                DEFAULTS['h_slot'],
                DEFAULTS['g2_npmi'],
                DEFAULTS['g2_h_slot']
            )
        else:
            # Schema-level: use default instance filters
            formulaic = apply_instance_prefilters(
                constructions,
                DEFAULTS['atp'],
                DEFAULTS['dpb'],
                DEFAULTS['hr']
            )

            if len(formulaic) == 0:
                results['n_accepted'].append(0)
                results['schemas_accepted'].append(set())
                continue

            metrics = calculate_lc_metrics(formulaic)

            # Vary schema-level threshold
            npmi_t = thresh if parameter == 'npmi' else DEFAULTS['npmi']
            h_slot_t = thresh if parameter == 'h_slot' else DEFAULTS['h_slot']

            accepted = apply_dual_lane_g2(
                metrics, npmi_t, h_slot_t,
                DEFAULTS['g2_npmi'], DEFAULTS['g2_h_slot']
            )

        results['n_accepted'].append(len(accepted))
        results['schemas_accepted'].append(accepted)

        if verbose:
            print(f"  {parameter}={thresh}: {len(accepted)} schemas")

    # Find stable schemas (appear in majority of threshold settings)
    all_schemas = set()
    for schemas in results['schemas_accepted']:
        all_schemas.update(schemas)

    stability_threshold = len(values) // 2 + 1  # Majority
    for schema in all_schemas:
        count = sum(1 for schemas in results['schemas_accepted'] if schema in schemas)
        if count >= stability_threshold:
            results['stable_schemas'].add(schema)

    return results


def run_full_sensitivity_analysis(
    constructions: List[Dict],
    verbose: bool = True
) -> Dict:
    """
    Run complete sensitivity analysis across all parameters.

    Args:
        constructions: List of constructions with instance metrics
        verbose: Print progress

    Returns:
        Dict with results for all parameters
    """
    if verbose:
        print("=" * 70)
        print("SENSITIVITY ANALYSIS - THRESHOLD STABILITY")
        print("=" * 70)
        print(f"Total constructions: {len(constructions):,}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'n_constructions': len(constructions),
        'defaults': DEFAULTS,
        'parameters': {}
    }

    # Test each parameter
    param_configs = [
        ('atp', ATP_THRESHOLDS),
        ('dpb', DPB_THRESHOLDS),
        ('hr', HR_THRESHOLDS),
        ('npmi', NPMI_THRESHOLDS),
        ('h_slot', H_SLOT_THRESHOLDS),
    ]

    all_stable = set()
    for param, values in param_configs:
        param_results = run_single_parameter_sensitivity(
            constructions, param, values, verbose
        )
        results['parameters'][param] = {
            'values': values,
            'n_accepted': param_results['n_accepted'],
            'stable_schemas': list(param_results['stable_schemas'])
        }
        all_stable.update(param_results['stable_schemas'])

    # Find schemas stable across ALL parameters
    results['globally_stable_schemas'] = []
    for schema in all_stable:
        stable_count = 0
        for param in results['parameters']:
            if schema in results['parameters'][param]['stable_schemas']:
                stable_count += 1
        if stable_count == len(param_configs):
            results['globally_stable_schemas'].append(schema)

    if verbose:
        print()
        print("=" * 70)
        print("STABILITY SUMMARY")
        print("=" * 70)
        print(f"\nGlobally stable schemas (stable across all parameters):")
        for schema in sorted(results['globally_stable_schemas']):
            print(f"  ✓ {schema}")
        print(f"\nTotal: {len(results['globally_stable_schemas'])} globally stable schemas")

    return results


def generate_sensitivity_report(results: Dict, output_file: Path):
    """Generate detailed sensitivity analysis report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SENSITIVITY ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Constructions analyzed: {results['n_constructions']:,}\n\n")

        f.write("-" * 70 + "\n")
        f.write("DEFAULT THRESHOLDS\n")
        f.write("-" * 70 + "\n")
        for param, value in results['defaults'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

        for param, data in results['parameters'].items():
            f.write("-" * 70 + "\n")
            f.write(f"PARAMETER: {param.upper()}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Tested values: {data['values']}\n")
            f.write(f"Accepted schemas: {data['n_accepted']}\n")
            f.write(f"Stable schemas: {len(data['stable_schemas'])}\n")
            if data['stable_schemas']:
                f.write("  " + ", ".join(sorted(data['stable_schemas'])[:10]))
                if len(data['stable_schemas']) > 10:
                    f.write(f" ... and {len(data['stable_schemas'])-10} more")
                f.write("\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("GLOBALLY STABLE SCHEMAS\n")
        f.write("=" * 70 + "\n")
        f.write("(Schemas that remain accepted across majority of threshold settings\n")
        f.write(" for ALL parameters)\n\n")

        if results['globally_stable_schemas']:
            for schema in sorted(results['globally_stable_schemas']):
                f.write(f"  ✓ {schema}\n")
            f.write(f"\nTotal: {len(results['globally_stable_schemas'])} globally stable schemas\n")
        else:
            f.write("  No globally stable schemas found.\n")
            f.write("  This may indicate threshold sensitivity.\n")

        f.write("\n" + "=" * 70 + "\n")


def main():
    """Run sensitivity analysis on current data."""
    # Load filtered constructions
    results_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean/analysis_results')
    filtered_csv = results_dir / 'constructions_filtered.csv'

    if not filtered_csv.exists():
        print(f"Error: {filtered_csv} not found.")
        print("Please run the full pipeline first.")
        sys.exit(1)

    print(f"Loading constructions from {filtered_csv}...")

    constructions = []
    with open(filtered_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['instance_atp'] = float(row.get('instance_atp', 0) or 0)
            row['instance_dpb'] = float(row.get('instance_dpb', 0) or 0)
            row['instance_hr'] = float(row.get('instance_hr', 999) or 999)
            constructions.append(row)

    print(f"Loaded {len(constructions):,} constructions\n")

    # Run analysis
    results = run_full_sensitivity_analysis(constructions, verbose=True)

    # Save report
    report_file = results_dir / 'sensitivity_analysis_report.txt'
    generate_sensitivity_report(results, report_file)
    print(f"\n✓ Saved report to {report_file}")

    # Save JSON
    json_file = results_dir / 'sensitivity_analysis_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON to {json_file}")


if __name__ == "__main__":
    main()
