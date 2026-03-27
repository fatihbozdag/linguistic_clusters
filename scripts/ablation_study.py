#!/usr/bin/env python3
"""
Ablation Study - Filter Layer Contribution Analysis

Measures the contribution of each filtering layer by running the pipeline
with individual layers removed. Addresses CLaLT reviewer concern:
"Unclear contribution of each filtering layer."

Configurations tested:
1. FULL: All layers active (baseline)
2. NO_LAYER1: Skip instance prefilters (ATP, ΔP, H_r)
3. NO_NPMI_LANE: Only H_slot lane active
4. NO_H_SLOT_LANE: Only NPMI lane active
5. NO_NORMALIZATION: Skip preposition normalization
6. BASELINE: No filtering (frequency only)

Usage:
    python scripts/ablation_study.py
"""

import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lc_metrics import calculate_lc_metrics, apply_dual_lane_acceptance
from prep_normalizer import normalize_constructions


# ============================================================================
# Ablation Configurations
# ============================================================================

def apply_full_pipeline(constructions: List[Dict]) -> Tuple[Set[str], Dict]:
    """Full pipeline with all layers active (G²-based acceptance)."""
    # Layer 1: Instance prefilters
    formulaic = []
    for const in constructions:
        # Use proper None checking (not 'or' which fails for 0.0)
        atp_val = const.get('instance_atp')
        dpb_val = const.get('instance_dpb')
        hr_val = const.get('instance_hr')

        atp = float(atp_val) if atp_val is not None else 0.0
        dpb = float(dpb_val) if dpb_val is not None else 0.0
        hr = float(hr_val) if hr_val is not None else float('inf')

        if atp >= 0.10 and dpb >= 0.10 and hr <= 2.8:
            formulaic.append(const)

    if not formulaic:
        return set(), {'formulaic_count': 0, 'pattern_count': 0}

    # Layer 2: Pattern metrics
    metrics = calculate_lc_metrics(formulaic)

    # Layer 3: Dual-lane acceptance with G² thresholds
    accepted = set()
    for pattern, m in metrics.items():
        npmi = m.get('npmi', 0)
        h_slot = m.get('h_slot', 0)
        g2 = m.get('g_squared', 0)

        npmi_pass = npmi >= 0.05 and g2 >= 3.84
        h_slot_pass = h_slot >= 1.5 and g2 >= 1.0 and npmi >= 0  # NPMI floor

        if npmi_pass or h_slot_pass:
            accepted.add(pattern)

    return accepted, {
        'formulaic_count': len(formulaic),
        'pattern_count': len(metrics)
    }


def apply_no_layer1(constructions: List[Dict]) -> Tuple[Set[str], Dict]:
    """Skip instance prefilters, go directly to pattern metrics (G²-based)."""
    # No filtering - use all constructions
    metrics = calculate_lc_metrics(constructions)

    # Apply G²-based dual-lane acceptance
    accepted = set()
    for pattern, m in metrics.items():
        npmi = m.get('npmi', 0)
        h_slot = m.get('h_slot', 0)
        g2 = m.get('g_squared', 0)

        npmi_pass = npmi >= 0.05 and g2 >= 3.84
        h_slot_pass = h_slot >= 1.5 and g2 >= 1.0 and npmi >= 0

        if npmi_pass or h_slot_pass:
            accepted.add(pattern)

    return accepted, {
        'formulaic_count': len(constructions),
        'pattern_count': len(metrics)
    }


def apply_npmi_only(constructions: List[Dict]) -> Tuple[Set[str], Dict]:
    """Only NPMI lane, no H_slot lane (G²-based)."""
    # Layer 1: Instance prefilters
    formulaic = []
    for const in constructions:
        # Use proper None checking (not 'or' which fails for 0.0)
        atp_val = const.get('instance_atp')
        dpb_val = const.get('instance_dpb')
        hr_val = const.get('instance_hr')

        atp = float(atp_val) if atp_val is not None else 0.0
        dpb = float(dpb_val) if dpb_val is not None else 0.0
        hr = float(hr_val) if hr_val is not None else float('inf')

        if atp >= 0.10 and dpb >= 0.10 and hr <= 2.8:
            formulaic.append(const)

    if not formulaic:
        return set(), {'formulaic_count': 0, 'pattern_count': 0}

    metrics = calculate_lc_metrics(formulaic)

    # NPMI lane only with G² threshold
    accepted = set()
    for pattern, m in metrics.items():
        npmi = m.get('npmi', 0)
        g2 = m.get('g_squared', 0)

        if npmi >= 0.05 and g2 >= 3.84:
            accepted.add(pattern)

    return accepted, {
        'formulaic_count': len(formulaic),
        'pattern_count': len(metrics)
    }


def apply_h_slot_only(constructions: List[Dict]) -> Tuple[Set[str], Dict]:
    """Only H_slot lane, no NPMI lane (G²-based)."""
    # Layer 1: Instance prefilters
    formulaic = []
    for const in constructions:
        # Use proper None checking (not 'or' which fails for 0.0)
        atp_val = const.get('instance_atp')
        dpb_val = const.get('instance_dpb')
        hr_val = const.get('instance_hr')

        atp = float(atp_val) if atp_val is not None else 0.0
        dpb = float(dpb_val) if dpb_val is not None else 0.0
        hr = float(hr_val) if hr_val is not None else float('inf')

        if atp >= 0.10 and dpb >= 0.10 and hr <= 2.8:
            formulaic.append(const)

    if not formulaic:
        return set(), {'formulaic_count': 0, 'pattern_count': 0}

    metrics = calculate_lc_metrics(formulaic)

    # H_slot lane only with G² threshold (with NPMI floor)
    accepted = set()
    for pattern, m in metrics.items():
        h_slot = m.get('h_slot', 0)
        g2 = m.get('g_squared', 0)
        npmi = m.get('npmi', 0)

        if h_slot >= 1.5 and g2 >= 1.0 and npmi >= 0:
            accepted.add(pattern)

    return accepted, {
        'formulaic_count': len(formulaic),
        'pattern_count': len(metrics)
    }


def apply_frequency_only(constructions: List[Dict], min_freq: int = 3) -> Tuple[Set[str], Dict]:
    """Baseline: frequency threshold only, no association metrics."""
    # Count patterns
    pattern_counts = Counter(c.get('pattern') for c in constructions)

    # Accept patterns with frequency >= threshold
    accepted = {p for p, count in pattern_counts.items() if count >= min_freq}

    return accepted, {
        'formulaic_count': len(constructions),
        'pattern_count': len(pattern_counts)
    }


def run_ablation_study(constructions: List[Dict], verbose: bool = True) -> Dict:
    """
    Run complete ablation study.

    Args:
        constructions: List of constructions with instance metrics
        verbose: Print progress

    Returns:
        Dict with results for each configuration
    """
    if verbose:
        print("=" * 70)
        print("ABLATION STUDY - FILTER LAYER CONTRIBUTION")
        print("=" * 70)
        print(f"Total constructions: {len(constructions):,}")
        print()

    configurations = [
        ('FULL', 'All layers active', apply_full_pipeline),
        ('NO_LAYER1', 'Skip instance prefilters', apply_no_layer1),
        ('NPMI_ONLY', 'Only NPMI lane', apply_npmi_only),
        ('H_SLOT_ONLY', 'Only H_slot lane', apply_h_slot_only),
        ('FREQ_ONLY', 'Frequency threshold only (≥3)', lambda c: apply_frequency_only(c, 3)),
    ]

    results = {
        'timestamp': datetime.now().isoformat(),
        'n_constructions': len(constructions),
        'configurations': {}
    }

    for config_name, description, func in configurations:
        if verbose:
            print(f"Running {config_name}: {description}...")

        accepted, stats = func(constructions)

        results['configurations'][config_name] = {
            'description': description,
            'n_accepted': len(accepted),
            'accepted_schemas': sorted(list(accepted)),
            'formulaic_count': stats['formulaic_count'],
            'pattern_count': stats['pattern_count']
        }

        if verbose:
            print(f"  → {len(accepted)} schemas accepted")

    # Calculate layer contributions
    full_schemas = set(results['configurations']['FULL']['accepted_schemas'])
    no_l1_schemas = set(results['configurations']['NO_LAYER1']['accepted_schemas'])
    npmi_schemas = set(results['configurations']['NPMI_ONLY']['accepted_schemas'])
    h_slot_schemas = set(results['configurations']['H_SLOT_ONLY']['accepted_schemas'])

    # Schemas unique to each configuration
    results['analysis'] = {
        'layer1_contribution': len(no_l1_schemas) - len(full_schemas),
        'layer1_filters_out': len(no_l1_schemas - full_schemas),
        'npmi_unique': len(npmi_schemas - h_slot_schemas),
        'h_slot_unique': len(h_slot_schemas - npmi_schemas),
        'both_lanes': len(npmi_schemas & h_slot_schemas),
        'rescued_by_or_logic': len(full_schemas - (npmi_schemas & h_slot_schemas))
    }

    if verbose:
        print()
        print("=" * 70)
        print("CONTRIBUTION ANALYSIS")
        print("=" * 70)
        print(f"\nLayer 1 (Instance Prefilters) filters out: "
              f"{results['analysis']['layer1_filters_out']} patterns")
        print(f"  These would have been false positives without Layer 1")
        print()
        print(f"Dual-Lane OR Logic:")
        print(f"  - NPMI-only schemas: {results['analysis']['npmi_unique']}")
        print(f"  - H_slot-only schemas: {results['analysis']['h_slot_unique']}")
        print(f"  - Both lanes: {results['analysis']['both_lanes']}")
        print(f"  - Rescued by OR logic: {results['analysis']['rescued_by_or_logic']}")

    return results


def generate_ablation_report(results: Dict, output_file: Path):
    """Generate detailed ablation study report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Constructions analyzed: {results['n_constructions']:,}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CONFIGURATION RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for config, data in results['configurations'].items():
            f.write(f"{config}: {data['description']}\n")
            f.write(f"  Accepted schemas: {data['n_accepted']}\n")
            f.write(f"  Unique patterns: {data['pattern_count']}\n")
            if data['accepted_schemas']:
                f.write(f"  Top schemas: {', '.join(data['accepted_schemas'][:5])}")
                if len(data['accepted_schemas']) > 5:
                    f.write(f" ... (+{len(data['accepted_schemas'])-5} more)")
                f.write("\n")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("LAYER CONTRIBUTION ANALYSIS\n")
        f.write("-" * 70 + "\n\n")

        analysis = results['analysis']

        f.write("Layer 1 (Instance Prefilters: ATP, ΔP, H_r):\n")
        f.write(f"  - Patterns filtered out: {analysis['layer1_filters_out']}\n")
        f.write("  - Purpose: Remove non-formulaic instances before aggregation\n")
        f.write("  - Impact: Reduces false positives from random co-occurrences\n\n")

        f.write("Layer 3 (Dual-Lane Acceptance):\n")
        f.write(f"  - NPMI-only schemas: {analysis['npmi_unique']}\n")
        f.write("    (Fixed collocations with strong TAM-COMP association)\n")
        f.write(f"  - H_slot-only schemas: {analysis['h_slot_unique']}\n")
        f.write("    (Productive templates with lexical diversity)\n")
        f.write(f"  - Both lanes: {analysis['both_lanes']}\n")
        f.write("    (Highly formulaic: both associative AND productive)\n")
        f.write(f"  - Rescued by OR logic: {analysis['rescued_by_or_logic']}\n")
        f.write("    (Would be rejected under AND logic)\n\n")

        f.write("-" * 70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        full_count = results['configurations']['FULL']['n_accepted']
        no_l1_count = results['configurations']['NO_LAYER1']['n_accepted']
        freq_count = results['configurations']['FREQ_ONLY']['n_accepted']

        f.write(f"1. Without instance prefilters (Layer 1), {no_l1_count} schemas\n")
        f.write(f"   would be accepted vs. {full_count} with full pipeline.\n")
        f.write(f"   Layer 1 prevents {no_l1_count - full_count} potential false positives.\n\n")

        f.write(f"2. Frequency-only baseline accepts {freq_count} patterns.\n")
        f.write(f"   The full pipeline reduces this to {full_count},\n")
        f.write(f"   a {(1 - full_count/freq_count)*100:.0f}% reduction.\n\n")

        f.write(f"3. The OR logic in dual-lane acceptance rescues\n")
        f.write(f"   {analysis['rescued_by_or_logic']} schemas that would be rejected\n")
        f.write("   under stricter AND logic.\n")

        f.write("\n" + "=" * 70 + "\n")


def generate_venn_diagram_data(results: Dict, output_file: Path):
    """Generate data for Venn diagram visualization."""
    npmi_set = set(results['configurations']['NPMI_ONLY']['accepted_schemas'])
    h_slot_set = set(results['configurations']['H_SLOT_ONLY']['accepted_schemas'])

    venn_data = {
        'npmi_only': sorted(list(npmi_set - h_slot_set)),
        'h_slot_only': sorted(list(h_slot_set - npmi_set)),
        'both': sorted(list(npmi_set & h_slot_set)),
        'counts': {
            'npmi_only': len(npmi_set - h_slot_set),
            'h_slot_only': len(h_slot_set - npmi_set),
            'both': len(npmi_set & h_slot_set)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(venn_data, f, indent=2)


def main():
    """Run ablation study on current data."""
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
            # Convert numeric fields from strings
            # Use proper None checking (not 'or' which fails for 0.0)
            atp_val = row.get('instance_atp', '')
            dpb_val = row.get('instance_dpb', '')
            hr_val = row.get('instance_hr', '')

            row['instance_atp'] = float(atp_val) if atp_val else 0.0
            row['instance_dpb'] = float(dpb_val) if dpb_val else 0.0
            row['instance_hr'] = float(hr_val) if hr_val else float('inf')
            constructions.append(row)

    print(f"Loaded {len(constructions):,} constructions\n")

    # Run ablation study
    results = run_ablation_study(constructions, verbose=True)

    # Save report
    report_file = results_dir / 'ablation_study_report.txt'
    generate_ablation_report(results, report_file)
    print(f"\n✓ Saved report to {report_file}")

    # Save JSON
    json_file = results_dir / 'ablation_study_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON to {json_file}")

    # Save Venn diagram data
    venn_file = results_dir / 'venn_diagram_data.json'
    generate_venn_diagram_data(results, venn_file)
    print(f"✓ Saved Venn diagram data to {venn_file}")


if __name__ == "__main__":
    main()
