#!/usr/bin/env python3
"""
Complete LC Framework Pipeline - Fresh Run with Full Validation

Generates all results needed for manuscript preparation:
1. Complete filtering cascade
2. Permutation testing (TAM-shuffling + Verb-shuffling)
3. Sensitivity analysis (OFAT methodology)
4. Ablation analysis (with vs without Layer 1)
5. Comprehensive JSON report and SUMMARY.md

Configuration (FINAL VALIDATED):
- Layer 1: ATP ≥ 0.10, ΔP_backward ≥ 0.10, H_r ≤ 2.8
- Layer 2 NPMI Lane: NPMI ≥ 0.05, G² ≥ 3.84
- Layer 2 H_slot Lane: H_slot ≥ 1.5, G² ≥ 1.0, NPMI ≥ 0
"""

import sys
import csv
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lc_metrics import calculate_lc_metrics
from passive_filter import filter_constructions, print_filter_statistics

# =============================================================================
# CONFIGURATION - FINAL VALIDATED THRESHOLDS
# =============================================================================

CONFIG = {
    # Layer 1: Instance-level prefilters
    'atp_threshold': 0.10,
    'dpb_threshold': 0.10,
    'hr_threshold': 2.8,

    # Layer 2: Dual-lane acceptance
    'npmi_threshold': 0.05,
    'h_slot_threshold': 1.5,
    'g2_npmi_threshold': 3.84,  # chi2(1) at p < 0.05
    'g2_h_slot_threshold': 1.0,
    'npmi_floor': 0.0,  # For H_slot lane

    # Validation settings
    'permutation_iterations': 1000,
    'sensitivity_values': 5,  # Values per parameter
}

# Sensitivity parameter grids
SENSITIVITY_GRIDS = {
    'atp': [0.05, 0.08, 0.10, 0.12, 0.15],
    'dpb': [0.05, 0.08, 0.10, 0.12, 0.15],
    'hr': [2.0, 2.5, 2.8, 3.0, 3.5],
    'npmi': [0.02, 0.03, 0.05, 0.08, 0.10],
    'h_slot': [1.0, 1.25, 1.5, 1.75, 2.0],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log(msg: str):
    """Log with timestamp."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def apply_instance_prefilters(constructions: List[Dict],
                               atp_t: float, dpb_t: float, hr_t: float) -> List[Dict]:
    """Apply Layer 1 instance-level prefilters."""
    return [c for c in constructions
            if c.get('instance_atp', 0) >= atp_t
            and c.get('instance_dpb', 0) >= dpb_t
            and c.get('instance_hr', float('inf')) <= hr_t]


def apply_dual_lane_g2(metrics: Dict[str, Dict],
                        npmi_t: float = 0.05,
                        h_slot_t: float = 1.5,
                        g2_npmi_t: float = 3.84,
                        g2_h_slot_t: float = 1.0,
                        npmi_floor: float = 0.0) -> Tuple[Set[str], Dict]:
    """
    Apply dual-lane acceptance with G²-based significance.
    Returns: (accepted_schemas, lane_info)
    """
    accepted = set()
    lane_info = {}

    for pattern, m in metrics.items():
        npmi = m.get('npmi', 0)
        h_slot = m.get('h_slot', 0)
        g2 = m.get('g_squared', 0)

        # Lane 1: NPMI (Constructional Association)
        npmi_pass = npmi >= npmi_t and g2 >= g2_npmi_t

        # Lane 2: H_slot (Lexical Productivity) with NPMI floor
        h_slot_pass = h_slot >= h_slot_t and g2 >= g2_h_slot_t and npmi >= npmi_floor

        if npmi_pass or h_slot_pass:
            accepted.add(pattern)
            if npmi_pass and h_slot_pass:
                lane_info[pattern] = 'Both'
            elif npmi_pass:
                lane_info[pattern] = 'NPMI'
            else:
                lane_info[pattern] = 'H_slot'

    return accepted, lane_info


def calculate_jaccard(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# =============================================================================
# STEP 2: COMPLETE PIPELINE
# =============================================================================

def run_complete_pipeline(asc_dir: Path, output_dir: Path) -> Dict:
    """Run the complete LC Framework pipeline."""
    log("="*70)
    log("STEP 2: COMPLETE LC FRAMEWORK PIPELINE")
    log("="*70)

    # Import passive extractor
    from passive_extractor import PassiveExtractor

    # Load and extract constructions
    log("Loading ASC files and extracting constructions...")
    extractor = PassiveExtractor()

    asc_files = sorted(asc_dir.glob('*_ASCinfo.txt'))
    log(f"Found {len(asc_files)} ASC files")

    all_constructions = []
    for i, asc_file in enumerate(asc_files):
        constructions = extractor.extract_from_file(str(asc_file))
        all_constructions.extend(constructions)
        if (i + 1) % 1000 == 0:
            log(f"  Processed {i+1}/{len(asc_files)} files...")

    total_raw = len(all_constructions)
    log(f"Total constructions extracted: {total_raw:,}")

    # Basic filtering using proper filter_constructions from passive_filter.py
    log("Applying basic filters (TAM validation, extraposed it-passives, etc.)...")
    valid_constructions, filter_stats = filter_constructions(all_constructions)

    total_valid = len(valid_constructions)
    log(f"Valid constructions (after basic filters): {total_valid:,} ({total_valid/total_raw*100:.1f}%)")
    log(f"  - Invalid TAM: {filter_stats.get('filtered_tam', 0):,}")
    log(f"  - Extraposed it-passives: {filter_stats.get('filtered_extraposed', 0):,}")

    # Calculate instance-level metrics
    log("Calculating instance-level metrics...")
    from lc_metrics import build_corpus_statistics, calculate_instance_atp, \
                          calculate_instance_delta_p_backward, calculate_instance_boundary_entropy

    corpus_stats = build_corpus_statistics(valid_constructions)

    for const in valid_constructions:
        const['instance_atp'] = calculate_instance_atp(const, corpus_stats)
        const['instance_dpb'] = calculate_instance_delta_p_backward(const, corpus_stats)
        const['instance_hr'] = calculate_instance_boundary_entropy(const, corpus_stats)

    # Apply Layer 1 prefilters
    log("Applying Layer 1 prefilters...")
    formulaic = apply_instance_prefilters(
        valid_constructions,
        CONFIG['atp_threshold'],
        CONFIG['dpb_threshold'],
        CONFIG['hr_threshold']
    )

    total_formulaic = len(formulaic)
    log(f"Formulaic instances (Layer 1): {total_formulaic:,} ({total_formulaic/total_valid*100:.2f}%)")

    # Calculate schema-level metrics
    log("Calculating schema-level metrics...")
    metrics = calculate_lc_metrics(formulaic)
    unique_patterns = len(metrics)
    log(f"Unique TAM×COMP patterns: {unique_patterns}")

    # Apply dual-lane acceptance
    log("Applying Layer 2 dual-lane acceptance...")
    accepted, lane_info = apply_dual_lane_g2(
        metrics,
        CONFIG['npmi_threshold'],
        CONFIG['h_slot_threshold'],
        CONFIG['g2_npmi_threshold'],
        CONFIG['g2_h_slot_threshold'],
        CONFIG['npmi_floor']
    )

    log(f"Accepted schemas: {len(accepted)} ({len(accepted)/unique_patterns*100:.1f}%)")

    # Build schema inventory
    schema_inventory = []
    for pattern in sorted(accepted):
        m = metrics[pattern]
        schema_inventory.append({
            'schema': pattern,
            'n': m.get('n_tokens', 0),
            'n_docs': m.get('n_docs', 0),
            'npmi': round(m.get('npmi', 0), 4),
            'h_slot': round(m.get('h_slot', 0), 4),
            'g_squared': round(m.get('g_squared', 0), 2),
            'atp': round(m.get('atp', 0), 4),
            'delta_p': round(m.get('delta_p', 0), 4),
            'lane': lane_info.get(pattern, 'Unknown')
        })

    # Sort by n (descending)
    schema_inventory.sort(key=lambda x: -x['n'])

    # Save intermediate results
    log("Saving intermediate results...")

    # Save filtered constructions
    filtered_csv = output_dir / 'constructions_filtered.csv'
    if valid_constructions:
        fieldnames = list(valid_constructions[0].keys())
        with open(filtered_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_constructions)

    # Save formulaic instances
    formulaic_csv = output_dir / 'constructions_formulaic.csv'
    if formulaic:
        fieldnames = list(formulaic[0].keys())
        with open(formulaic_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formulaic)

    # Save all patterns with metrics
    patterns_csv = output_dir / 'patterns_all_metrics.csv'
    patterns_data = []
    for pattern, m in metrics.items():
        patterns_data.append({
            'pattern': pattern,
            'n': m.get('n_tokens', 0),
            'n_docs': m.get('n_docs', 0),
            'npmi': m.get('npmi', 0),
            'h_slot': m.get('h_slot', 0),
            'g_squared': m.get('g_squared', 0),
            'atp': m.get('atp', 0),
            'delta_p': m.get('delta_p', 0),
            'accepted': pattern in accepted,
            'lane': lane_info.get(pattern, '')
        })
    with open(patterns_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(patterns_data[0].keys()))
        writer.writeheader()
        writer.writerows(patterns_data)

    # Save accepted schemas
    schemas_csv = output_dir / 'schemas_accepted.csv'
    with open(schemas_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(schema_inventory[0].keys()))
        writer.writeheader()
        writer.writerows(schema_inventory)

    results = {
        'filtering_cascade': {
            'total_raw': total_raw,
            'total_valid': total_valid,
            'valid_percentage': round(total_valid/total_raw*100, 2),
            'filtered_tam': filter_stats.get('filtered_tam', 0),
            'filtered_extraposed': filter_stats.get('filtered_extraposed', 0),
            'formulaic_instances': total_formulaic,
            'formulaic_percentage': round(total_formulaic/total_valid*100, 2),
            'unique_patterns': unique_patterns,
            'accepted_schemas': len(accepted),
            'acceptance_rate': round(len(accepted)/unique_patterns*100, 2)
        },
        'schema_inventory': schema_inventory,
        'metrics': metrics,
        'formulaic': formulaic,
        'valid_constructions': valid_constructions,
        'accepted_set': accepted
    }

    return results


# =============================================================================
# STEP 3: PERMUTATION TESTING
# =============================================================================

def run_tam_shuffling(formulaic: List[Dict], observed_schemas: Set[str],
                      n_iter: int = 1000) -> Dict:
    """TAM-shuffling permutation test."""
    log("\n--- TAM-Shuffling Permutation Test ---")

    jaccard_values = []
    schema_counts = []

    for i in range(n_iter):
        # Shuffle TAM values
        tams = [c['tam'] for c in formulaic]
        random.shuffle(tams)

        # Create permuted constructions
        permuted = []
        for j, c in enumerate(formulaic):
            pc = c.copy()
            pc['tam'] = tams[j]
            pc['pattern'] = f"{tams[j]},{c['comp']}"
            permuted.append(pc)

        # Calculate metrics and apply dual-lane
        metrics = calculate_lc_metrics(permuted)
        accepted, _ = apply_dual_lane_g2(
            metrics,
            CONFIG['npmi_threshold'],
            CONFIG['h_slot_threshold'],
            CONFIG['g2_npmi_threshold'],
            CONFIG['g2_h_slot_threshold'],
            CONFIG['npmi_floor']
        )

        schema_counts.append(len(accepted))
        jaccard_values.append(calculate_jaccard(observed_schemas, accepted))

        if (i + 1) % 100 == 0:
            log(f"  TAM-shuffling: {i+1}/{n_iter}")

    # Calculate statistics
    jaccard_arr = np.array(jaccard_values)
    counts_arr = np.array(schema_counts)

    # p-value: proportion with Jaccard >= 0.5 (high overlap = problematic)
    p_value = np.mean(jaccard_arr >= 0.5)

    return {
        'test_type': 'TAM-shuffling',
        'iterations': n_iter,
        'observed_schemas': len(observed_schemas),
        'mean_jaccard': round(float(np.mean(jaccard_arr)), 4),
        'std_jaccard': round(float(np.std(jaccard_arr)), 4),
        'median_jaccard': round(float(np.median(jaccard_arr)), 4),
        'min_jaccard': round(float(np.min(jaccard_arr)), 4),
        'max_jaccard': round(float(np.max(jaccard_arr)), 4),
        'p_value': round(float(p_value), 4),
        'mean_schema_count': round(float(np.mean(counts_arr)), 2),
        'validated': float(np.mean(jaccard_arr)) < 0.3
    }


def run_verb_shuffling(formulaic: List[Dict], observed_schemas: Set[str],
                       n_iter: int = 1000) -> Dict:
    """Verb-shuffling permutation test (within TAM categories)."""
    log("\n--- Verb-Shuffling Permutation Test ---")

    jaccard_values = []
    schema_counts = []

    # Group by TAM
    tam_groups = defaultdict(list)
    for i, c in enumerate(formulaic):
        tam_groups[c['tam']].append(i)

    for i in range(n_iter):
        # Shuffle verbs within each TAM group
        permuted = [c.copy() for c in formulaic]

        for tam, indices in tam_groups.items():
            verbs = [formulaic[idx]['head_lemma'] for idx in indices]
            random.shuffle(verbs)
            for j, idx in enumerate(indices):
                permuted[idx]['head_lemma'] = verbs[j]

        # Calculate metrics and apply dual-lane
        metrics = calculate_lc_metrics(permuted)
        accepted, _ = apply_dual_lane_g2(
            metrics,
            CONFIG['npmi_threshold'],
            CONFIG['h_slot_threshold'],
            CONFIG['g2_npmi_threshold'],
            CONFIG['g2_h_slot_threshold'],
            CONFIG['npmi_floor']
        )

        schema_counts.append(len(accepted))
        jaccard_values.append(calculate_jaccard(observed_schemas, accepted))

        if (i + 1) % 100 == 0:
            log(f"  Verb-shuffling: {i+1}/{n_iter}")

    # Calculate statistics
    jaccard_arr = np.array(jaccard_values)
    counts_arr = np.array(schema_counts)
    p_value = np.mean(jaccard_arr >= 0.5)

    return {
        'test_type': 'Verb-shuffling',
        'iterations': n_iter,
        'observed_schemas': len(observed_schemas),
        'mean_jaccard': round(float(np.mean(jaccard_arr)), 4),
        'std_jaccard': round(float(np.std(jaccard_arr)), 4),
        'median_jaccard': round(float(np.median(jaccard_arr)), 4),
        'min_jaccard': round(float(np.min(jaccard_arr)), 4),
        'max_jaccard': round(float(np.max(jaccard_arr)), 4),
        'p_value': round(float(p_value), 4),
        'mean_schema_count': round(float(np.mean(counts_arr)), 2),
        'validated': float(np.mean(jaccard_arr)) < 0.3
    }


# =============================================================================
# STEP 4: SENSITIVITY ANALYSIS (OFAT)
# =============================================================================

def run_sensitivity_analysis(valid_constructions: List[Dict],
                              observed_schemas: Set[str]) -> Dict:
    """One-At-a-Time (OFAT) sensitivity analysis."""
    log("\n" + "="*70)
    log("STEP 4: SENSITIVITY ANALYSIS (OFAT)")
    log("="*70)

    total_combinations = len(SENSITIVITY_GRIDS) * CONFIG['sensitivity_values']
    log(f"Methodology: One-At-a-Time (OFAT)")
    log(f"Total combinations: {total_combinations} ({len(SENSITIVITY_GRIDS)} params × {CONFIG['sensitivity_values']} values)")

    # Track schema presence across all combinations
    schema_presence = defaultdict(int)
    parameter_results = {}

    for param, values in SENSITIVITY_GRIDS.items():
        log(f"\nTesting {param}: {values}")
        param_schemas = []

        for val in values:
            # Set thresholds
            atp_t = val if param == 'atp' else CONFIG['atp_threshold']
            dpb_t = val if param == 'dpb' else CONFIG['dpb_threshold']
            hr_t = val if param == 'hr' else CONFIG['hr_threshold']
            npmi_t = val if param == 'npmi' else CONFIG['npmi_threshold']
            h_slot_t = val if param == 'h_slot' else CONFIG['h_slot_threshold']

            # Apply Layer 1
            formulaic = apply_instance_prefilters(valid_constructions, atp_t, dpb_t, hr_t)

            if not formulaic:
                param_schemas.append(set())
                continue

            # Calculate metrics and apply Layer 2
            metrics = calculate_lc_metrics(formulaic)
            accepted, _ = apply_dual_lane_g2(metrics, npmi_t, h_slot_t,
                                             CONFIG['g2_npmi_threshold'],
                                             CONFIG['g2_h_slot_threshold'],
                                             CONFIG['npmi_floor'])

            param_schemas.append(accepted)

            # Track presence
            for schema in accepted:
                schema_presence[schema] += 1

            log(f"  {param}={val}: {len(accepted)} schemas")

        # Calculate parameter-level stability
        all_schemas_param = set()
        for s in param_schemas:
            all_schemas_param.update(s)

        stable = []
        for schema in all_schemas_param:
            count = sum(1 for s in param_schemas if schema in s)
            if count >= 3:  # Majority (≥3/5)
                stable.append(schema)

        parameter_results[param] = {
            'values': values,
            'schema_counts': [len(s) for s in param_schemas],
            'stable_schemas': sorted(stable)
        }

    # Calculate overall stability for each schema
    schema_stability = {}
    for schema in observed_schemas:
        presence_count = schema_presence.get(schema, 0)
        pct = presence_count / total_combinations * 100
        schema_stability[schema] = {
            'presence_count': presence_count,
            'total_combinations': total_combinations,
            'stability_percentage': round(pct, 1),
            'classification': 'Globally stable' if pct >= 80 else 'Parameter-sensitive'
        }

    # Find globally stable schemas (stable across ALL parameters)
    globally_stable = []
    for schema in observed_schemas:
        stable_in_all = True
        for param, data in parameter_results.items():
            if schema not in data['stable_schemas']:
                stable_in_all = False
                break
        if stable_in_all:
            globally_stable.append(schema)

    return {
        'methodology': 'One-At-a-Time (OFAT)',
        'total_combinations': total_combinations,
        'parameters_tested': list(SENSITIVITY_GRIDS.keys()),
        'values_per_parameter': CONFIG['sensitivity_values'],
        'parameter_results': parameter_results,
        'schema_stability': schema_stability,
        'globally_stable_schemas': sorted(globally_stable),
        'parameter_sensitive_schemas': sorted([s for s in observed_schemas if s not in globally_stable])
    }


# =============================================================================
# STEP 5: ABLATION ANALYSIS
# =============================================================================

def run_ablation_analysis(valid_constructions: List[Dict],
                          observed_schemas: Set[str]) -> Dict:
    """Ablation analysis: with vs without Layer 1."""
    log("\n" + "="*70)
    log("STEP 5: ABLATION ANALYSIS")
    log("="*70)

    # WITH Layer 1 (standard pipeline)
    formulaic_with = apply_instance_prefilters(
        valid_constructions,
        CONFIG['atp_threshold'],
        CONFIG['dpb_threshold'],
        CONFIG['hr_threshold']
    )
    metrics_with = calculate_lc_metrics(formulaic_with)
    accepted_with, _ = apply_dual_lane_g2(
        metrics_with,
        CONFIG['npmi_threshold'],
        CONFIG['h_slot_threshold'],
        CONFIG['g2_npmi_threshold'],
        CONFIG['g2_h_slot_threshold'],
        CONFIG['npmi_floor']
    )

    log(f"WITH Layer 1: {len(formulaic_with)} instances → {len(accepted_with)} schemas")

    # WITHOUT Layer 1 (skip instance prefilters)
    metrics_without = calculate_lc_metrics(valid_constructions)
    accepted_without, _ = apply_dual_lane_g2(
        metrics_without,
        CONFIG['npmi_threshold'],
        CONFIG['h_slot_threshold'],
        CONFIG['g2_npmi_threshold'],
        CONFIG['g2_h_slot_threshold'],
        CONFIG['npmi_floor']
    )

    log(f"WITHOUT Layer 1: {len(valid_constructions)} instances → {len(accepted_without)} schemas")

    # Calculate differences
    false_positives = accepted_without - accepted_with
    retained = accepted_with & accepted_without

    log(f"False positives eliminated by Layer 1: {len(false_positives)}")
    log(f"Schemas retained in both: {len(retained)}")

    return {
        'with_layer1': {
            'instances': len(formulaic_with),
            'schemas': len(accepted_with),
            'schema_list': sorted(list(accepted_with))
        },
        'without_layer1': {
            'instances': len(valid_constructions),
            'schemas': len(accepted_without),
            'schema_list': sorted(list(accepted_without))
        },
        'false_positives_eliminated': len(false_positives),
        'false_positive_list': sorted(list(false_positives)),
        'reduction_percentage': round((1 - len(accepted_with)/len(accepted_without)) * 100, 1) if accepted_without else 0,
        'retained_schemas': sorted(list(retained))
    }


# =============================================================================
# STEP 6: GENERATE REPORTS
# =============================================================================

def generate_summary_md(results: Dict, output_dir: Path):
    """Generate SUMMARY.md for manuscript copy-paste."""
    summary_path = output_dir / 'SUMMARY.md'

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# LC Framework Analysis Results - Summary for Manuscript\n\n")
        f.write(f"**Generated:** {results['run_timestamp']}\n\n")

        f.write("---\n\n")

        # Configuration
        f.write("## Configuration (Final Validated)\n\n")
        f.write("### Layer 1 Thresholds:\n")
        f.write(f"- ATP ≥ {CONFIG['atp_threshold']}\n")
        f.write(f"- ΔP_backward ≥ {CONFIG['dpb_threshold']}\n")
        f.write(f"- H_r ≤ {CONFIG['hr_threshold']}\n\n")

        f.write("### Layer 2 Dual-Lane Acceptance:\n")
        f.write(f"- **NPMI Lane:** NPMI ≥ {CONFIG['npmi_threshold']} AND G² ≥ {CONFIG['g2_npmi_threshold']}\n")
        f.write(f"- **H_slot Lane:** H_slot ≥ {CONFIG['h_slot_threshold']} AND G² ≥ {CONFIG['g2_h_slot_threshold']} AND NPMI ≥ {CONFIG['npmi_floor']}\n\n")

        f.write("---\n\n")

        # Filtering Cascade
        fc = results['filtering_cascade']
        f.write("## Filtering Cascade\n\n")
        f.write("| Stage | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        f.write(f"| Total constructions | {fc['total_raw']:,} | 100% |\n")
        f.write(f"| Valid constructions | {fc['total_valid']:,} | {fc['valid_percentage']}% |\n")
        f.write(f"| Formulaic instances (Layer 1) | {fc['formulaic_instances']:,} | {fc['formulaic_percentage']}% |\n")
        f.write(f"| Unique TAM×COMP patterns | {fc['unique_patterns']} | - |\n")
        f.write(f"| **Accepted schemas** | **{fc['accepted_schemas']}** | **{fc['acceptance_rate']}%** |\n\n")

        f.write("---\n\n")

        # Schema Inventory
        f.write("## Accepted Schema Inventory\n\n")
        f.write("| Schema | n | NPMI | H_slot | G² | Lane |\n")
        f.write("|--------|---|------|--------|-----|------|\n")
        for s in results['schema_inventory']:
            f.write(f"| {s['schema']} | {s['n']} | {s['npmi']:.3f} | {s['h_slot']:.2f} | {s['g_squared']:.1f} | {s['lane']} |\n")
        f.write("\n---\n\n")

        # Permutation Testing
        f.write("## Permutation Testing\n\n")
        tam = results['permutation_testing']['tam_shuffling']
        verb = results['permutation_testing']['verb_shuffling']

        f.write("| Test | Iterations | Mean Jaccard | SD | p-value | Validated |\n")
        f.write("|------|------------|--------------|-----|---------|----------|\n")
        f.write(f"| TAM-shuffling | {tam['iterations']:,} | {tam['mean_jaccard']:.3f} | {tam['std_jaccard']:.3f} | {tam['p_value']:.3f} | {'✓' if tam['validated'] else '✗'} |\n")
        f.write(f"| Verb-shuffling | {verb['iterations']:,} | {verb['mean_jaccard']:.3f} | {verb['std_jaccard']:.3f} | {verb['p_value']:.3f} | {'✓' if verb['validated'] else '✗'} |\n\n")

        f.write("**Interpretation:** Jaccard < 0.3 indicates random permutations produce different schemas → method validated.\n\n")

        f.write("---\n\n")

        # Sensitivity Analysis
        sens = results['sensitivity_analysis']
        f.write("## Sensitivity Analysis\n\n")
        f.write(f"**Methodology:** {sens['methodology']}\n")
        f.write(f"**Total combinations:** {sens['total_combinations']} ({len(sens['parameters_tested'])} params × {sens['values_per_parameter']} values)\n\n")

        f.write("### Schema Stability:\n\n")
        f.write("| Schema | Stability | Classification |\n")
        f.write("|--------|-----------|----------------|\n")
        for schema, data in sorted(sens['schema_stability'].items()):
            f.write(f"| {schema} | {data['stability_percentage']}% | {data['classification']} |\n")

        f.write(f"\n**Globally stable:** {', '.join(sens['globally_stable_schemas']) or 'None'}\n")
        f.write(f"**Parameter-sensitive:** {', '.join(sens['parameter_sensitive_schemas']) or 'None'}\n\n")

        f.write("---\n\n")

        # Ablation Analysis
        abl = results['ablation_analysis']
        f.write("## Ablation Analysis (Layer 1 Impact)\n\n")
        f.write("| Condition | Instances | Schemas |\n")
        f.write("|-----------|-----------|--------|\n")
        f.write(f"| WITH Layer 1 | {abl['with_layer1']['instances']:,} | {abl['with_layer1']['schemas']} |\n")
        f.write(f"| WITHOUT Layer 1 | {abl['without_layer1']['instances']:,} | {abl['without_layer1']['schemas']} |\n\n")

        f.write(f"**False positives eliminated:** {abl['false_positives_eliminated']}\n")
        f.write(f"**Reduction:** {abl['reduction_percentage']}%\n\n")

        f.write("---\n\n")
        f.write("*Report generated by ConstructionMiner LC Framework*\n")

    log(f"Saved SUMMARY.md to {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Setup paths
    base_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean')
    asc_dir = base_dir / 'corpus_asc_output'

    # Read fresh directory name
    with open(base_dir / '.fresh_dir_name', 'r') as f:
        output_dir = base_dir / f.read().strip()

    log("="*70)
    log("LC FRAMEWORK - COMPLETE VALIDATION RUN")
    log("="*70)
    log(f"Output directory: {output_dir}")
    log(f"ASC input: {asc_dir}")
    log("")

    # Record start time
    start_time = datetime.now()

    # STEP 2: Run complete pipeline
    pipeline_results = run_complete_pipeline(asc_dir, output_dir)

    # STEP 3: Permutation testing
    log("\n" + "="*70)
    log("STEP 3: PERMUTATION TESTING")
    log("="*70)

    tam_results = run_tam_shuffling(
        pipeline_results['formulaic'],
        pipeline_results['accepted_set'],
        CONFIG['permutation_iterations']
    )

    verb_results = run_verb_shuffling(
        pipeline_results['formulaic'],
        pipeline_results['accepted_set'],
        CONFIG['permutation_iterations']
    )

    # STEP 4: Sensitivity analysis
    sens_results = run_sensitivity_analysis(
        pipeline_results['valid_constructions'],
        pipeline_results['accepted_set']
    )

    # STEP 5: Ablation analysis
    abl_results = run_ablation_analysis(
        pipeline_results['valid_constructions'],
        pipeline_results['accepted_set']
    )

    # STEP 6: Generate comprehensive report
    log("\n" + "="*70)
    log("STEP 6: GENERATING REPORTS")
    log("="*70)

    end_time = datetime.now()
    duration = end_time - start_time

    # Compile all results
    final_results = {
        'run_timestamp': start_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'configuration': CONFIG,
        'filtering_cascade': pipeline_results['filtering_cascade'],
        'schema_inventory': pipeline_results['schema_inventory'],
        'permutation_testing': {
            'tam_shuffling': tam_results,
            'verb_shuffling': verb_results
        },
        'sensitivity_analysis': sens_results,
        'ablation_analysis': abl_results
    }

    # Save comprehensive JSON
    json_path = output_dir / 'COMPLETE_RESULTS.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    log(f"Saved COMPLETE_RESULTS.json to {json_path}")

    # Generate SUMMARY.md
    generate_summary_md(final_results, output_dir)

    # Print final summary
    log("\n" + "="*70)
    log("COMPLETE - FINAL SUMMARY")
    log("="*70)
    log(f"Duration: {duration}")
    log(f"Output directory: {output_dir}")
    log(f"")
    log(f"Filtering Cascade:")
    log(f"  Total constructions: {pipeline_results['filtering_cascade']['total_raw']:,}")
    log(f"  Valid constructions: {pipeline_results['filtering_cascade']['total_valid']:,}")
    log(f"  Formulaic instances: {pipeline_results['filtering_cascade']['formulaic_instances']:,}")
    log(f"  Accepted schemas: {pipeline_results['filtering_cascade']['accepted_schemas']}")
    log(f"")
    log(f"Permutation Testing:")
    log(f"  TAM-shuffling: Jaccard={tam_results['mean_jaccard']:.3f}, p={tam_results['p_value']:.3f}")
    log(f"  Verb-shuffling: Jaccard={verb_results['mean_jaccard']:.3f}, p={verb_results['p_value']:.3f}")
    log(f"")
    log(f"Sensitivity Analysis:")
    log(f"  Globally stable: {len(sens_results['globally_stable_schemas'])} schemas")
    log(f"  Parameter-sensitive: {len(sens_results['parameter_sensitive_schemas'])} schemas")
    log(f"")
    log(f"Ablation Analysis:")
    log(f"  False positives eliminated: {abl_results['false_positives_eliminated']}")
    log("="*70)


if __name__ == "__main__":
    main()
