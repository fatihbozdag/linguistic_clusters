#!/usr/bin/env python3
"""
Qualitative Example Extractor for Accepted Schemas

Extracts real corpus examples for each accepted schema pattern,
providing qualitative evidence to complement quantitative metrics.

This addresses CLaLT reviewer concern:
"Need qualitative examples to illustrate schema patterns."

Usage:
    python scripts/extract_examples.py
"""

import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_formulaic_instances(csv_path: Path) -> List[Dict]:
    """Load formulaic instances from CSV."""
    instances = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append(row)
    return instances


def load_accepted_schemas(csv_path: Path) -> Set[str]:
    """Load accepted schema patterns."""
    schemas = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            schemas.add(row['pattern'])
    return schemas


def extract_examples_for_schema(
    instances: List[Dict],
    pattern: str,
    max_examples: int = 5
) -> List[Dict]:
    """
    Extract example instances for a specific schema pattern.

    Args:
        instances: List of all formulaic instances
        pattern: The TAM,COMP pattern to find examples for
        max_examples: Maximum number of examples to return

    Returns:
        List of example dictionaries with surface form and metadata
    """
    matching = [inst for inst in instances if inst.get('pattern') == pattern]

    # If we have more than max_examples, sample randomly
    if len(matching) > max_examples:
        random.seed(42)  # Reproducibility
        matching = random.sample(matching, max_examples)

    examples = []
    for inst in matching:
        examples.append({
            'surface': inst.get('surface', ''),
            'subject': inst.get('subject_text', ''),
            'verb': inst.get('head_lemma', ''),
            'doc_id': inst.get('doc_id', ''),
            'atp': float(inst.get('instance_atp', 0) or 0),
            'dpb': float(inst.get('instance_dpb', 0) or 0),
            'hr': float(inst.get('instance_hr', 999) or 999),
        })

    return examples


def analyze_schema_semantics(examples: List[Dict]) -> Dict:
    """
    Analyze semantic patterns in schema examples.

    Returns summary statistics about the examples.
    """
    verbs = [ex['verb'] for ex in examples]
    verb_counts = defaultdict(int)
    for v in verbs:
        verb_counts[v] += 1

    return {
        'n_examples': len(examples),
        'unique_verbs': len(set(verbs)),
        'top_verbs': sorted(verb_counts.items(), key=lambda x: -x[1])[:5],
        'avg_atp': sum(ex['atp'] for ex in examples) / len(examples) if examples else 0,
        'avg_dpb': sum(ex['dpb'] for ex in examples) / len(examples) if examples else 0,
    }


def generate_examples_report(
    schemas: Set[str],
    instances: List[Dict],
    output_file: Path
):
    """Generate comprehensive examples report."""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("QUALITATIVE EXAMPLES FOR ACCEPTED SCHEMAS\n")
        f.write("=" * 80 + "\n\n")

        f.write("This report provides real corpus examples for each accepted schema,\n")
        f.write("demonstrating the concrete linguistic patterns captured by the metrics.\n\n")

        all_examples = {}

        for pattern in sorted(schemas):
            examples = extract_examples_for_schema(instances, pattern)
            semantics = analyze_schema_semantics(examples)
            all_examples[pattern] = {
                'examples': examples,
                'semantics': semantics
            }

            f.write("-" * 80 + "\n")
            f.write(f"SCHEMA: {pattern}\n")
            f.write("-" * 80 + "\n\n")

            # Parse pattern
            parts = pattern.split(',')
            tam = parts[0] if parts else 'unknown'
            comp = parts[1] if len(parts) > 1 else 'unknown'

            f.write(f"TAM Category: {tam}\n")
            f.write(f"Complement Type: {comp}\n")
            f.write(f"Total Instances: {semantics['n_examples']}\n")
            f.write(f"Unique Verbs: {semantics['unique_verbs']}\n")

            if semantics['top_verbs']:
                f.write(f"Top Verbs: {', '.join(f'{v}({c})' for v, c in semantics['top_verbs'])}\n")

            f.write(f"Average ATP: {semantics['avg_atp']:.3f}\n")
            f.write(f"Average ΔP: {semantics['avg_dpb']:.3f}\n\n")

            f.write("EXAMPLES:\n\n")

            for i, ex in enumerate(examples, 1):
                f.write(f"  {i}. \"{ex['surface']}\"\n")
                f.write(f"     Subject: {ex['subject']}\n")
                f.write(f"     Verb: {ex['verb']}\n")
                f.write(f"     Source: {ex['doc_id']}\n")
                f.write(f"     Metrics: ATP={ex['atp']:.2f}, ΔP={ex['dpb']:.2f}, H_r={ex['hr']:.2f}\n\n")

            f.write("\n")

        # Summary section
        f.write("=" * 80 + "\n")
        f.write("SUMMARY OBSERVATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Analyze patterns across schemas
        all_verbs = defaultdict(set)
        for pattern, data in all_examples.items():
            for ex in data['examples']:
                all_verbs[ex['verb']].add(pattern)

        # Find verbs that appear across multiple schemas
        multi_schema_verbs = [(v, patterns) for v, patterns in all_verbs.items()
                              if len(patterns) > 1]

        if multi_schema_verbs:
            f.write("Verbs appearing across multiple schemas:\n")
            for verb, patterns in sorted(multi_schema_verbs, key=lambda x: -len(x[1])):
                f.write(f"  - {verb}: {', '.join(sorted(patterns))}\n")
            f.write("\n")

        # Identify semantic clusters
        f.write("Schema Semantic Clusters:\n\n")

        # Spatial/directional prepositions
        spatial = [p for p in schemas if any(x in p for x in ['_up_', '_over_', '_above_', '_around_', '_towards_', '_within_', '_out_'])]
        if spatial:
            f.write("  1. SPATIAL/DIRECTIONAL PREPOSITIONS:\n")
            for p in sorted(spatial):
                f.write(f"     - {p}\n")
            f.write("     Observation: Modal passives frequently combine with spatial complements,\n")
            f.write("     suggesting formulaic 'potential movement/change' constructions.\n\n")

        # Note: Entity-marked by-phrases section removed
        # NER annotations were removed for consistent granularity across all complement types
        # All by-phrases are now unified as by_NP

        f.write("=" * 80 + "\n")

    return all_examples


def main():
    """Extract examples for all accepted schemas."""
    results_dir = Path('/Users/fatihbozdag/Documents/ConstructionMiner-Clean/analysis_results')

    formulaic_csv = results_dir / 'constructions_formulaic.csv'
    schemas_csv = results_dir / 'schemas_accepted.csv'

    if not formulaic_csv.exists():
        print(f"Error: {formulaic_csv} not found.")
        print("Please run the full pipeline first.")
        sys.exit(1)

    if not schemas_csv.exists():
        print(f"Error: {schemas_csv} not found.")
        print("Please run the full pipeline first.")
        sys.exit(1)

    print("Loading data...")
    instances = load_formulaic_instances(formulaic_csv)
    schemas = load_accepted_schemas(schemas_csv)

    print(f"Loaded {len(instances):,} formulaic instances")
    print(f"Found {len(schemas)} accepted schemas")

    # Generate report
    report_file = results_dir / 'schema_examples_report.txt'
    examples_data = generate_examples_report(schemas, instances, report_file)
    print(f"\n✓ Saved examples report to {report_file}")

    # Save JSON for further analysis
    json_file = results_dir / 'schema_examples.json'

    # Convert for JSON serialization
    json_data = {}
    for pattern, data in examples_data.items():
        json_data[pattern] = {
            'examples': data['examples'],
            'semantics': {
                'n_examples': data['semantics']['n_examples'],
                'unique_verbs': data['semantics']['unique_verbs'],
                'top_verbs': data['semantics']['top_verbs'],
                'avg_atp': data['semantics']['avg_atp'],
                'avg_dpb': data['semantics']['avg_dpb'],
            }
        }

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved JSON to {json_file}")


if __name__ == "__main__":
    main()
