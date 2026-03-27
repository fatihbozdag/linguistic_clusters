#!/usr/bin/env python3
"""
Full Corpus Analysis Pipeline - ConstructionMiner Phase 2

Complete pipeline for passive construction analysis:
1. Load all ASC-analyzer output files
2. Extract TAM×COMP constructions
3. Apply filtering (invalid TAM, extraposed it-passives)
4. Calculate LC Framework metrics (ATP, ΔP, H_slot, Dispersion, G²)
5. Apply dual-lane acceptance criteria
6. Export results to CSV files

Output Files:
- constructions_filtered.csv: All valid constructions
- patterns_all_metrics.csv: All patterns with LC metrics
- schemas_accepted.csv: Patterns passing dual-lane acceptance
- analysis_statistics.txt: Comprehensive statistics report
"""

import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Import our modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from passive_extractor import PassiveExtractor
from passive_filter import filter_constructions, print_filter_statistics
from prep_normalizer import (
    normalize_constructions,
    generate_correction_report,
    normalize_pattern_metrics
)
from verb_validator import (
    validate_constructions as validate_verb_lemmas,
    generate_validation_report as generate_verb_report
)
from lc_metrics import (
    calculate_lc_metrics,
    apply_dual_lane_acceptance,
    print_dual_lane_statistics,
    build_corpus_statistics,
    calculate_instance_atp,
    calculate_instance_delta_p_backward,
    calculate_instance_boundary_entropy
)


class CorpusAnalysisPipeline:
    """Complete pipeline for passive construction corpus analysis."""

    def __init__(self, asc_output_dir: str, results_dir: str, mode: str = 'production'):
        """
        Initialize pipeline.

        Args:
            asc_output_dir: Directory containing *_ASCinfo.txt files
            results_dir: Directory for output CSV files and reports
            mode: 'production' or 'discovery' for dual-lane acceptance
        """
        self.asc_output_dir = Path(asc_output_dir)
        self.results_dir = Path(results_dir)
        self.mode = mode

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        # Initialize extractor with MPS
        self.extractor = PassiveExtractor(use_mps=True)

        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_constructions': 0,
            'filtered_constructions': 0,
            'normalized_constructions': 0,
            'prep_corrections': 0,
            'prep_filtered': 0,
            'verb_corrections': 0,
            'verb_filtered': 0,
            'formulaic_instances': 0,
            'filtered_by_atp': 0,
            'filtered_by_dpb': 0,
            'filtered_by_hr': 0,
            'total_patterns': 0,
            'accepted_patterns': 0
        }

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")

    def load_all_constructions(self) -> List[Dict]:
        """
        Load and extract constructions from all ASC files.

        Returns:
            List of construction dicts with keys:
                - doc_id: Document identifier
                - surface: Full construction text
                - tam: TAM category
                - comp: Complement type
                - subjtype: Subject type
                - head_lemma: Main verb lemma
                - pattern: Full pattern string
                - subject_text: Subject text (for extraposed detection)
        """
        self.log("="*70)
        self.log("STEP 1: Loading Constructions from ASC Files")
        self.log("="*70)

        asc_files = list(self.asc_output_dir.glob('*_ASCinfo.txt'))
        self.stats['total_files'] = len(asc_files)

        self.log(f"Found {len(asc_files)} ASC files to process")

        all_constructions = []

        for i, asc_file in enumerate(asc_files, 1):
            try:
                # Extract doc_id from filename (remove _ASCinfo.txt)
                doc_id = asc_file.stem.replace('_ASCinfo', '')

                # Extract constructions
                constructions = self.extractor.extract_from_file(str(asc_file))

                # Add doc_id to each construction
                for const in constructions:
                    const['doc_id'] = doc_id

                all_constructions.extend(constructions)
                self.stats['processed_files'] += 1

                if i % 100 == 0:
                    self.log(f"  Processed {i}/{len(asc_files)} files, "
                            f"{len(all_constructions)} constructions so far...")

            except Exception as e:
                self.stats['failed_files'] += 1
                self.log(f"  ✗ Error processing {asc_file.name}: {e}")

        self.stats['total_constructions'] = len(all_constructions)

        self.log(f"\n✓ Loaded {len(all_constructions):,} constructions from "
                f"{self.stats['processed_files']:,} files")
        if self.stats['failed_files'] > 0:
            self.log(f"  ⚠ {self.stats['failed_files']} files failed to process")

        return all_constructions

    def run_pipeline(self):
        """Run complete analysis pipeline."""
        self.log("\n" + "="*70)
        self.log("CONSTRUCTIONMINER - FULL CORPUS ANALYSIS PIPELINE")
        self.log("="*70)
        self.log(f"Mode: {self.mode.upper()}")
        self.log(f"ASC output directory: {self.asc_output_dir}")
        self.log(f"Results directory: {self.results_dir}")
        self.log("")

        # Step 1: Load constructions
        all_constructions = self.load_all_constructions()

        if len(all_constructions) == 0:
            self.log("\n✗ No constructions found. Exiting.")
            return

        # Step 2: Apply filtering
        self.log("\n" + "="*70)
        self.log("STEP 2: Applying Filters")
        self.log("="*70)

        valid_constructions, filter_stats = filter_constructions(all_constructions)
        self.stats['filtered_constructions'] = len(valid_constructions)

        print_filter_statistics(filter_stats)

        # Step 2.3: Normalize prepositions (fix typos from learner corpus)
        self.log("\n" + "="*70)
        self.log("STEP 2.3: Normalizing Prepositions")
        self.log("="*70)
        self.log("Correcting typos and tokenization errors in complements...")

        valid_constructions, norm_stats = normalize_constructions(valid_constructions)
        self.stats['normalized_constructions'] = len(valid_constructions)
        self.stats['prep_corrections'] = norm_stats['normalized']
        self.stats['prep_filtered'] = norm_stats['filtered']

        self.log(f"\n✓ Normalization Results:")
        self.log(f"  - Unchanged (valid): {norm_stats['unchanged']:,}")
        self.log(f"  - Corrected: {norm_stats['normalized']:,}")
        self.log(f"  - Filtered (invalid preps): {norm_stats['filtered']:,}")

        if norm_stats['corrections']:
            self.log(f"\n  Top corrections applied:")
            for correction, count in list(norm_stats['corrections'].most_common(10)):
                self.log(f"    {correction}: {count:,}")

        # Save normalization report
        norm_report_file = self.results_dir / 'normalization_report.txt'
        with open(norm_report_file, 'w', encoding='utf-8') as f:
            f.write(generate_correction_report(norm_stats))
        self.log(f"\n✓ Saved normalization report to {norm_report_file}")

        # Step 2.4: Validate verb lemmas (fix typos and tokenization errors)
        self.log("\n" + "="*70)
        self.log("STEP 2.4: Validating Verb Lemmas")
        self.log("="*70)
        self.log("Correcting typos and tokenization errors in verb lemmas...")

        valid_constructions, verb_stats = validate_verb_lemmas(valid_constructions)
        self.stats['verb_corrections'] = verb_stats['corrected'] + verb_stats['tokenization_errors']
        self.stats['verb_filtered'] = verb_stats['filtered']

        self.log(f"\n✓ Verb Validation Results:")
        self.log(f"  - Valid (unchanged): {verb_stats['valid']:,}")
        self.log(f"  - Corrected: {verb_stats['corrected']:,}")
        self.log(f"  - Tokenization errors fixed: {verb_stats['tokenization_errors']:,}")
        self.log(f"  - Filtered (invalid): {verb_stats['filtered']:,}")

        if verb_stats['corrections']:
            self.log(f"\n  Top corrections applied:")
            for orig, corrected in list(verb_stats['corrections'].items())[:10]:
                self.log(f"    {orig} -> {corrected}")

        # Save verb validation report
        verb_report_file = self.results_dir / 'verb_validation_report.txt'
        with open(verb_report_file, 'w', encoding='utf-8') as f:
            f.write(generate_verb_report(verb_stats))
        self.log(f"\n✓ Saved verb validation report to {verb_report_file}")

        # NOTE: Will save constructions_filtered.csv AFTER calculating instance metrics

        # Step 2.5: Build corpus statistics for instance metrics
        self.log("\n" + "="*70)
        self.log("STEP 2.5: Building Corpus Statistics for Instance Metrics")
        self.log("="*70)
        self.log(f"Analyzing {len(valid_constructions):,} constructions...")

        corpus_stats = build_corpus_statistics(valid_constructions)

        self.log(f"✓ Unique unigrams: {len(corpus_stats['unigram_counts']):,}")
        self.log(f"✓ Unique bigrams: {len(corpus_stats['bigram_counts']):,}")
        self.log(f"✓ Total words: {corpus_stats['total_words']:,}")
        self.log(f"✓ Patterns with followers: {len(corpus_stats['follower_counts']):,}")

        # Step 2.6: Calculate instance-level metrics
        self.log("\n" + "="*70)
        self.log("STEP 2.6: Calculating Instance-Level Metrics")
        self.log("="*70)
        self.log("Measuring surface-level formulaicity (ATP, ΔP_backward, H_r)...")

        for i, const in enumerate(valid_constructions):
            if i % 1000 == 0 and i > 0:
                self.log(f"  Progress: {i:,}/{len(valid_constructions):,} ({i/len(valid_constructions)*100:.1f}%)")

            const['instance_atp'] = calculate_instance_atp(const, corpus_stats)
            const['instance_dpb'] = calculate_instance_delta_p_backward(const, corpus_stats)
            const['instance_hr'] = calculate_instance_boundary_entropy(const, corpus_stats)

        self.log(f"✓ Calculated metrics for {len(valid_constructions):,} constructions")

        # Save filtered constructions WITH instance metrics to CSV
        constructions_csv = self.results_dir / 'constructions_filtered.csv'
        self.save_constructions_csv(valid_constructions, constructions_csv)
        self.log(f"\n✓ Saved {len(valid_constructions):,} filtered constructions to {constructions_csv}")

        # Step 2.7: Apply instance prefilters (Layer 1 - Surface Formulaicity)
        self.log("\n" + "="*70)
        self.log("STEP 2.7: Applying Instance-Level Prefilters (Layer 1)")
        self.log("="*70)
        self.log("Thresholds: ATP ≥ 0.10, ΔP_backward ≥ 0.10, H_r ≤ 2.8")

        formulaic_instances = []
        instance_filter_stats = {
            'total': len(valid_constructions),
            'filtered_by_atp': 0,
            'filtered_by_dpb': 0,
            'filtered_by_hr': 0,
            'passed': 0
        }

        for const in valid_constructions:
            # Apply thresholds (must pass ALL three)
            atp = const.get('instance_atp', 0.0)
            dpb = const.get('instance_dpb', 0.0)
            hr = const.get('instance_hr', float('inf'))

            # Track which filters fail
            if atp < 0.10:
                instance_filter_stats['filtered_by_atp'] += 1
                continue
            if dpb < 0.10:
                instance_filter_stats['filtered_by_dpb'] += 1
                continue
            if hr > 2.8:
                instance_filter_stats['filtered_by_hr'] += 1
                continue

            # Passed all filters
            formulaic_instances.append(const)
            instance_filter_stats['passed'] += 1

        self.stats['formulaic_instances'] = len(formulaic_instances)
        self.stats['filtered_by_atp'] = instance_filter_stats['filtered_by_atp']
        self.stats['filtered_by_dpb'] = instance_filter_stats['filtered_by_dpb']
        self.stats['filtered_by_hr'] = instance_filter_stats['filtered_by_hr']

        # Print instance filter statistics
        self.log(f"\n✓ Instance Prefilter Results:")
        self.log(f"  Total valid constructions: {instance_filter_stats['total']:,}")
        self.log(f"  Formulaic instances (passed): {instance_filter_stats['passed']:,} "
                f"({instance_filter_stats['passed']/instance_filter_stats['total']*100:.2f}%)")
        self.log(f"\n  Filtered out: {instance_filter_stats['total']-instance_filter_stats['passed']:,} "
                f"({(instance_filter_stats['total']-instance_filter_stats['passed'])/instance_filter_stats['total']*100:.2f}%)")
        self.log(f"    - ATP < 0.10: {instance_filter_stats['filtered_by_atp']:,} "
                f"({instance_filter_stats['filtered_by_atp']/instance_filter_stats['total']*100:.1f}%)")
        self.log(f"    - ΔP_backward < 0.10: {instance_filter_stats['filtered_by_dpb']:,} "
                f"({instance_filter_stats['filtered_by_dpb']/instance_filter_stats['total']*100:.1f}%)")
        self.log(f"    - H_r > 2.8: {instance_filter_stats['filtered_by_hr']:,} "
                f"({instance_filter_stats['filtered_by_hr']/instance_filter_stats['total']*100:.1f}%)")

        # Save formulaic instances to CSV
        formulaic_csv = self.results_dir / 'constructions_formulaic.csv'
        self.save_constructions_csv(formulaic_instances, formulaic_csv)
        self.log(f"\n✓ Saved {len(formulaic_instances):,} formulaic instances to {formulaic_csv}")

        if len(formulaic_instances) == 0:
            self.log("\n⚠ No formulaic instances passed prefilters. Exiting.")
            return

        # Step 3: Calculate LC metrics (on formulaic instances only)
        self.log("\n" + "="*70)
        self.log("STEP 3: Calculating LC Framework Metrics (Layer 2)")
        self.log("="*70)
        self.log("Computing schema-level metrics (NPMI, H_slot, G²) for formulaic instances...")

        metrics = calculate_lc_metrics(formulaic_instances)
        self.stats['total_patterns'] = len(metrics)

        self.log(f"\n✓ Calculated metrics for {len(metrics):,} unique patterns")

        # Save all patterns with metrics
        metrics_csv = self.results_dir / 'patterns_all_metrics.csv'
        self.save_metrics_csv(metrics, metrics_csv)
        self.log(f"✓ Saved all patterns with metrics to {metrics_csv}")

        # Step 4: Apply dual-lane acceptance
        self.log("\n" + "="*70)
        self.log("STEP 4: Applying Dual-Lane Acceptance")
        self.log("="*70)

        accepted_schemas, acceptance_stats = apply_dual_lane_acceptance(metrics, mode=self.mode)
        self.stats['accepted_patterns'] = len(accepted_schemas)

        print_dual_lane_statistics(acceptance_stats)

        # Save accepted schemas
        schemas_csv = self.results_dir / 'schemas_accepted.csv'
        self.save_metrics_csv(accepted_schemas, schemas_csv)
        self.log(f"\n✓ Saved {len(accepted_schemas):,} accepted schemas to {schemas_csv}")

        # Step 5: Generate final report
        self.log("\n" + "="*70)
        self.log("STEP 5: Generating Final Report")
        self.log("="*70)

        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = self.stats['end_time'] - self.stats['start_time']

        report_file = self.results_dir / 'analysis_statistics.txt'
        self.generate_report(report_file, filter_stats, acceptance_stats)
        self.log(f"\n✓ Saved comprehensive report to {report_file}")

        # Optional: Run validation tests
        # Uncomment to enable statistical validation
        """
        self.log("\n" + "="*70)
        self.log("STEP 6: Statistical Validation Tests")
        self.log("="*70)

        from validation_tests import (
            permutation_test_tam_comp,
            ablation_test_filters,
            ablation_test_dual_lane,
            print_validation_report
        )

        # Run permutation test
        self.log("\n1. Permutation Test (TAM×COMP Associations)")
        self.log("-"*70)
        perm_results = permutation_test_tam_comp(
            formulaic_instances,
            n_iterations=1000
        )

        # Run ablation tests
        self.log("\n2. Ablation Test (Instance Filters)")
        self.log("-"*70)
        ablation_filter_results = ablation_test_filters(valid_constructions)

        self.log("\n3. Ablation Test (Dual-Lane)")
        self.log("-"*70)
        ablation_lane_results = ablation_test_dual_lane(lc_metrics)

        # Print comprehensive report
        print_validation_report(
            perm_results,
            ablation_filter_results,
            ablation_lane_results
        )

        # Export validation results
        import json
        validation_file = self.results_dir / 'validation_results.json'
        with open(validation_file, 'w') as f:
            json.dump({
                'permutation_test': {
                    'real_schema_count': perm_results['real_schema_count'],
                    'mean_permuted': perm_results['mean_permuted'],
                    'std_permuted': perm_results['std_permuted'],
                    'p_value': perm_results['p_value'],
                    'significant': perm_results['significant']
                },
                'ablation_filters': ablation_filter_results,
                'ablation_dual_lane': ablation_lane_results
            }, f, indent=2)

        self.log(f"\n✓ Saved validation results to {validation_file}")
        """

        # Summary
        self.log("\n" + "="*70)
        self.log("PIPELINE COMPLETE")
        self.log("="*70)
        self.log(f"\nProcessed: {self.stats['processed_files']:,} files")
        self.log(f"Total constructions: {self.stats['total_constructions']:,}")
        self.log(f"Valid constructions: {self.stats['filtered_constructions']:,} "
                f"({self.stats['filtered_constructions']/self.stats['total_constructions']*100:.1f}%)")
        self.log(f"Formulaic instances (Layer 1): {self.stats['formulaic_instances']:,} "
                f"({self.stats['formulaic_instances']/self.stats['filtered_constructions']*100:.2f}%)")
        self.log(f"Unique patterns (Layer 2): {self.stats['total_patterns']:,}")
        self.log(f"Accepted schemas (Layer 3): {self.stats['accepted_patterns']:,} "
                f"({self.stats['accepted_patterns']/self.stats['total_patterns']*100:.1f}%)")
        self.log(f"\nDuration: {self.stats['duration']}")
        self.log("\nOutput files:")
        self.log(f"  1. {constructions_csv}")
        self.log(f"  2. {formulaic_csv}")
        self.log(f"  3. {metrics_csv}")
        self.log(f"  4. {schemas_csv}")
        self.log(f"  5. {report_file}")
        self.log("="*70)

    def save_constructions_csv(self, constructions: List[Dict], output_file: Path):
        """Save constructions to CSV."""
        if len(constructions) == 0:
            return

        fieldnames = [
            'doc_id', 'surface', 'tam', 'comp', 'subjtype',
            'head_lemma', 'pattern', 'subject_text',
            'instance_atp', 'instance_dpb', 'instance_hr'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(constructions)

    def save_metrics_csv(self, metrics: Dict[str, Dict], output_file: Path):
        """Save patterns with metrics to CSV."""
        if len(metrics) == 0:
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'pattern', 'n_tokens', 'n_docs', 'atp', 'delta_p',
                'h_slot', 'npmi', 'dispersion', 'g_squared', 'p_value', 'p_value_fdr'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Sort by token frequency
            for pattern in sorted(metrics.keys(), key=lambda p: metrics[p]['n_tokens'], reverse=True):
                row = {'pattern': pattern, **metrics[pattern]}
                writer.writerow(row)

    def generate_report(self, report_file: Path, filter_stats: Dict, acceptance_stats: Dict):
        """Generate comprehensive text report."""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CONSTRUCTIONMINER - FULL CORPUS ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Generated: {self.stats['end_time']}\n")
            f.write(f"Mode: {self.mode.upper()}\n")
            f.write(f"Duration: {self.stats['duration']}\n\n")

            # File processing
            f.write("-"*70 + "\n")
            f.write("FILE PROCESSING\n")
            f.write("-"*70 + "\n")
            f.write(f"Total ASC files: {self.stats['total_files']:,}\n")
            f.write(f"Successfully processed: {self.stats['processed_files']:,}\n")
            f.write(f"Failed: {self.stats['failed_files']:,}\n\n")

            # Construction extraction
            f.write("-"*70 + "\n")
            f.write("CONSTRUCTION EXTRACTION\n")
            f.write("-"*70 + "\n")
            f.write(f"Total constructions extracted: {self.stats['total_constructions']:,}\n\n")

            # Filtering
            f.write("-"*70 + "\n")
            f.write("FILTERING\n")
            f.write("-"*70 + "\n")
            f.write(f"Valid constructions: {filter_stats['valid']:,} "
                   f"({filter_stats['valid']/filter_stats['total']*100:.1f}%)\n")
            f.write(f"Filtered out: {filter_stats['total']-filter_stats['valid']:,} "
                   f"({(filter_stats['total']-filter_stats['valid'])/filter_stats['total']*100:.1f}%)\n\n")
            f.write(f"  Invalid TAM: {filter_stats['filtered_invalid_tam']:,}\n")
            f.write(f"  Extraposed it-passives: {filter_stats['filtered_extraposed']:,}\n")
            f.write(f"  Missing fields: {filter_stats['filtered_missing_fields']:,}\n\n")

            # TAM distribution
            f.write("Valid TAM Distribution:\n")
            for tam in sorted(filter_stats['by_tam'].keys()):
                count = filter_stats['by_tam'][tam]
                pct = count / filter_stats['valid'] * 100
                f.write(f"  {tam:<15} {count:>6,} ({pct:>5.1f}%)\n")
            f.write("\n")

            # Preposition normalization
            f.write("-"*70 + "\n")
            f.write("PREPOSITION NORMALIZATION (Data Quality)\n")
            f.write("-"*70 + "\n")
            f.write(f"Input: {filter_stats['valid']:,} valid constructions\n")
            f.write(f"Output: {self.stats['normalized_constructions']:,} normalized constructions\n\n")
            f.write(f"  Corrected (typos fixed): {self.stats['prep_corrections']:,}\n")
            f.write(f"  Filtered (invalid preps): {self.stats['prep_filtered']:,}\n\n")

            # Instance prefilters (Layer 1)
            f.write("-"*70 + "\n")
            f.write("INSTANCE-LEVEL PREFILTERS (Layer 1 - Surface Formulaicity)\n")
            f.write("-"*70 + "\n")
            f.write(f"Input: {self.stats['normalized_constructions']:,} normalized constructions\n")
            f.write(f"Output: {self.stats['formulaic_instances']:,} formulaic instances "
                   f"({self.stats['formulaic_instances']/self.stats['normalized_constructions']*100:.2f}%)\n\n")

            f.write("Thresholds:\n")
            f.write("  - ATP (Average Transition Probability) ≥ 0.10\n")
            f.write("  - ΔP_backward (Delta P Backward) ≥ 0.10\n")
            f.write("  - H_r (Boundary Entropy) ≤ 2.8\n\n")

            total_filtered = self.stats['normalized_constructions'] - self.stats['formulaic_instances']
            f.write(f"Filtered out: {total_filtered:,} "
                   f"({total_filtered/self.stats['normalized_constructions']*100:.2f}%)\n")
            f.write(f"  - ATP < 0.10: {self.stats['filtered_by_atp']:,} "
                   f"({self.stats['filtered_by_atp']/self.stats['normalized_constructions']*100:.1f}%)\n")
            f.write(f"  - ΔP_backward < 0.10: {self.stats['filtered_by_dpb']:,} "
                   f"({self.stats['filtered_by_dpb']/self.stats['normalized_constructions']*100:.1f}%)\n")
            f.write(f"  - H_r > 2.8: {self.stats['filtered_by_hr']:,} "
                   f"({self.stats['filtered_by_hr']/self.stats['normalized_constructions']*100:.1f}%)\n\n")

            # Pattern analysis (Layer 2)
            f.write("-"*70 + "\n")
            f.write("PATTERN ANALYSIS (Layer 2 - Schema-Level Metrics)\n")
            f.write("-"*70 + "\n")
            f.write(f"Input: {self.stats['formulaic_instances']:,} formulaic instances\n")
            f.write(f"Output: {self.stats['total_patterns']:,} unique patterns\n\n")

            # Dual-Lane Acceptance (Layer 3 - OR Logic with tightened H_slot)
            f.write("-"*70 + "\n")
            f.write("DUAL-LANE FORMULAICITY VALIDATION (Layer 3 - OR LOGIC, H_slot≥1.5)\n")
            f.write("-"*70 + "\n")
            f.write(f"Mode: {acceptance_stats['mode'].upper()}\n")
            f.write(f"Input: {self.stats['total_patterns']:,} unique patterns\n")
            f.write(f"Output: {self.stats['accepted_patterns']:,} accepted schemas\n\n")

            f.write(f"Thresholds (G²-based significance with permutation validation):\n")
            f.write(f"  Layer 1 (NPMI - Constructional Association):\n")
            f.write(f"    - NPMI ≥ {acceptance_stats['thresholds']['npmi']}\n")
            f.write(f"    - G² ≥ {acceptance_stats['thresholds']['g2_npmi']} (p < 0.05)\n")
            f.write(f"  Layer 2 (H_slot - Lexical Productivity):\n")
            f.write(f"    - H_slot ≥ {acceptance_stats['thresholds']['h_slot']}\n")
            f.write(f"    - G² ≥ {acceptance_stats['thresholds']['g2_h_slot']}\n")
            f.write(f"    - NPMI floor ≥ 0 (filters negative associations)\n\n")

            f.write(f"Results:\n")
            f.write(f"  Layer 1 PASS (NPMI): {acceptance_stats['lane1_npmi_passed']:,} "
                   f"({acceptance_stats['lane1_npmi_passed']/acceptance_stats['total_patterns']*100:.1f}%)\n")
            f.write(f"  Layer 2 PASS (H_slot): {acceptance_stats['lane2_h_slot_passed']:,} "
                   f"({acceptance_stats['lane2_h_slot_passed']/acceptance_stats['total_patterns']*100:.1f}%)\n\n")
            f.write(f"  ✅ ACCEPTED SCHEMAS (EITHER lane passes):\n")
            f.write(f"    - NPMI-only (fixed collocations): {acceptance_stats['npmi_only_passed']:,}\n")
            f.write(f"    - H_slot-only (productive slots): {acceptance_stats['h_slot_only_passed']:,}\n")
            f.write(f"    - Both (highly formulaic): {acceptance_stats['both_lanes_passed']:,}\n")
            f.write(f"    - Total accepted: {acceptance_stats['total_accepted']:,} "
                   f"({acceptance_stats['total_accepted']/acceptance_stats['total_patterns']*100:.1f}%)\n\n")

            # Final summary
            f.write("="*70 + "\n")
            f.write("SUMMARY - FOUR-LAYER FILTERING PIPELINE\n")
            f.write("="*70 + "\n")
            f.write(f"Files processed: {self.stats['processed_files']:,}\n\n")

            f.write("Pipeline Flow:\n")
            f.write(f"  1. Total constructions extracted: {self.stats['total_constructions']:,}\n")
            f.write(f"  2. Valid constructions (basic filters): {self.stats['filtered_constructions']:,} "
                   f"({self.stats['filtered_constructions']/self.stats['total_constructions']*100:.1f}%)\n")
            f.write(f"  3. Normalized constructions (prep fix): {self.stats['normalized_constructions']:,} "
                   f"(corrections: {self.stats['prep_corrections']:,}, filtered: {self.stats['prep_filtered']:,})\n")
            f.write(f"  4. Formulaic instances (Layer 1): {self.stats['formulaic_instances']:,} "
                   f"({self.stats['formulaic_instances']/self.stats['normalized_constructions']*100:.2f}%)\n")
            f.write(f"  5. Unique patterns (Layer 2): {self.stats['total_patterns']:,}\n")
            f.write(f"  6. Accepted schemas (Layer 3): {self.stats['accepted_patterns']:,} "
                   f"({self.stats['accepted_patterns']/self.stats['total_patterns']*100:.1f}%)\n\n")

            f.write("Overall Acceptance Rate:\n")
            f.write(f"  Formulaic instances / Normalized constructions: "
                   f"{self.stats['formulaic_instances']/self.stats['normalized_constructions']*100:.2f}%\n")
            f.write(f"  Accepted schemas / Total patterns: "
                   f"{self.stats['accepted_patterns']/self.stats['total_patterns']*100:.1f}%\n")
            f.write("="*70 + "\n")


def main():
    """Run full corpus analysis pipeline."""
    # Configuration
    asc_output_dir = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/corpus_asc_output'
    results_dir = '/Users/fatihbozdag/Documents/ConstructionMiner-Clean/analysis_results'
    mode = 'production'  # or 'discovery'

    # Create pipeline
    pipeline = CorpusAnalysisPipeline(asc_output_dir, results_dir, mode)

    # Run
    try:
        pipeline.run_pipeline()
        print("\n✓ Pipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
