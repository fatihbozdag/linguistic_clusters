#!/usr/bin/env python3
"""
Test TAM×COMP extraction on ASC-analyzer output.

Processes 3 test documents and provides:
- Individual construction details
- Distribution statistics (TAM, COMP, SUBJTYPE)
- Example constructions
- Validation of extraction quality
"""

import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from passive_extractor import PassiveExtractor, parse_asc_file


def test_extraction():
    """Test on 3 documents with comprehensive output."""

    test_files = [
        "test_data/BGSU1001_ASCinfo.txt",
        "test_data/BGSU1002_ASCinfo.txt",
        "test_data/BGSU1003_ASCinfo.txt"
    ]

    print("="*80)
    print("TAM×COMP EXTRACTION TEST")
    print("="*80)
    print()

    extractor = PassiveExtractor()
    all_results = []

    # Process each file
    for asc_file in test_files:
        print(f"\n{'='*80}")
        print(f"Processing: {asc_file}")
        print('='*80)

        sentences = parse_asc_file(asc_file)
        print(f"Sentences: {len(sentences)}")

        file_results = []

        for sent_idx, tokens in enumerate(sentences, 1):
            constructions = extractor.extract_from_sentence(tokens, sent_idx)

            for const in constructions:
                print(f"\nSentence {sent_idx}:")
                print(f"  Surface:   {const['surface']}")
                print(f"  TAM:       {const['tam']}")
                print(f"  COMP:      {const['comp']}")
                print(f"  SUBJTYPE:  {const['subjtype']}")
                print(f"  Head:      {const['head_lemma']}")
                print(f"  Pattern:   {const['pattern']}")

                file_results.append(const)
                all_results.append(const)

        print(f"\n✓ Extracted {len(file_results)} constructions from {asc_file}")

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)
    print(f"\nTotal constructions extracted: {len(all_results)}")

    # TAM distribution
    tam_counts = Counter(c['tam'] for c in all_results)
    print("\n" + "-"*80)
    print("TAM Distribution:")
    print("-"*80)
    for tam, count in tam_counts.most_common():
        pct = (count / len(all_results)) * 100
        print(f"  {tam:<15} {count:>3} ({pct:>5.1f}%)")

    # COMP distribution
    comp_counts = Counter(c['comp'] for c in all_results)
    print("\n" + "-"*80)
    print("COMP Distribution:")
    print("-"*80)
    for comp, count in comp_counts.most_common():
        pct = (count / len(all_results)) * 100
        print(f"  {comp:<15} {count:>3} ({pct:>5.1f}%)")

    # SUBJTYPE distribution
    subj_counts = Counter(c['subjtype'] for c in all_results)
    print("\n" + "-"*80)
    print("SUBJTYPE Distribution:")
    print("-"*80)
    for subj, count in subj_counts.most_common():
        pct = (count / len(all_results)) * 100
        print(f"  {subj:<15} {count:>3} ({pct:>5.1f}%)")

    # Pattern distribution (top 10)
    pattern_counts = Counter(c['pattern'] for c in all_results)
    print("\n" + "-"*80)
    print("Top 10 Complete Patterns:")
    print("-"*80)
    for pattern, count in pattern_counts.most_common(10):
        pct = (count / len(all_results)) * 100
        print(f"  {pattern:<50} {count:>3} ({pct:>5.1f}%)")

    # Head verb distribution (top 10)
    verb_counts = Counter(c['head_lemma'] for c in all_results)
    print("\n" + "-"*80)
    print("Top 10 Passive Verbs:")
    print("-"*80)
    for verb, count in verb_counts.most_common(10):
        pct = (count / len(all_results)) * 100
        print(f"  {verb:<15} {count:>3} ({pct:>5.1f}%)")

    # ========================================================================
    # Example Constructions (one per TAM type)
    # ========================================================================

    print(f"\n{'='*80}")
    print("EXAMPLE CONSTRUCTIONS BY TAM TYPE")
    print('='*80)

    for tam_type in tam_counts.keys():
        example = next((c for c in all_results if c['tam'] == tam_type), None)
        if example:
            print(f"\n{tam_type.upper()}:")
            print(f"  \"{example['surface']}\"")
            print(f"  Pattern: {example['pattern']}")

    # ========================================================================
    # Validation Checks
    # ========================================================================

    print(f"\n{'='*80}")
    print("VALIDATION CHECKS")
    print('='*80)

    # Check for unknowns
    unknown_tam = [c for c in all_results if c['tam'] == 'unknown']
    if unknown_tam:
        print(f"\n⚠ Found {len(unknown_tam)} constructions with unknown TAM:")
        for c in unknown_tam[:3]:
            print(f"  - {c['surface']}")
            print(f"    Aux chain: {c['aux_chain']}")
    else:
        print("\n✓ No unknown TAM types")

    # Check for zero subjects
    no_subject = [c for c in all_results if c['subject_text'] is None]
    if no_subject:
        print(f"\n⚠ Found {len(no_subject)} constructions with no subject:")
        for c in no_subject[:3]:
            print(f"  - {c['surface']}")
    else:
        print("✓ All constructions have subjects")

    # Check TAM coverage
    expected_tams = {'pres-be', 'past-be', 'perf-be', 'modal-be', 'prog-be', 'get-be'}
    found_tams = set(tam_counts.keys()) - {'unknown'}
    print(f"\n✓ Found {len(found_tams)}/{len(expected_tams)} expected TAM types: {sorted(found_tams)}")

    # ========================================================================
    # Per-File Breakdown
    # ========================================================================

    print(f"\n{'='*80}")
    print("PER-FILE BREAKDOWN")
    print('='*80)

    for asc_file in test_files:
        file_name = Path(asc_file).name
        file_results = [c for c in all_results if asc_file in str(asc_file)]

        sentences = parse_asc_file(asc_file)
        passive_tags = sum(1 for sent in sentences
                          for token in sent
                          if token.get('asc_tag') == 'PASSIVE')

        constructions_extracted = sum(1 for c in all_results
                                     if c.get('sent_idx', 0) > 0)  # Rough count

        print(f"\n{file_name}:")
        print(f"  Sentences: {len(sentences)}")
        print(f"  PASSIVE tags: {passive_tags}")
        print(f"  Constructions: extracted from this file batch")

    print(f"\n{'='*80}")
    print("✓ TAM×COMP EXTRACTION TEST COMPLETE")
    print('='*80)

    return all_results


def main():
    """Run test and return results."""
    results = test_extraction()

    # Save results summary
    print(f"\n💾 Results can be found in the terminal output above")
    print(f"📊 Total: {len(results)} passive constructions extracted")

    return results


if __name__ == "__main__":
    results = main()
