#!/usr/bin/env python3
"""
Analyze ASC-analyzer output to understand what it produces.
This script helps explore the format and content of ASC-tagged files and CSV summaries.
"""

import csv
from collections import Counter
from pathlib import Path


def parse_ascinfo_file(filepath):
    """
    Parse an ASCinfo.txt file and extract statistics.

    Returns:
        dict: Statistics including sentences, tokens, ASC tags, and sample data
    """
    sentences = []
    current_sentence = []
    asc_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and sentence markers
            if not line or line.startswith('#'):
                if current_sentence:  # End of sentence
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Parse token line: token_num \t token \t lemma \t ASC_tag (optional)
            parts = line.split('\t')
            if len(parts) >= 3:
                token_data = {
                    'num': parts[0],
                    'token': parts[1],
                    'lemma': parts[2],
                    'asc_tag': parts[3] if len(parts) > 3 and parts[3] else '-'
                }
                current_sentence.append(token_data)

                # Collect ASC tags
                if len(parts) > 3 and parts[3]:
                    asc_tags.append(parts[3])

        # Don't forget the last sentence
        if current_sentence:
            sentences.append(current_sentence)

    # Count tokens
    total_tokens = sum(len(sent) for sent in sentences)

    # Count ASC tag types
    asc_counter = Counter(asc_tags)

    return {
        'sentences': sentences,
        'total_sentences': len(sentences),
        'total_tokens': total_tokens,
        'asc_tags': asc_counter,
        'unique_asc_tags': sorted(asc_counter.keys())
    }


def analyze_csv_summary(csv_filepath):
    """
    Analyze the CSV summary file and identify passive-related metrics.

    Returns:
        dict: CSV analysis including columns and passive metrics
    """
    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    # Identify passive-related columns
    passive_columns = [col for col in columns if 'PASSIVE' in col.upper()]

    # Get summary stats for key metrics
    summary_stats = {}
    for col in ['clauseCount', 'clauseCountNoBe', 'ascMATTR11', 'PASSIVE_Prop']:
        if col in columns:
            values = [float(row[col]) for row in rows if row[col]]
            if values:
                summary_stats[col] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values)
                }

    return {
        'columns': columns,
        'num_columns': len(columns),
        'num_rows': len(rows),
        'passive_columns': passive_columns,
        'summary_stats': summary_stats,
        'sample_rows': rows[:3]  # First 3 rows for inspection
    }


def print_analysis(ascinfo_file, csv_file):
    """
    Print comprehensive analysis of ASC-analyzer outputs.
    """
    print("=" * 80)
    print(f"=== Analyzing {Path(ascinfo_file).name} ===")
    print("=" * 80)

    # Analyze ASCinfo file
    asc_data = parse_ascinfo_file(ascinfo_file)

    print(f"\nTotal sentences: {asc_data['total_sentences']}")
    print(f"Total tokens: {asc_data['total_tokens']}")

    print(f"\nASC Tags found ({len(asc_data['unique_asc_tags'])} unique types):")
    for tag in asc_data['unique_asc_tags']:
        count = asc_data['asc_tags'][tag]
        print(f"  - {tag}: {count} occurrences")

    print(f"\nSample (first 3 sentences):")
    for i, sent in enumerate(asc_data['sentences'][:3], 1):
        print(f"\nSentence {i}:")
        for token in sent:
            print(f"  {token['num']:>3}  {token['token']:<15}  {token['lemma']:<15}  {token['asc_tag']}")

    print("\n" + "=" * 80)
    print("=== CSV Summary Analysis ===")
    print("=" * 80)

    # Analyze CSV
    csv_data = analyze_csv_summary(csv_file)

    print(f"\nNumber of rows: {csv_data['num_rows']}")
    print(f"Number of columns: {csv_data['num_columns']}")

    print(f"\nAll columns ({csv_data['num_columns']} total):")
    for i, col in enumerate(csv_data['columns'], 1):
        print(f"  {i:>2}. {col}")

    print(f"\nPassive-related metrics ({len(csv_data['passive_columns'])} columns):")
    for col in csv_data['passive_columns']:
        print(f"  - {col}")

    print("\nSummary statistics for key metrics:")
    for metric, stats in csv_data['summary_stats'].items():
        print(f"\n  {metric}:")
        print(f"    Min: {stats['min']:.4f}")
        print(f"    Max: {stats['max']:.4f}")
        print(f"    Avg: {stats['avg']:.4f}")

    print("\n" + "=" * 80)
    print("=== ASC Type Descriptions ===")
    print("=" * 80)
    print("""
Based on the output, here are the ASC (Argument Structure Construction) types found:

  - ATTR:        Attributive constructions (e.g., "It is time...")
  - PASSIVE:     Passive voice constructions (e.g., "is dominated", "is based")
  - TRAN-S:      Transitive constructions with simple arguments
  - INTRAN-S:    Intransitive constructions with simple arguments
  - INTRAN-MOT:  Intransitive motion constructions
  - TRAN-RES:    Transitive resultative constructions
  - CAUS-MOT:    Caused-motion constructions
  - DITRAN:      Ditransitive constructions (e.g., "give X to Y")
  - INTRAN-RES:  Intransitive resultative constructions

These constructions represent different argument structure patterns that verbs
can participate in, based on Construction Grammar theory.
    """)

    print("=" * 80)


def main():
    # File paths
    base_dir = Path("/Users/fatihbozdag/Documents/ConstructionMiner-Clean")
    ascinfo_file = base_dir / "test_data" / "BGSU1001_ASCinfo.txt"
    csv_file = base_dir / "asc_output" / "summary.csv"

    # Run analysis
    print_analysis(ascinfo_file, csv_file)

    print("\n✓ Analysis complete!")
    print("\nNext steps:")
    print("  1. Review the ASC tag types and their frequencies")
    print("  2. Examine the CSV metrics to understand available measures")
    print("  3. Consider which metrics are most relevant for your research")
    print("  4. Explore the other ASCinfo files to see variation across texts")


if __name__ == "__main__":
    main()
