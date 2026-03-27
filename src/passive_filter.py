#!/usr/bin/env python3
"""
Passive Construction Filtering for LC Framework Analysis

Filters out:
- TAM: unknown, get-be
- Extraposed it-passives ("It is known that...")
- Non-finite constructions
- Constructions with missing required fields

Expected filtering rate: ~40% of constructions removed
"""

from typing import Dict, List, Tuple


# Invalid TAM categories (to be filtered out)
INVALID_TAM = {'unknown', 'get-be'}

# Valid TAM categories (explicit whitelist)
VALID_TAM = {'pres-be', 'past-be', 'perf-be', 'modal-be', 'prog-be'}


def is_extraposed_it_passive(construction: Dict) -> bool:
    """
    Detect extraposed it-passives.

    Pattern: "It is/was VERB that/whether/if CLAUSE"
    Example: "It is known that X", "It was found that Y"

    These are syntactically different from regular passives:
    - "It" is expletive (non-referential)
    - Real subject is the clausal complement
    - Common in academic writing but not true argument structure passives

    Args:
        construction: PassiveConstruction dict with keys:
            - subjtype: Subject type classification
            - subject_text: Actual subject text
            - surface: Full construction surface

    Returns:
        True if extraposed it-passive, False otherwise

    Examples:
        >>> const = {'subjtype': 'PRON', 'subject_text': 'it',
        ...          'surface': 'it is known that language is complex'}
        >>> is_extraposed_it_passive(const)
        True

        >>> const = {'subjtype': 'PRON', 'subject_text': 'it',
        ...          'surface': 'it was destroyed by fire'}
        >>> is_extraposed_it_passive(const)
        False
    """
    # Check subject is pronoun (first filter)
    if construction.get('subjtype') != 'PRON':
        return False

    # Check subject is specifically "it" (expletive)
    subject_text = construction.get('subject_text', '').lower().strip()
    if subject_text != 'it':
        return False

    # Check for clausal complement markers in surface
    surface = construction.get('surface', '').lower()

    # Common extraposition markers
    # Note: Must appear AFTER the passive verb (in second half of sentence)
    markers = [
        ' that ',    # "it is known that X"
        ' whether ', # "it was unclear whether Y"
        ' if ',      # "it is uncertain if Z"
        ' to ',      # "it was difficult to V"
        ' how ',     # "it is unknown how W"
        ' what ',    # "it is unclear what Q"
        ' when ',    # "it is undecided when R"
        ' where ',   # "it is unknown where S"
        ' why '      # "it is unclear why T"
    ]

    # Check if any marker appears in the surface
    for marker in markers:
        if marker in surface:
            # Heuristic: marker should appear after "it is/was VERB"
            # Check if marker appears after position 10 (allows for "it is/was VERB")
            marker_pos = surface.find(marker)

            # If marker found and not at the very beginning, likely extraposed
            if marker_pos > 8:  # After "it is VERB" (minimum 8 chars)
                return True

    return False


def has_required_fields(construction: Dict) -> bool:
    """
    Check if construction has all required fields.

    Required fields for analysis:
    - surface: Full construction text
    - tam: TAM category
    - comp: Complement type
    - subjtype: Subject type
    - head_lemma: Main verb lemma
    - pattern: Complete TAM×COMP pattern

    Args:
        construction: PassiveConstruction dict

    Returns:
        True if all required fields present, False otherwise
    """
    required_fields = ['surface', 'tam', 'comp', 'subjtype', 'head_lemma', 'pattern']
    return all(field in construction and construction[field] is not None
               for field in required_fields)


def is_valid_passive(construction: Dict) -> bool:
    """
    Apply all filtering rules to determine if construction is valid.

    Filtering criteria:
    1. Has all required fields
    2. TAM is in VALID_TAM (pres-be, past-be, perf-be, modal-be, prog-be)
    3. Not an extraposed it-passive

    Args:
        construction: PassiveConstruction dict

    Returns:
        True if construction passes all filters, False otherwise

    Examples:
        >>> const = {'surface': 'the book was written', 'tam': 'past-be',
        ...          'comp': 'Ø', 'subjtype': 'DEF_NP', 'head_lemma': 'write',
        ...          'pattern': '[SUBJTYPE=DEF_NP],past-be,Ø'}
        >>> is_valid_passive(const)
        True

        >>> const = {'surface': 'to be examined', 'tam': 'unknown',
        ...          'comp': 'Ø', 'subjtype': 'DEF_NP', 'head_lemma': 'examine',
        ...          'pattern': '[SUBJTYPE=DEF_NP],unknown,Ø'}
        >>> is_valid_passive(const)
        False
    """
    # Check 1: Required fields
    if not has_required_fields(construction):
        return False

    # Check 2: TAM must be valid
    tam = construction['tam']
    if tam not in VALID_TAM:
        return False

    # Check 3: Not extraposed it-passive
    if is_extraposed_it_passive(construction):
        return False

    return True


def filter_constructions(constructions: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Filter constructions and collect detailed statistics.

    Applies all filtering rules and tracks why each construction
    was filtered out (for analysis and debugging).

    Args:
        constructions: List of PassiveConstruction dicts

    Returns:
        Tuple of (valid_constructions, filter_stats)

        filter_stats contains:
        - total: Total constructions processed
        - valid: Number passing all filters
        - filtered_*: Counts for each filter type
        - by_tam/comp/subjtype: Distributions of valid constructions

    Example:
        >>> constructions = [...]
        >>> valid, stats = filter_constructions(constructions)
        >>> print(f"Kept {stats['valid']}/{stats['total']} constructions")
        >>> print(f"Filtered {stats['filtered_invalid_tam']} for invalid TAM")
    """
    valid = []

    # Initialize statistics
    stats = {
        'total': len(constructions),
        'valid': 0,
        'filtered_invalid_tam': 0,
        'filtered_extraposed': 0,
        'filtered_missing_fields': 0,
        'by_tam': {},
        'by_comp': {},
        'by_subjtype': {},
        'invalid_tam_breakdown': {}  # Track which invalid TAMs were filtered
    }

    for const in constructions:
        # Check 1: Required fields
        if not has_required_fields(const):
            stats['filtered_missing_fields'] += 1
            continue

        # Check 2: TAM validity
        tam = const['tam']
        if tam not in VALID_TAM:
            stats['filtered_invalid_tam'] += 1
            # Track which invalid TAM types
            stats['invalid_tam_breakdown'][tam] = stats['invalid_tam_breakdown'].get(tam, 0) + 1
            continue

        # Check 3: Extraposed it-passive
        if is_extraposed_it_passive(const):
            stats['filtered_extraposed'] += 1
            continue

        # Valid construction!
        valid.append(const)
        stats['valid'] += 1

        # Collect distributions of valid constructions
        comp = const['comp']
        subjtype = const['subjtype']

        stats['by_tam'][tam] = stats['by_tam'].get(tam, 0) + 1
        stats['by_comp'][comp] = stats['by_comp'].get(comp, 0) + 1
        stats['by_subjtype'][subjtype] = stats['by_subjtype'].get(subjtype, 0) + 1

    return valid, stats


def print_filter_statistics(stats: Dict):
    """
    Print detailed filtering statistics.

    Args:
        stats: Statistics dict from filter_constructions()
    """
    print("\n" + "="*70)
    print("FILTERING STATISTICS")
    print("="*70)

    # Overall numbers
    total = stats['total']
    valid = stats['valid']
    filtered = total - valid

    print(f"\nTotal constructions: {total:,}")
    print(f"Valid constructions: {valid:,} ({valid/total*100:.1f}%)")
    print(f"Filtered out: {filtered:,} ({filtered/total*100:.1f}%)")

    # Breakdown of filtered constructions
    print(f"\nFiltering breakdown:")
    print(f"  - Invalid TAM (unknown/get-be): {stats['filtered_invalid_tam']:,} "
          f"({stats['filtered_invalid_tam']/total*100:.1f}%)")

    if stats.get('invalid_tam_breakdown'):
        print(f"    Breakdown by TAM type:")
        for tam, count in sorted(stats['invalid_tam_breakdown'].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"      • {tam}: {count:,}")

    print(f"  - Extraposed it-passives: {stats['filtered_extraposed']:,} "
          f"({stats['filtered_extraposed']/total*100:.1f}%)")
    print(f"  - Missing required fields: {stats['filtered_missing_fields']:,} "
          f"({stats['filtered_missing_fields']/total*100:.1f}%)")

    # Valid construction distributions
    if stats['by_tam']:
        print(f"\n" + "-"*70)
        print("Valid TAM Distribution:")
        print("-"*70)
        for tam in sorted(stats['by_tam'].keys()):
            count = stats['by_tam'][tam]
            pct = count / valid * 100
            print(f"  {tam:<15} {count:>6,} ({pct:>5.1f}%)")

    if stats['by_comp']:
        print(f"\n" + "-"*70)
        print("COMP Distribution (Top 10):")
        print("-"*70)
        sorted_comps = sorted(stats['by_comp'].items(),
                             key=lambda x: x[1], reverse=True)[:10]
        for comp, count in sorted_comps:
            pct = count / valid * 100
            print(f"  {comp:<30} {count:>6,} ({pct:>5.1f}%)")

    if stats['by_subjtype']:
        print(f"\n" + "-"*70)
        print("SUBJTYPE Distribution:")
        print("-"*70)
        for subj in sorted(stats['by_subjtype'].keys()):
            count = stats['by_subjtype'][subj]
            pct = count / valid * 100
            print(f"  {subj:<15} {count:>6,} ({pct:>5.1f}%)")

    print("\n" + "="*70)


# ============================================================================
# Testing
# ============================================================================

def test_filter():
    """Test filtering functions."""
    print("="*70)
    print("Testing Passive Filter")
    print("="*70)

    # Test extraposed detection
    print("\n1. Testing extraposed it-passive detection:")

    test_cases = [
        {
            'name': 'Extraposed (that-clause)',
            'const': {
                'subjtype': 'PRON',
                'subject_text': 'it',
                'surface': 'it is known that language is complex',
                'tam': 'pres-be'
            },
            'expected': True
        },
        {
            'name': 'Not extraposed (regular it-passive)',
            'const': {
                'subjtype': 'PRON',
                'subject_text': 'it',
                'surface': 'it was destroyed by fire',
                'tam': 'past-be'
            },
            'expected': False
        },
        {
            'name': 'Not extraposed (not "it" subject)',
            'const': {
                'subjtype': 'DEF_NP',
                'subject_text': 'the book',
                'surface': 'the book was written that year',
                'tam': 'past-be'
            },
            'expected': False
        }
    ]

    for test in test_cases:
        result = is_extraposed_it_passive(test['const'])
        status = "✓" if result == test['expected'] else "✗"
        print(f"  {status} {test['name']}: {result} (expected: {test['expected']})")

    # Test overall filtering
    print("\n2. Testing overall filtering:")

    test_constructions = [
        {'surface': 'book was written', 'tam': 'past-be', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'head_lemma': 'write',
         'pattern': '[SUBJTYPE=DEF_NP],past-be,Ø'},
        {'surface': 'to be examined', 'tam': 'unknown', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'head_lemma': 'examine',
         'pattern': '[SUBJTYPE=DEF_NP],unknown,Ø'},
        {'surface': 'to get married', 'tam': 'get-be', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'head_lemma': 'marry',
         'pattern': '[SUBJTYPE=DEF_NP],get-be,Ø'},
    ]

    valid, stats = filter_constructions(test_constructions)

    print(f"  Total: {stats['total']}")
    print(f"  Valid: {stats['valid']}")
    print(f"  Filtered (invalid TAM): {stats['filtered_invalid_tam']}")

    print("\n✓ Filter module ready!")


if __name__ == "__main__":
    test_filter()
