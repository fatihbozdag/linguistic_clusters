#!/usr/bin/env python3
"""
Preposition Normalization Module

Corrects common misspellings and tokenization errors in prepositions
extracted from learner corpora (ICLE). This is critical for data quality
in the ConstructionMiner pipeline.

Learner corpora contain systematic spelling errors that create spurious
patterns (e.g., pp_form_NP instead of pp_from_NP). This module normalizes
prepositions to their standard forms.

Categories of errors handled:
1. Simple typos (form → from, amangst → amongst)
2. Tokenization errors (tooin → filter, notin → filter)
3. Non-standard spellings (thru → through)
4. Invalid prepositions (filter entirely)
"""

from typing import Dict, List, Tuple, Optional, Set
import re
from collections import Counter


# ============================================================================
# Preposition Correction Dictionary
# ============================================================================

# Mapping of typo → correct form
# Source: Analysis of patterns_all_metrics.csv from ICLE corpus
PREP_CORRECTIONS: Dict[str, str] = {
    # Simple typos
    'form': 'from',
    'fro': 'from',
    'frm': 'from',
    'fom': 'from',

    'amangst': 'amongst',
    'amoungst': 'amongst',
    'amongts': 'amongst',

    'thorughout': 'throughout',
    'throghout': 'throughout',
    'throught': 'through',
    'thru': 'through',
    'trough': 'through',
    'throug': 'through',

    'whithin': 'within',
    'witin': 'within',
    'withing': 'within',

    'whit': 'with',
    'wiht': 'with',
    'wth': 'with',
    'wit': 'with',

    'withiout': 'without',
    'whitout': 'without',
    'withou': 'without',
    'whithout': 'without',

    'arround': 'around',
    'aroud': 'around',
    'aorund': 'around',

    'inspite': 'despite',  # "in spite of" → normalize to "despite"

    'tot': 'to',
    'ot': 'to',

    'upto': 'up',  # "up to" → just use "up"

    'ion': 'in',  # likely OCR/tokenization error

    'towars': 'towards',
    'towardas': 'towards',
    'toward': 'towards',  # normalize US → UK spelling

    'betwee': 'between',
    'beetween': 'between',

    'againts': 'against',
    'agains': 'against',

    'untill': 'until',
    'util': 'until',

    'accross': 'across',
    'acros': 'across',

    'behing': 'behind',
    'behinde': 'behind',

    'besid': 'beside',
    'bside': 'beside',

    'beneth': 'beneath',
    'beneith': 'beneath',

    'abov': 'above',
    'abve': 'above',

    'belo': 'below',
    'belwo': 'below',

    'alon': 'along',
    'alnog': 'along',

    'nea': 'near',
    'naer': 'near',

    'outsid': 'outside',
    'outisde': 'outside',

    'insid': 'inside',
    'insdie': 'inside',

    'beyon': 'beyond',
    'beyound': 'beyond',

    'oposite': 'opposite',
    'opposit': 'opposite',

    'beneat': 'beneath',

    'througout': 'throughout',

    'althrough': 'although',  # Will be filtered - not a preposition
}

# Prepositions that should be filtered entirely (not real prepositions)
# These are tokenization errors, abbreviations, or parsing artifacts
INVALID_PREPS: Set[str] = {
    # Tokenization errors (word + preposition merged)
    'tooin',
    'notin',
    'forwhy',
    'languageas',
    'appropriatelyas',
    'againstin',
    'withapart',
    'notwithstanding',  # Keep if you want complex preps, remove otherwise

    # Abbreviations
    'ie',
    'eg',

    # Not prepositions
    'a',  # article misidentified
    'o',  # parsing error
    'although',  # conjunction
    'althrough',  # misspelled conjunction

    # Other parsing artifacts
    'per',  # Keep? Latin preposition, uncommon
}

# Standard English prepositions (for validation)
VALID_PREPOSITIONS: Set[str] = {
    # Simple prepositions
    'about', 'above', 'across', 'after', 'against', 'along', 'alongside',
    'amid', 'amidst', 'among', 'amongst', 'around', 'as', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'by',
    'concerning',
    'despite', 'down', 'during',
    'except', 'excluding',
    'following', 'for', 'from',
    'in', 'inside', 'into',
    'like',
    'near', 'notwithstanding',
    'of', 'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over',
    'past', 'pending', 'per', 'plus',
    'regarding', 'round',
    'since',
    'than', 'through', 'throughout', 'till', 'to', 'toward', 'towards',
    'under', 'underneath', 'unlike', 'until', 'up', 'upon',
    'versus', 'via',
    'with', 'within', 'without',
}


# ============================================================================
# Normalization Functions
# ============================================================================

def normalize_preposition(prep: str) -> Optional[str]:
    """
    Normalize a single preposition to its standard form.

    Args:
        prep: Raw preposition string (may contain typos)

    Returns:
        Normalized preposition, or None if should be filtered

    Examples:
        >>> normalize_preposition('form')
        'from'
        >>> normalize_preposition('tooin')
        None
        >>> normalize_preposition('with')
        'with'
    """
    if not prep:
        return None

    prep_lower = prep.lower().strip()

    # Check if it should be filtered entirely
    if prep_lower in INVALID_PREPS:
        return None

    # Check if it needs correction
    if prep_lower in PREP_CORRECTIONS:
        corrected = PREP_CORRECTIONS[prep_lower]
        # The correction might also be invalid (e.g., 'althrough' → 'although')
        if corrected in INVALID_PREPS or corrected not in VALID_PREPOSITIONS:
            return None
        return corrected

    # Check if it's a valid preposition
    if prep_lower in VALID_PREPOSITIONS:
        return prep_lower

    # Unknown preposition - could be a new typo or parsing error
    # Return as-is but flag for review
    return prep_lower


def normalize_complement(comp: str) -> Optional[str]:
    """
    Normalize a complement type string (e.g., pp_form_NP → pp_from_NP).

    Args:
        comp: Complement type string (TAM,COMP format or just COMP)

    Returns:
        Normalized complement, or None if should be filtered

    Examples:
        >>> normalize_complement('pp_form_NP')
        'pp_from_NP'
        >>> normalize_complement('pp_tooin_NP')
        None
        >>> normalize_complement('by_NP[ENT=PERSON]')
        'by_NP'  # NER patterns collapsed to by_NP
    """
    if not comp:
        return None

    # Handle special complement types that don't need normalization
    if comp in ('Ø', 'to_VP', 'by_NP'):
        return comp

    # Handle legacy by_NP with entity (by_NP[ENT=X]) - collapse to by_NP
    # NER entity types removed for consistent granularity with other PP types
    if comp.startswith('by_NP[ENT='):
        return 'by_NP'

    # Handle prepositional phrases (pp_X_NP)
    if comp.startswith('pp_') and comp.endswith('_NP'):
        # Extract preposition
        prep = comp[3:-3]  # Remove 'pp_' and '_NP'

        # Normalize the preposition
        normalized_prep = normalize_preposition(prep)

        if normalized_prep is None:
            return None

        return f'pp_{normalized_prep}_NP'

    # Unknown format - return as-is
    return comp


def normalize_pattern(pattern: str) -> Optional[str]:
    """
    Normalize a full TAM×COMP pattern string.

    Args:
        pattern: Full pattern string (e.g., "modal-be,pp_form_NP")

    Returns:
        Normalized pattern, or None if should be filtered

    Examples:
        >>> normalize_pattern('modal-be,pp_form_NP')
        'modal-be,pp_from_NP'
        >>> normalize_pattern('pres-be,pp_tooin_NP')
        None
    """
    if not pattern:
        return None

    # Split into TAM and COMP
    parts = pattern.split(',', 1)

    if len(parts) != 2:
        return pattern  # Unknown format

    tam, comp = parts

    # Normalize the complement
    normalized_comp = normalize_complement(comp)

    if normalized_comp is None:
        return None

    return f'{tam},{normalized_comp}'


def normalize_construction(construction: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Normalize a construction dict in place.

    Args:
        construction: PassiveConstruction dict with 'pattern' and 'comp' keys

    Returns:
        Tuple of (normalized_construction or None, correction_note or None)
        Returns None for construction if it should be filtered.

    Examples:
        >>> const = {'pattern': 'modal-be,pp_form_NP', 'comp': 'pp_form_NP'}
        >>> normalized, note = normalize_construction(const)
        >>> normalized['pattern']
        'modal-be,pp_from_NP'
        >>> note
        'form → from'
    """
    if not construction:
        return None, None

    original_comp = construction.get('comp', '')
    original_pattern = construction.get('pattern', '')

    # Normalize complement
    normalized_comp = normalize_complement(original_comp)

    if normalized_comp is None:
        return None, f'Filtered: {original_comp}'

    # Normalize pattern
    normalized_pattern = normalize_pattern(original_pattern)

    if normalized_pattern is None:
        return None, f'Filtered: {original_pattern}'

    # Check if any changes were made
    if normalized_comp == original_comp and normalized_pattern == original_pattern:
        return construction, None

    # Create correction note
    correction_note = None
    if normalized_comp != original_comp:
        # Extract the preposition change
        if original_comp.startswith('pp_') and normalized_comp.startswith('pp_'):
            old_prep = original_comp[3:-3] if original_comp.endswith('_NP') else original_comp[3:]
            new_prep = normalized_comp[3:-3] if normalized_comp.endswith('_NP') else normalized_comp[3:]
            correction_note = f'{old_prep} → {new_prep}'

    # Update construction
    construction['comp'] = normalized_comp
    construction['pattern'] = normalized_pattern
    construction['_normalized'] = True
    if correction_note:
        construction['_correction'] = correction_note

    return construction, correction_note


# ============================================================================
# Batch Processing
# ============================================================================

def normalize_constructions(constructions: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Normalize all constructions and collect statistics.

    Args:
        constructions: List of PassiveConstruction dicts

    Returns:
        Tuple of (normalized_constructions, stats)

        stats contains:
        - total: Total input constructions
        - normalized: Constructions with corrections applied
        - filtered: Constructions removed
        - unchanged: Constructions with no changes needed
        - corrections: Counter of specific corrections made
        - filtered_patterns: Counter of filtered pattern types
    """
    normalized = []

    stats = {
        'total': len(constructions),
        'normalized': 0,
        'filtered': 0,
        'unchanged': 0,
        'corrections': Counter(),
        'filtered_patterns': Counter()
    }

    for const in constructions:
        result, note = normalize_construction(const.copy())

        if result is None:
            stats['filtered'] += 1
            # Track what was filtered
            pattern = const.get('pattern', 'unknown')
            stats['filtered_patterns'][pattern] += 1
        elif note:
            stats['normalized'] += 1
            stats['corrections'][note] += 1
            normalized.append(result)
        else:
            stats['unchanged'] += 1
            normalized.append(result)

    return normalized, stats


def generate_correction_report(stats: Dict) -> str:
    """
    Generate a human-readable correction report.

    Args:
        stats: Statistics dict from normalize_constructions()

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "PREPOSITION NORMALIZATION REPORT",
        "=" * 70,
        "",
        f"Total constructions processed: {stats['total']:,}",
        f"  - Unchanged (valid): {stats['unchanged']:,} ({stats['unchanged']/stats['total']*100:.1f}%)",
        f"  - Normalized (corrected): {stats['normalized']:,} ({stats['normalized']/stats['total']*100:.1f}%)",
        f"  - Filtered (invalid): {stats['filtered']:,} ({stats['filtered']/stats['total']*100:.1f}%)",
        "",
    ]

    if stats['corrections']:
        lines.append("-" * 70)
        lines.append("Corrections Applied:")
        lines.append("-" * 70)
        for correction, count in stats['corrections'].most_common():
            lines.append(f"  {correction}: {count:,}")
        lines.append("")

    if stats['filtered_patterns']:
        lines.append("-" * 70)
        lines.append("Filtered Patterns:")
        lines.append("-" * 70)
        for pattern, count in stats['filtered_patterns'].most_common(20):
            lines.append(f"  {pattern}: {count:,}")
        if len(stats['filtered_patterns']) > 20:
            lines.append(f"  ... and {len(stats['filtered_patterns']) - 20} more")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================================
# Pattern-Level Normalization (for metrics aggregation)
# ============================================================================

def normalize_pattern_metrics(metrics: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Dict]:
    """
    Normalize pattern keys in metrics dict and merge duplicates.

    When normalizing patterns, some previously distinct patterns may become
    identical (e.g., pp_form_NP and pp_from_NP both become pp_from_NP).
    This function merges them by summing token counts.

    Args:
        metrics: Dict mapping pattern → metrics dict

    Returns:
        Tuple of (normalized_metrics, merge_log)
    """
    normalized = {}
    merge_log = {}

    for pattern, pattern_metrics in metrics.items():
        # Normalize the pattern
        norm_pattern = normalize_pattern(pattern)

        if norm_pattern is None:
            # Track filtered patterns
            merge_log[pattern] = 'FILTERED'
            continue

        if norm_pattern in normalized:
            # Merge with existing
            existing = normalized[norm_pattern]
            existing['n_tokens'] += pattern_metrics.get('n_tokens', 0)
            existing['n_docs'] = max(existing['n_docs'], pattern_metrics.get('n_docs', 0))
            # For other metrics, take weighted average or max
            # (simplified: just keep the existing values)
            merge_log[pattern] = f'MERGED → {norm_pattern}'
        else:
            normalized[norm_pattern] = pattern_metrics.copy()
            if norm_pattern != pattern:
                merge_log[pattern] = f'RENAMED → {norm_pattern}'

    return normalized, merge_log


# ============================================================================
# Testing
# ============================================================================

def test_normalizer():
    """Test normalization functions."""
    print("=" * 70)
    print("Testing Preposition Normalizer")
    print("=" * 70)

    # Test preposition normalization
    print("\n1. Testing preposition normalization:")
    test_preps = [
        ('form', 'from'),
        ('fro', 'from'),
        ('amangst', 'amongst'),
        ('thorughout', 'throughout'),
        ('with', 'with'),
        ('tooin', None),
        ('ie', None),
        ('unknown_prep', 'unknown_prep'),
    ]

    for prep, expected in test_preps:
        result = normalize_preposition(prep)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{prep}' → '{result}' (expected: '{expected}')")

    # Test complement normalization
    print("\n2. Testing complement normalization:")
    test_comps = [
        ('pp_form_NP', 'pp_from_NP'),
        ('pp_amangst_NP', 'pp_amongst_NP'),
        ('pp_tooin_NP', None),
        ('by_NP', 'by_NP'),
        ('by_NP[ENT=PERSON]', 'by_NP'),  # NER patterns collapsed to by_NP
        ('Ø', 'Ø'),
    ]

    for comp, expected in test_comps:
        result = normalize_complement(comp)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{comp}' → '{result}' (expected: '{expected}')")

    # Test pattern normalization
    print("\n3. Testing pattern normalization:")
    test_patterns = [
        ('modal-be,pp_form_NP', 'modal-be,pp_from_NP'),
        ('pres-be,pp_tooin_NP', None),
        ('past-be,by_NP', 'past-be,by_NP'),
    ]

    for pattern, expected in test_patterns:
        result = normalize_pattern(pattern)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{pattern}' → '{result}' (expected: '{expected}')")

    # Test batch processing
    print("\n4. Testing batch normalization:")
    test_constructions = [
        {'pattern': 'modal-be,pp_form_NP', 'comp': 'pp_form_NP'},
        {'pattern': 'pres-be,pp_with_NP', 'comp': 'pp_with_NP'},
        {'pattern': 'past-be,pp_tooin_NP', 'comp': 'pp_tooin_NP'},
    ]

    normalized, stats = normalize_constructions(test_constructions)
    print(f"  Total: {stats['total']}")
    print(f"  Normalized: {stats['normalized']}")
    print(f"  Filtered: {stats['filtered']}")
    print(f"  Unchanged: {stats['unchanged']}")

    print("\n5. Correction report:")
    print(generate_correction_report(stats))

    print("\n✓ Preposition normalizer ready!")


if __name__ == "__main__":
    test_normalizer()
