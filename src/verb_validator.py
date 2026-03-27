#!/usr/bin/env python3
"""
Verb Lemma Validator

Validates and corrects verb lemmas from learner corpus data.
Handles:
1. Common spelling errors in verb lemmas
2. Tokenization errors (words joined together)
3. Non-standard lemmatization artifacts

Usage:
    from verb_validator import validate_verb_lemma, validate_constructions
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# Common verb lemma corrections (learner errors and tokenization issues)
VERB_CORRECTIONS: Dict[str, str] = {
    # Spelling errors
    'highlightene': 'highlight',
    'instutationalize': 'institutionalize',
    'welcomme': 'welcome',
    'surpresse': 'suppress',
    'exggrate': 'exaggerate',
    'borderd': 'border',
    'hypnose': 'hypnotize',
    'summorise': 'summarise',
    'entitele': 'entitle',
    'convinse': 'convince',
    'consule': 'consult',
    'elecete': 'elect',
    'shattre': 'shatter',
    'proclame': 'proclaim',
    'selebrate': 'celebrate',
    'accure': 'occur',
    'headhunte': 'headhunt',
    'devorte': 'divorce',
    'preceede': 'precede',
    'threathene': 'threaten',
    'restablishe': 'reestablish',
    'challege': 'challenge',
    'supppose': 'suppose',
    'sensibilize': 'sensitize',
    'technolize': 'technologize',
    'bose': 'boss',

    # British/American variants (normalize to American)
    'recognise': 'recognize',
    'realise': 'realize',
    'organise': 'organize',
    'summarise': 'summarize',
    'criticise': 'criticize',
    'emphasise': 'emphasize',
    'analyse': 'analyze',
    'characterise': 'characterize',
    'specialise': 'specialize',
    'generalise': 'generalize',
    'legalise': 'legalize',
    'hospitalise': 'hospitalize',
    'centralise': 'centralize',
    'privatise': 'privatize',
    'socialise': 'socialize',
    'maximise': 'maximize',
    'minimise': 'minimize',
    'optimise': 'optimize',
    'incentivise': 'incentivize',

    # Common lemmatization errors
    'chose': 'choose',  # Past tense used as lemma
    'bore': 'bear',
    'wore': 'wear',
    'tore': 'tear',
    'swore': 'swear',
    'froze': 'freeze',
    'spoke': 'speak',
    'broke': 'break',
    'woke': 'wake',
    'drove': 'drive',
    'rode': 'ride',
    'wrote': 'write',
    'rose': 'rise',
    'arose': 'arise',
    'strove': 'strive',
    'throve': 'thrive',
}

# Regex pattern for tokenization errors (lowercase followed by uppercase)
TOKENIZATION_ERROR_PATTERN = re.compile(r'^([a-z]+)([A-Z][a-z]+.*)$')

# Pattern for words ending with common suffixes that got truncated
TRUNCATED_SUFFIX_PATTERN = re.compile(r'^(.+)(e)$')

# Valid verb suffixes for checking
VALID_VERB_ENDINGS = {
    'ate', 'ize', 'ify', 'en', 'ish',  # Common verb-forming suffixes
    'ed', 'ing',  # Should not appear in lemmas but sometimes do
}

# Invalid lemma patterns (filter these out)
INVALID_LEMMA_PATTERNS = [
    re.compile(r'^[A-Z]'),  # Starts with capital (proper noun)
    re.compile(r'\d'),  # Contains digits
    re.compile(r'[^a-zA-Z\-]'),  # Contains non-letter characters (except hyphen)
    re.compile(r'^.{1,2}$'),  # Too short (1-2 chars)
    re.compile(r'^.{25,}$'),  # Too long (likely tokenization error)
]


def clean_tokenization_error(lemma: str) -> Optional[str]:
    """
    Fix tokenization errors where words are joined.

    Examples:
        'crushedthi' -> 'crush'
        'gainedSome' -> 'gain'
        'seenmore' -> 'see'
        'glorifiedTo' -> 'glorify'
    """
    original = lemma

    # Check for camelCase errors (e.g., gainedSome)
    match = TOKENIZATION_ERROR_PATTERN.match(lemma)
    if match:
        first_part = match.group(1)
        lemma = first_part  # Use the first part

    # Try to extract base verb from past tense/participle forms
    # that appear joined with other text
    if lemma.endswith('ed') and len(lemma) > 4:
        base = lemma[:-2]
        # Handle doubled consonants (e.g., 'crushed' -> 'crush')
        if len(base) > 2 and base[-1] == base[-2] and base[-1] in 'bdgklmnprst':
            base = base[:-1]
        # Handle -ied -> -y (e.g., 'glorified' -> 'glorify')
        if base.endswith('i'):
            base = base[:-1] + 'y'
        if len(base) >= 3:
            return base

    if lemma.endswith('en') and len(lemma) > 4:
        base = lemma[:-2]
        if len(base) >= 3:
            return base

    # Check for 'seen', 'been' type patterns at start
    irregular_past = {
        'seen': 'see', 'been': 'be', 'gone': 'go', 'done': 'do',
        'given': 'give', 'taken': 'take', 'eaten': 'eat', 'beaten': 'beat',
        'written': 'write', 'driven': 'drive', 'risen': 'rise',
    }
    for past, base in irregular_past.items():
        if lemma.startswith(past) and len(lemma) > len(past):
            return base

    # If camelCase was detected, return the cleaned first part
    if original != lemma:
        return lemma

    # Check for all-lowercase joined words (harder to detect)
    if len(lemma) > 8:  # Suspiciously long for a verb lemma
        # Common verb patterns that might be joined
        for end_pos in range(4, min(len(lemma), 12)):
            potential_verb = lemma[:end_pos]
            remainder = lemma[end_pos:]
            # Check if remainder starts with common words/fragments
            if remainder.lower().startswith(('the', 'thi', 'some', 'more', 'to', 'in', 'on', 'is', 'it', 'an', 'a')):
                # Clean up the potential verb if it's in past tense form
                if potential_verb.endswith('ed') and len(potential_verb) > 4:
                    base = potential_verb[:-2]
                    if len(base) > 2 and base[-1] == base[-2] and base[-1] in 'bdgklmnprst':
                        base = base[:-1]
                    if base.endswith('i'):
                        base = base[:-1] + 'y'
                    return base
                return potential_verb

    return None


def validate_verb_lemma(lemma: str) -> Tuple[Optional[str], str]:
    """
    Validate and normalize a verb lemma.

    Args:
        lemma: The verb lemma to validate

    Returns:
        Tuple of (normalized_lemma, status)
        - status: 'valid', 'corrected', 'filtered', 'tokenization_error'
    """
    if not lemma or not isinstance(lemma, str):
        return None, 'filtered'

    lemma = lemma.strip().lower()

    # Check for invalid patterns
    for pattern in INVALID_LEMMA_PATTERNS:
        if pattern.search(lemma):
            return None, 'filtered'

    # Check if it's a known correction
    if lemma in VERB_CORRECTIONS:
        return VERB_CORRECTIONS[lemma], 'corrected'

    # Check for tokenization errors
    cleaned = clean_tokenization_error(lemma)
    if cleaned and cleaned != lemma:
        # Recursively validate the cleaned version
        final, status = validate_verb_lemma(cleaned)
        if final:
            return final, 'tokenization_error'

    # Basic validation passed
    return lemma, 'valid'


def validate_constructions(
    constructions: List[Dict],
    lemma_field: str = 'head_lemma'
) -> Tuple[List[Dict], Dict]:
    """
    Validate verb lemmas in a list of constructions.

    Args:
        constructions: List of construction dictionaries
        lemma_field: Field name containing the verb lemma

    Returns:
        Tuple of (validated_constructions, statistics)
    """
    validated = []
    stats = {
        'total': len(constructions),
        'valid': 0,
        'corrected': 0,
        'filtered': 0,
        'tokenization_errors': 0,
        'corrections': {},  # original -> corrected
        'filtered_lemmas': set(),
    }

    for const in constructions:
        original_lemma = const.get(lemma_field, '')
        normalized, status = validate_verb_lemma(original_lemma)

        if normalized:
            new_const = const.copy()
            new_const[lemma_field] = normalized
            validated.append(new_const)

            if status == 'valid':
                stats['valid'] += 1
            elif status == 'corrected':
                stats['corrected'] += 1
                stats['corrections'][original_lemma] = normalized
            elif status == 'tokenization_error':
                stats['tokenization_errors'] += 1
                stats['corrections'][original_lemma] = normalized
        else:
            stats['filtered'] += 1
            stats['filtered_lemmas'].add(original_lemma)

    # Convert set to sorted list for reporting
    stats['filtered_lemmas'] = sorted(stats['filtered_lemmas'])

    return validated, stats


def generate_validation_report(stats: Dict) -> str:
    """Generate a human-readable validation report."""
    lines = [
        "=" * 70,
        "VERB LEMMA VALIDATION REPORT",
        "=" * 70,
        "",
        f"Total constructions processed: {stats['total']:,}",
        f"Valid (unchanged): {stats['valid']:,} ({stats['valid']/stats['total']*100:.1f}%)",
        f"Corrected: {stats['corrected']:,} ({stats['corrected']/stats['total']*100:.1f}%)",
        f"Tokenization errors fixed: {stats['tokenization_errors']:,}",
        f"Filtered out: {stats['filtered']:,} ({stats['filtered']/stats['total']*100:.1f}%)",
        "",
    ]

    if stats['corrections']:
        lines.append("-" * 70)
        lines.append("CORRECTIONS APPLIED:")
        lines.append("-" * 70)
        for orig, corrected in sorted(stats['corrections'].items()):
            lines.append(f"  {orig} -> {corrected}")
        lines.append("")

    if stats['filtered_lemmas']:
        lines.append("-" * 70)
        lines.append("FILTERED LEMMAS:")
        lines.append("-" * 70)
        for lemma in stats['filtered_lemmas'][:50]:  # Show first 50
            lines.append(f"  {lemma}")
        if len(stats['filtered_lemmas']) > 50:
            lines.append(f"  ... and {len(stats['filtered_lemmas']) - 50} more")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the validator
    test_cases = [
        'use',  # valid
        'highlightene',  # spelling error
        'crushedthi',  # tokenization error
        'gainedSome',  # camelCase tokenization
        'seenmore',  # irregular + joined
        'glorifiedTo',  # camelCase with -ified
        'recognise',  # British spelling
        'chose',  # past tense as lemma
        '123verb',  # invalid (contains digits)
        'a',  # too short
        'consider',  # common valid verb
        'instutationalize',  # spelling error
    ]

    print("Verb Lemma Validator Test")
    print("-" * 40)
    for lemma in test_cases:
        result, status = validate_verb_lemma(lemma)
        result_str = result if result else 'None'
        print(f"  {lemma:20} -> {result_str:15} ({status})")
