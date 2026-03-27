#!/usr/bin/env python3
"""
TAM×COMP Classification Module

Deterministic classification functions for:
- TAM (Tense-Aspect-Modality): pres-be, past-be, perf-be, modal-be, prog-be, get-be
- COMP (Complement type): by_NP, pp_X_NP, to_VP, Ø
- SUBJTYPE (Subject type): PRON, PROPN, DEF_NP, INDEF_NP, PL_NP, EXPL, REL

Note: NER entity annotations (by_NP[ENT=X]) were removed for consistent granularity
with other PP complement types. All by-phrases are now classified as by_NP.

Ported from old pipeline: ConstructionMiner/src/seqminer/schema/extract.py
"""

from typing import Tuple, Optional, Set, Dict

# ============================================================================
# Constants
# ============================================================================

# Modal auxiliaries
MODALS = {
    'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must', 'ought'
}

# Pronouns for subject classification
PRONOUNS = {
    'it', 'they', 'we', 'you', 'i', 'he', 'she',
    'this', 'that', 'these', 'those'
}

# Relative pronouns
RELATIVE_PRONOUNS = {
    'which', 'that', 'who', 'whom', 'whose', 'where', 'when'
}

# Expletive subjects
EXPLETIVES = {'there', 'it'}  # context-dependent


# ============================================================================
# TAM Classification
# ============================================================================

def aux_to_tam(aux_surface: Tuple[str, ...], aux_lemmas: Tuple[str, ...], has_modal: bool) -> str:
    """
    Map auxiliary chain to normalized TAM signature.

    Args:
        aux_surface: Surface forms of auxiliaries, e.g., ("was",) or ("has", "been")
        aux_lemmas: Lemmas of auxiliaries, e.g., ("be",) or ("have", "be")
        has_modal: Whether the chain contains a modal auxiliary

    Returns:
        TAM category: "pres-be", "past-be", "perf-be", "modal-be", "prog-be", "get-be", "unknown"

    Examples:
        >>> aux_to_tam(("is",), ("be",), False)
        'pres-be'
        >>> aux_to_tam(("was",), ("be",), False)
        'past-be'
        >>> aux_to_tam(("will", "be"), ("will", "be"), True)
        'modal-be'
        >>> aux_to_tam(("has", "been"), ("have", "be"), False)
        'perf-be'
    """
    # Handle empty input
    if not aux_surface and not aux_lemmas:
        return "unknown"

    # Normalize to lowercase
    s = tuple(w.lower() for w in aux_surface)
    L = {l.lower() for l in aux_lemmas}

    # 1. Modal + be
    if has_modal and "be" in L:
        return "modal-be"

    # 2. Simple past
    if s in {("was",), ("were",)}:
        return "past-be"

    # 3. Simple present
    if s in {("is",), ("are",), ("am",)}:
        return "pres-be"

    # 4. Perfect (has/have/had been)
    if ("have" in L or "has" in s or "have" in s) and ("been" in s or "be" in L):
        return "perf-be"

    # 5. Progressive (is/was being)
    if "being" in s and "be" in L:
        return "prog-be"

    # 6. Get-passive
    if "get" in L:
        return "get-be"

    # 7. Unknown - transparent failure handling
    return "unknown"


# ============================================================================
# Subject Classification
# ============================================================================

def classify_subject(subj_span) -> str:
    """
    Classify subject type using spaCy span.

    Args:
        subj_span: spaCy Span object representing the subject

    Returns:
        SUBJTYPE: "PRON", "PROPN", "DEF_NP", "INDEF_NP", "PL_NP", "EXPL", "REL"

    Examples:
        Subject: "it" -> "PRON"
        Subject: "John" -> "PROPN"
        Subject: "the book" -> "DEF_NP"
        Subject: "a student" -> "INDEF_NP"
        Subject: "students" -> "PL_NP"
        Subject: "there" (expletive) -> "EXPL"
    """
    if not subj_span:
        return "DEF_NP"  # default

    # Use spaCy span if available
    if hasattr(subj_span, 'root'):
        head = subj_span.root

        # 1. Expletive (there, it in "it is known")
        if head.dep_ == "expl":
            return "EXPL"

        # 2. Pronoun
        if head.pos_ == "PRON":
            # Check if relative pronoun
            if head.lemma_.lower() in RELATIVE_PRONOUNS:
                return "REL"
            return "PRON"

        # 3. Proper noun
        if head.pos_ == "PROPN":
            return "PROPN"

        # 4. Plural NP (morphologically plural)
        if hasattr(head, 'morph') and "Plur" in head.morph.get("Number", []):
            return "PL_NP"

        # 5. Check determiners for definite/indefinite
        dets = {t.lemma_.lower() for t in subj_span if t.dep_ == "det"}
        if dets & {"the", "this", "that", "these", "those"}:
            return "DEF_NP"
        if dets & {"a", "an", "some", "any"}:
            return "INDEF_NP"

        return "DEF_NP"  # default

    # Fallback for string-based processing (if no spaCy span)
    subj_lower = str(subj_span).lower().strip()

    if subj_lower in PRONOUNS:
        return "PRON"
    if subj_lower.startswith(('the ', 'this ', 'that ')):
        return "DEF_NP" if not subj_lower.endswith('s') else "PL_NP"
    if subj_lower.startswith(('a ', 'an ')):
        return "INDEF_NP"

    return "DEF_NP"


# ============================================================================
# Complement Classification
# ============================================================================

def canonicalize_complement(complements_dict: Dict[str, str], prep_head: Optional[str] = None) -> str:
    """
    Classify complement type based on extracted complements.

    Args:
        complements_dict: Dictionary of extracted complements
            - 'by': by-phrase text
            - 'by_entity': entity type (if available)
            - 'to_vp': to-infinitive text
            - 'pp_X': prepositional phrase with preposition X
        prep_head: Main preposition (e.g., "by", "with", "on")

    Returns:
        COMP type: "by_NP", "pp_X_NP", "to_VP", "Ø"

    Examples:
        complements={'by': 'by students'} -> "by_NP"
        complements={'by_entity': 'PERSON'} -> "by_NP"  # NER collapsed
        complements={'to_vp': 'to complete'} -> "to_VP"
        complements={'pp_with': 'with care'} -> "pp_with_NP"
        complements={} -> "Ø"
    """
    # 1. Check for by-phrase (NER entity removed for consistency with other PP types)
    # Previously: by_NP[ENT=X] - now collapsed to by_NP for uniform granularity
    if 'by_entity' in complements_dict:
        return "by_NP"

    # 2. Check for preposition
    if prep_head:
        prep_lower = prep_head.lower()
        if prep_lower == "by":
            return "by_NP"
        else:
            return f"pp_{prep_lower}_NP"

    # 3. Check for to-infinitive
    if 'to_inf' in complements_dict or 'to_vp' in complements_dict:
        return "to_VP"

    # 4. Check for any prepositional phrase in dict
    for key in complements_dict:
        if key.startswith('pp_'):
            prep = key.split('_')[1] if '_' in key else key
            return f"pp_{prep}_NP"

    # 5. No complement - zero complement
    return "Ø"


# ============================================================================
# Helper Functions
# ============================================================================

def has_modal_in_chain(aux_chain) -> bool:
    """
    Check if auxiliary chain contains a modal.

    Args:
        aux_chain: List of spaCy Token objects or strings

    Returns:
        True if modal present, False otherwise
    """
    for aux in aux_chain:
        lemma = aux.lemma_.lower() if hasattr(aux, 'lemma_') else str(aux).lower()
        if lemma in MODALS:
            return True
    return False


def format_pattern(subjtype: str, tam: str, comp: str) -> str:
    """
    Format complete TAM×COMP pattern.

    Args:
        subjtype: Subject type classification (kept for signature compatibility, not used)
        tam: TAM category
        comp: Complement type

    Returns:
        Formatted pattern string: TAM,COMP

    Note:
        SUBJTYPE excluded from pattern key to match old pipeline behavior.
        Grouping by TAM×COMP only (with NER tags in COMP when applicable).
        This prevents artificial schema splitting by subject type.

    Example:
        >>> format_pattern("DEF_NP", "pres-be", "by_NP")
        'pres-be,by_NP'
    """
    return f"{tam},{comp}"


# ============================================================================
# Validation
# ============================================================================

def validate_tam(tam: str) -> bool:
    """Check if TAM category is valid."""
    valid_tams = {"pres-be", "past-be", "perf-be", "modal-be", "prog-be", "get-be", "unknown"}
    return tam in valid_tams


def validate_subjtype(subjtype: str) -> bool:
    """Check if SUBJTYPE is valid."""
    valid_types = {"PRON", "PROPN", "DEF_NP", "INDEF_NP", "PL_NP", "EXPL", "REL"}
    return subjtype in valid_types


def validate_comp(comp: str) -> bool:
    """Check if COMP type is valid (pp_X_NP, by_NP, to_VP, or Ø)."""
    if comp == "Ø" or comp == "to_VP" or comp == "by_NP":
        return True
    # NER variants removed - all by_NP are now collapsed
    if comp.startswith("pp_") and comp.endswith("_NP"):
        return True
    return False


# ============================================================================
# Test Cases (for validation)
# ============================================================================

if __name__ == "__main__":
    print("Testing TAM×COMP Classification Functions\n")

    # Test aux_to_tam
    print("=== TAM Classification ===")
    test_cases = [
        (("is",), ("be",), False, "pres-be"),
        (("was",), ("be",), False, "past-be"),
        (("has", "been"), ("have", "be"), False, "perf-be"),
        (("will", "be"), ("will", "be"), True, "modal-be"),
        (("is", "being"), ("be", "be"), False, "prog-be"),
        (("got",), ("get",), False, "get-be"),
    ]

    for surf, lemmas, modal, expected in test_cases:
        result = aux_to_tam(surf, lemmas, modal)
        status = "✓" if result == expected else "✗"
        print(f"{status} {surf} -> {result} (expected: {expected})")

    # Test complement classification
    print("\n=== COMP Classification ===")
    comp_tests = [
        ({'by': 'by students'}, 'by', "by_NP"),
        ({'by_entity': 'PERSON'}, None, "by_NP"),  # NER collapsed
        ({'to_vp': 'to complete'}, None, "to_VP"),
        ({}, None, "Ø"),
    ]

    for comps, prep, expected in comp_tests:
        result = canonicalize_complement(comps, prep)
        status = "✓" if result == expected else "✗"
        print(f"{status} {comps} -> {result} (expected: {expected})")

    print("\n✓ TAM×COMP classifier ready!")
