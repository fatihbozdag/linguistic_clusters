#!/usr/bin/env python3
"""
Passive Construction Extractor

Processes ASC-analyzer output and extracts complete passive constructions
with TAM×COMP classification using spaCy dependency parsing.

Architecture:
1. Parse ASC-analyzer *_ASCinfo.txt files
2. Identify PASSIVE-tagged tokens
3. Map to spaCy tokens for dependency analysis
4. Extract auxiliary chains, subjects, and complements
5. Classify using TAM×COMP schema
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import spacy
from spacy.tokens import Token, Span, Doc

from tam_comp_classifier import (
    aux_to_tam,
    classify_subject,
    canonicalize_complement,
    has_modal_in_chain,
    format_pattern,
    MODALS
)


# ============================================================================
# ASC File Parsing
# ============================================================================

def parse_asc_file(asc_file_path: str) -> List[List[Dict]]:
    """
    Parse *_ASCinfo.txt file from ASC-analyzer.

    Format:
        # sent_id = 1
        1   It      it
        2   is      be      ATTR
        3   time    time
        ...
        8   dominated   dominate    PASSIVE

    Args:
        asc_file_path: Path to *_ASCinfo.txt file

    Returns:
        List of sentences, where each sentence is a list of token dicts:
        {
            'token_num': int,
            'token': str,
            'lemma': str,
            'asc_tag': str or None
        }
    """
    sentences = []
    current_sent = []

    with open(asc_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Sentence boundary marker
            if line.startswith('# sent_id'):
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
                continue

            # Skip empty lines and other comments
            if not line or line.startswith('#'):
                continue

            # Parse token line: token_num \t token \t lemma \t [asc_tag]
            parts = line.split('\t')
            if len(parts) >= 3:
                token_info = {
                    'token_num': int(parts[0]),
                    'token': parts[1],
                    'lemma': parts[2],
                    'asc_tag': parts[3] if len(parts) > 3 and parts[3] else None
                }
                current_sent.append(token_info)

    # Don't forget the last sentence
    if current_sent:
        sentences.append(current_sent)

    return sentences


# ============================================================================
# Passive Extractor Class
# ============================================================================

class PassiveExtractor:
    """
    Extract complete passive constructions with TAM×COMP classification.

    Uses spaCy for dependency parsing and component extraction.
    Applies deterministic TAM×COMP classification rules.
    """

    def __init__(self, model_name: str = "en_core_web_trf", use_mps: bool = True):
        """
        Initialize extractor with spaCy model.

        Args:
            model_name: spaCy model to load (default: en_core_web_trf)
            use_mps: Use MPS acceleration if available (Apple Silicon)
        """
        # Try to enable MPS acceleration
        if use_mps:
            try:
                import torch
                if torch.backends.mps.is_available():
                    spacy.require_gpu()
                    print("✓ MPS acceleration enabled")
            except Exception as e:
                print(f"⚠ MPS not available: {e}")

        print(f"Loading spaCy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        print("✓ spaCy model loaded")

    def extract_from_file(self, asc_file_path: str) -> List[Dict]:
        """
        Extract all passive constructions from an ASC-analyzer output file.

        Args:
            asc_file_path: Path to *_ASCinfo.txt file

        Returns:
            List of PassiveConstruction dicts
        """
        sentences = parse_asc_file(asc_file_path)
        all_constructions = []

        for sent_idx, tokens_with_tags in enumerate(sentences, 1):
            constructions = self.extract_from_sentence(tokens_with_tags, sent_idx)
            all_constructions.extend(constructions)

        return all_constructions

    def extract_from_sentence(self, tokens_with_tags: List[Dict], sent_idx: int = 0) -> List[Dict]:
        """
        Extract passive constructions from a single sentence.

        Args:
            tokens_with_tags: List of token dicts from ASC-analyzer
            sent_idx: Sentence index for tracking

        Returns:
            List of PassiveConstruction dicts
        """
        # 1. Rebuild sentence text
        sentence_text = " ".join([t['token'] for t in tokens_with_tags])

        # 2. Parse with spaCy
        doc = self.nlp(sentence_text)

        # 3. Find PASSIVE-tagged tokens from ASC-analyzer
        passive_tokens = [t for t in tokens_with_tags if t.get('asc_tag') == 'PASSIVE']

        if not passive_tokens:
            return []

        # 4. Extract each passive construction
        constructions = []
        for passive_info in passive_tokens:
            # Map ASC token to spaCy token
            passive_head = self._find_spacy_token(doc, passive_info)

            if passive_head:
                # Extract complete construction
                construction = self._extract_construction(doc, passive_head, sent_idx)
                if construction:
                    constructions.append(construction)

        return constructions

    def _find_spacy_token(self, doc: Doc, passive_info: Dict) -> Optional[Token]:
        """
        Find spaCy token matching ASC PASSIVE tag.

        Matches by lemma and VBN tag (past participle).

        Args:
            doc: spaCy Doc object
            passive_info: Token info dict from ASC-analyzer

        Returns:
            Matching spaCy Token or None
        """
        target_lemma = passive_info['lemma'].lower()

        # Primary match: lemma + VBN tag
        for token in doc:
            if (token.lemma_.lower() == target_lemma and
                token.tag_ in ('VBN', 'VBD')):  # VBD for irregular forms
                return token

        # Fallback: match by lemma alone (if VBN match fails)
        for token in doc:
            if token.lemma_.lower() == target_lemma and token.pos_ == 'VERB':
                return token

        return None

    def _extract_construction(self, doc: Doc, passive_head: Token, sent_idx: int) -> Optional[Dict]:
        """
        Extract complete construction from passive head verb.

        Args:
            doc: spaCy Doc object
            passive_head: Token representing passive verb (VBN)
            sent_idx: Sentence index

        Returns:
            PassiveConstruction dict with all components
        """
        # Extract components
        aux_chain = self._extract_aux_chain(passive_head)
        subject = self._extract_subject(passive_head, aux_chain)
        complements = self._extract_complements(passive_head)

        # Classify TAM×COMP
        has_modal = has_modal_in_chain(aux_chain)
        aux_surface = tuple(t.text for t in aux_chain)
        aux_lemmas = tuple(t.lemma_ for t in aux_chain)

        tam = aux_to_tam(aux_surface, aux_lemmas, has_modal)
        subjtype = classify_subject(subject['span']) if subject else "DEF_NP"
        comp = canonicalize_complement(complements, complements.get('prep_head'))

        # Build surface representation
        surface = self._build_surface(subject, aux_chain, passive_head, complements)

        # Build full auxiliary chain for instance metrics (includes passive head)
        full_aux_chain = [t.text for t in aux_chain] + [passive_head.text]

        # Extract following word for boundary entropy calculation
        following_word = None
        for token in doc[passive_head.i + 1:]:
            # Get first content word after construction
            if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}:
                following_word = token.text
                break

        return {
            'sent_idx': sent_idx,
            'surface': surface,
            'tam': tam,
            'comp': comp,
            'subjtype': subjtype,
            'pattern': format_pattern(subjtype, tam, comp),
            'head_lemma': passive_head.lemma_,
            'subject_text': subject['text'] if subject else None,
            'aux_chain': full_aux_chain,  # NEW: includes passive head for ATP/ΔP
            'following_word': following_word,  # NEW: for boundary entropy
            'complements': complements
        }

    # ========================================================================
    # Component Extraction Methods
    # ========================================================================

    def _extract_aux_chain(self, passive_head: Token) -> List[Token]:
        """
        Extract auxiliary verbs attached to passive head.

        Looks for aux and auxpass dependencies.

        Args:
            passive_head: Main passive verb (VBN)

        Returns:
            List of auxiliary tokens, sorted by position
        """
        aux = []
        for child in passive_head.children:
            if child.dep_ in ('aux', 'auxpass'):
                aux.append(child)

        # Sort by token position
        aux.sort(key=lambda t: t.i)
        return aux

    def _extract_subject(self, passive_head: Token, aux_chain: List[Token]) -> Optional[Dict]:
        """
        Extract subject using dependency relations.

        Checks both passive head and auxiliary verbs for subject dependencies.

        Args:
            passive_head: Main passive verb
            aux_chain: List of auxiliary tokens

        Returns:
            Dict with subject info or None
        """
        # Check passive head's children first
        for child in passive_head.children:
            if child.dep_ in ('nsubjpass', 'csubj', 'csubjpass', 'nsubj'):
                return self._build_subject_span(child)

        # Check auxiliary's children
        for aux in aux_chain:
            for child in aux.children:
                if child.dep_ in ('nsubjpass', 'nsubj'):
                    return self._build_subject_span(child)

        return None

    def _build_subject_span(self, subject_token: Token) -> Dict:
        """
        Build complete subject span using subtree.

        Includes all dependents of the subject head.

        Args:
            subject_token: Head token of subject

        Returns:
            Dict with subject token, text, and span
        """
        subtree = list(subject_token.subtree)
        subtree.sort(key=lambda t: t.i)

        return {
            'token': subject_token,
            'text': " ".join(t.text for t in subtree),
            'span': subject_token.doc[subtree[0].i:subtree[-1].i+1]
        }

    def _extract_complements(self, passive_head: Token) -> Dict:
        """
        Extract complements (by-phrase, to-VP, prepositional phrases).

        Args:
            passive_head: Main passive verb

        Returns:
            Dict of complements with keys like 'by', 'to_vp', 'pp_with', etc.
        """
        complements = {}

        for child in passive_head.children:
            # By-phrase (agent)
            # Note: Entity type extraction removed - NER no longer used for by_NP patterns
            # All by-phrases are now classified as by_NP for consistent granularity
            if child.lower_ == "by" and child.pos_ == "ADP":
                complements['by'] = self._extract_pp_phrase(child)
                complements['prep_head'] = "by"

            # Other prepositional phrases
            elif child.dep_ == "prep" and child.pos_ == "ADP":
                prep = child.lower_
                complements[f'pp_{prep}'] = self._extract_pp_phrase(child)
                # Set prep_head only if not already set by "by"
                if 'prep_head' not in complements:
                    complements['prep_head'] = prep

            # To-infinitive (xcomp)
            elif child.dep_ == "xcomp":
                complements['to_vp'] = " ".join(t.text for t in child.subtree)

        return complements

    def _extract_pp_phrase(self, prep_token: Token) -> str:
        """
        Extract complete prepositional phrase.

        Args:
            prep_token: Preposition token

        Returns:
            Full PP text including preposition
        """
        subtree = list(prep_token.subtree)
        return " ".join(t.text for t in subtree)

    # Note: _get_entity_type method removed - NER entity extraction no longer used
    # All by-phrases are now uniformly classified as by_NP for methodological consistency

    def _build_surface(self, subject: Optional[Dict], aux_chain: List[Token],
                      passive_head: Token, complements: Dict) -> str:
        """
        Build surface representation of the construction.

        Format: [SUBJECT] [AUX-CHAIN] [PASSIVE-HEAD] [COMPLEMENTS]

        Args:
            subject: Subject dict
            aux_chain: List of auxiliary tokens
            passive_head: Main verb token
            complements: Complements dict

        Returns:
            Surface string representation
        """
        parts = []

        # Subject
        if subject:
            parts.append(subject['text'])

        # Auxiliary chain
        for aux in aux_chain:
            parts.append(aux.text)

        # Passive head
        parts.append(passive_head.text)

        # Complements
        if complements.get('by'):
            parts.append(complements['by'])
        elif complements.get('to_vp'):
            parts.append(complements['to_vp'])
        else:
            # Add first PP found
            for key, value in complements.items():
                if key.startswith('pp_'):
                    parts.append(value)
                    break

        return " ".join(parts)


# ============================================================================
# Test Function
# ============================================================================

def test_extractor():
    """Quick test of extractor."""
    print("\n=== Testing PassiveExtractor ===\n")

    extractor = PassiveExtractor()

    # Test parsing
    test_file = "test_data/BGSU1001_ASCinfo.txt"
    print(f"Parsing: {test_file}")

    sentences = parse_asc_file(test_file)
    print(f"✓ Parsed {len(sentences)} sentences")

    # Find passive tags
    passive_count = sum(1 for sent in sentences
                       for token in sent
                       if token.get('asc_tag') == 'PASSIVE')
    print(f"✓ Found {passive_count} PASSIVE tags")

    # Extract first construction
    for sent in sentences[:3]:
        constructions = extractor.extract_from_sentence(sent)
        for const in constructions:
            print(f"\n✓ Extracted construction:")
            print(f"  Surface: {const['surface']}")
            print(f"  TAM: {const['tam']}")
            print(f"  COMP: {const['comp']}")
            print(f"  SUBJTYPE: {const['subjtype']}")
            break

    print("\n✓ PassiveExtractor test complete!")


if __name__ == "__main__":
    test_extractor()
