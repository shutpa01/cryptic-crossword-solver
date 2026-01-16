#!/usr/bin/env python3
"""
Explanation Builder - Presentation layer for cryptic crossword solver.

This module handles:
1. Building formula notation for solved clues
2. Creating word-by-word breakdowns
3. Assessing explanation quality
4. Formatting output for display

Separated from compound_wordplay_analyzer.py to maintain separation of concerns:
- Analysis (compound_wordplay_analyzer.py) - finding indicators, solving wordplay
- Presentation (this file) - building explanations, formatting output
"""

import sys
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, r'C:\Users\shute\PycharmProjects\cryptic_solver')

from solver.solver_engine.resources import norm_letters
from solver.wordplay.anagram.compound_wordplay_analyzer import (
    WordRole, CompoundWordplayAnalyzer
)


class ExplanationBuilder:
    """Builds explanations from analyzed compound wordplay cases."""

    def __init__(self):
        # Link words to exclude from word counts
        self.link_words = {
            'a', 'an', 'the', 'to', 'for', 'of', 'in', 'on', 'at', 'by',
            'is', 'it', 'as', 'be', 'or', 'and', 'with', 'from', 'into',
            'that', 'this', 'are', 'was', 'were', 'been', 'being',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'its', 'his', 'her', 'your', 'their', 'our', 'my',
            'but', 'yet', 'so', 'if', 'then', 'than', 'when', 'where',
            'who', 'what', 'which', 'how', 'why',
            'one', 'ones', 'some', 'any', 'all', 'each', 'every',
            'no', 'not', 'nor', 'neither', 'either',
            'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'once', 'here', 'there',
            'about', 'after', 'before', 'between', 'through',
            'during', 'above', 'below', 'against', 'among',
            'such', 'only', 'just', 'also', 'very', 'too',
            'well', 'back', 'even', 'still', 'already',
            'always', 'never', 'ever', 'often', 'sometimes',
            'perhaps', 'maybe', 'rather', 'quite', 'almost',
            'get', 'gets', 'getting', 'got', 'make', 'makes', 'making', 'made',
            'give', 'gives', 'giving', 'gave', 'given',
            'take', 'takes', 'taking', 'took', 'taken',
            'come', 'comes', 'coming', 'came',
            'go', 'goes', 'going', 'went', 'gone',
            'see', 'sees', 'seeing', 'saw', 'seen',
            'say', 'says', 'saying', 'said',
            'know', 'knows', 'knowing', 'knew', 'known',
            'think', 'thinks', 'thinking', 'thought',
            'find', 'finds', 'finding', 'found',
            'want', 'wants', 'wanting', 'wanted',
            'use', 'uses', 'using', 'used',
            'try', 'tries', 'trying', 'tried',
            'need', 'needs', 'needing', 'needed',
            'seem', 'seems', 'seeming', 'seemed',
            'feel', 'feels', 'feeling', 'felt',
            'become', 'becomes', 'becoming', 'became',
            'leave', 'leaves', 'leaving', 'left',
            'put', 'puts', 'putting',
            'mean', 'means', 'meaning', 'meant',
            'keep', 'keeps', 'keeping', 'kept',
            'let', 'lets', 'letting',
            'begin', 'begins', 'beginning', 'began', 'begun',
            'show', 'shows', 'showing', 'showed', 'shown',
            'hear', 'hears', 'hearing', 'heard',
            'play', 'plays', 'playing', 'played',
            'run', 'runs', 'running', 'ran',
            'move', 'moves', 'moving', 'moved',
            'like', 'likes', 'liking', 'liked',
            'live', 'lives', 'living', 'lived',
            'believe', 'believes', 'believing', 'believed',
            'hold', 'holds', 'holding', 'held',
            'bring', 'brings', 'bringing', 'brought',
            'happen', 'happens', 'happening', 'happened',
            'write', 'writes', 'writing', 'wrote', 'written',
            'provide', 'provides', 'providing', 'provided',
            'sit', 'sits', 'sitting', 'sat',
            'stand', 'stands', 'standing', 'stood',
            'lose', 'loses', 'losing', 'lost',
            'pay', 'pays', 'paying', 'paid',
            'meet', 'meets', 'meeting', 'met',
            'include', 'includes', 'including', 'included',
            'continue', 'continues', 'continuing', 'continued',
            'set', 'sets', 'setting',
            'learn', 'learns', 'learning', 'learned', 'learnt',
            'change', 'changes', 'changing', 'changed',
            'lead', 'leads', 'leading', 'led',
            'understand', 'understands', 'understanding', 'understood',
            'watch', 'watches', 'watching', 'watched',
            'follow', 'follows', 'following', 'followed',
            'stop', 'stops', 'stopping', 'stopped',
            'create', 'creates', 'creating', 'created',
            'speak', 'speaks', 'speaking', 'spoke', 'spoken',
            'read', 'reads', 'reading',
            'allow', 'allows', 'allowing', 'allowed',
            'add', 'adds', 'adding', 'added',
            'spend', 'spends', 'spending', 'spent',
            'grow', 'grows', 'growing', 'grew', 'grown',
            'open', 'opens', 'opening', 'opened',
            'walk', 'walks', 'walking', 'walked',
            'win', 'wins', 'winning', 'won',
            'offer', 'offers', 'offering', 'offered',
            'remember', 'remembers', 'remembering', 'remembered',
            'love', 'loves', 'loving', 'loved',
            'consider', 'considers', 'considering', 'considered',
            'appear', 'appears', 'appearing', 'appeared',
            'buy', 'buys', 'buying', 'bought',
            'wait', 'waits', 'waiting', 'waited',
            'serve', 'serves', 'serving', 'served',
            'die', 'dies', 'dying', 'died',
            'send', 'sends', 'sending', 'sent',
            'expect', 'expects', 'expecting', 'expected',
            'build', 'builds', 'building', 'built',
            'stay', 'stays', 'staying', 'stayed',
            'fall', 'falls', 'falling', 'fell', 'fallen',
            'cut', 'cuts', 'cutting',
            'reach', 'reaches', 'reaching', 'reached',
            'kill', 'kills', 'killing', 'killed',
            'remain', 'remains', 'remaining', 'remained',
            'suggest', 'suggests', 'suggesting', 'suggested',
            'raise', 'raises', 'raising', 'raised',
            'pass', 'passes', 'passing', 'passed',
            'sell', 'sells', 'selling', 'sold',
            'require', 'requires', 'requiring', 'required',
            'report', 'reports', 'reporting', 'reported',
            'decide', 'decides', 'deciding', 'decided',
            'pull', 'pulls', 'pulling', 'pulled'
        }

    def build_explanation(self, case: Dict[str, Any],
                          word_roles: List[WordRole],
                          fodder_words: List[str],
                          fodder_letters: str,
                          anagram_indicator: Optional[str],
                          definition_window: Optional[str],
                          compound_solution: Optional[Dict[str, Any]],
                          clue_words: List[str],
                          likely_answer: str) -> Dict[str, Any]:
        """
        Build a complete explanation following the format spec.

        Uses likely_answer (from matched candidate), NOT database answer.
        """
        # Get substitutions from compound solution
        subs = []
        if compound_solution and compound_solution.get('substitutions'):
            subs = compound_solution['substitutions']

        # Get additional fodder from compound solution
        additional_fodder = []
        if compound_solution and compound_solution.get('additional_fodder'):
            additional_fodder = compound_solution['additional_fodder']

        # Build fodder part - include both original and additional fodder
        all_fodder_words = list(fodder_words) if fodder_words else []
        for word, letters in additional_fodder:
            all_fodder_words.append(word.upper().replace("'", ""))  # Strip apostrophes

        fodder_part = ' + '.join(
            w.upper() for w in all_fodder_words) if all_fodder_words else ''

        # Check for deletion operation (only if validated with indicator)
        if compound_solution and compound_solution.get('operation') == 'deletion':
            deletion_target = compound_solution.get('deletion_target')
            excess = compound_solution.get('excess_letters', '')

            if deletion_target:
                word, letters, category = deletion_target
                formula = f"anagram({fodder_part}) - {letters} ({word}) = {likely_answer}"
            else:
                formula = f"anagram({fodder_part}) - {excess} = {likely_answer}"
        elif compound_solution and compound_solution.get('operation') == 'reduced_fodder':
            # Reduced fodder with substitution - show the corrected fodder
            reduced = compound_solution.get('reduced_fodder', '')
            subs_from_compound = compound_solution.get('substitutions', [])
            if subs_from_compound:
                sub_part = ' + '.join(
                    f"{letters} ({word})" for word, letters, _ in subs_from_compound)
                # Get the actual fodder words (not the removed one)
                actual_fodder = [wr.word for wr in word_roles if wr.role == 'fodder']
                fodder_part = ' + '.join(w.upper() for w in actual_fodder)
                formula = f"anagram({fodder_part}) + {sub_part} = {likely_answer}"
            else:
                formula = f"anagram({reduced}) = {likely_answer}"
        elif compound_solution and compound_solution.get(
                'operation') == 'unresolved_excess':
            # Excess letters but no deletion indicator - show honest formula
            formula = f"anagram({fodder_part}) = {likely_answer} [excess letters unresolved]"
        elif subs:
            # Compound with substitutions (additions)
            sub_part = ' + '.join(f"{letters} ({word})" for word, letters, _ in subs)

            construction = compound_solution.get('construction',
                                                 {}) if compound_solution else {}
            op = construction.get('operation', 'concatenation')

            if op == 'insertion':
                formula = f"anagram({fodder_part}) with {sub_part} inserted = {likely_answer}"
            elif op == 'container':
                formula = f"{sub_part} inside anagram({fodder_part}) = {likely_answer}"
            else:
                formula = f"anagram({fodder_part}) + {sub_part} = {likely_answer}"
        else:
            # Pure anagram
            formula = f"anagram({fodder_part}) = {likely_answer}"

        # Build word-by-word explanation following clue order
        explanations = []
        # Use norm_letters for lookup to handle punctuation (e.g., "gate," matches "gate")
        role_lookup = {norm_letters(wr.word): wr for wr in word_roles}
        explained_words = set()

        for word in clue_words:
            word_norm = norm_letters(word)

            if word_norm in explained_words:
                continue

            if word_norm in role_lookup:
                wr = role_lookup[word_norm]
                explained_words.add(word_norm)

                if wr.role == 'definition':
                    explanations.append(f'• "{word}" = definition for {likely_answer}')
                elif wr.role == 'fodder':
                    explanations.append(f'• "{word}" = anagram fodder')
                elif wr.role == 'anagram_indicator':
                    explanations.append(f'• "{word}" = anagram indicator')
                elif wr.role == 'substitution':
                    explanations.append(f'• "{word}" = {wr.contributes} ({wr.source})')
                elif wr.role == 'deletion_target':
                    explanations.append(f'• "{word}" = {wr.contributes} (to be removed)')
                elif wr.role == 'deletion_indicator':
                    explanations.append(f'• "{word}" = deletion indicator')
                elif wr.role == 'positional_indicator':
                    explanations.append(
                        f'• "{word}" = positional indicator (construction order)')
                elif wr.role == 'insertion_indicator':
                    explanations.append(f'• "{word}" = insertion indicator')
                elif wr.role == 'container_indicator':
                    explanations.append(f'• "{word}" = container indicator')
                elif wr.role == 'parts_indicator':
                    # Extract subtype from source if available (e.g., "last_use" -> "last letter")
                    if 'last' in wr.source.lower():
                        explanations.append(f'• "{word}" = last letter indicator')
                    elif 'first' in wr.source.lower():
                        explanations.append(f'• "{word}" = first letter indicator')
                    else:
                        explanations.append(f'• "{word}" = letter selector')
                elif wr.role == 'link':
                    pass  # Skip link words in explanation
                else:
                    # Generic indicator
                    if '_indicator' in wr.role:
                        ind_type = wr.role.replace('_indicator', '')
                        explanations.append(f'• "{word}" = {ind_type} indicator')

        return {
            'formula': formula,
            'breakdown': explanations,
            'quality': self.assess_quality(compound_solution, word_roles, clue_words)
        }

    def assess_quality(self, compound_solution: Optional[Dict[str, Any]],
                       word_roles: List[WordRole],
                       clue_words: List[str]) -> str:
        """Assess explanation quality based on word coverage."""
        accounted = {norm_letters(wr.word) for wr in word_roles}
        total_words = len(
            [w for w in clue_words if norm_letters(w) not in self.link_words])
        accounted_content = len([w for w in clue_words
                                 if norm_letters(w) in accounted and norm_letters(
                w) not in self.link_words])

        # Calculate coverage ratio
        coverage = accounted_content / total_words if total_words > 0 else 0

        # Check for required elements
        has_definition = any(wr.role == 'definition' for wr in word_roles)
        has_fodder = any(wr.role == 'fodder' for wr in word_roles)
        has_indicator = any('indicator' in wr.role for wr in word_roles)

        # Compound with substitutions
        if compound_solution and compound_solution.get('fully_resolved'):
            if coverage >= 0.9:
                return 'solved'
            else:
                return 'high'

        # Pure anagram - check if fully explained
        if has_definition and has_fodder and has_indicator:
            if coverage >= 0.9:
                return 'solved'
            elif coverage >= 0.7:
                return 'high'
            else:
                return 'medium'

        # Partial explanation
        if compound_solution and compound_solution.get('substitutions'):
            return 'medium'
        elif has_fodder and has_indicator:
            return 'medium'
        elif has_fodder:
            return 'low'
        else:
            return 'none'

    def build_fallback(self, case: Dict[str, Any],
                       clue_words: List[str], db_answer: str) -> Dict[str, Any]:
        """Fallback when no evidence available."""
        return {
            'clue': case.get('clue', ''),
            'likely_answer': '',  # No likely answer - couldn't solve
            'db_answer': db_answer,  # For comparison only
            'answer_matches': False,
            'word_roles': [],
            'definition_window': None,
            'anagram_component': None,
            'compound_solution': None,
            'explanation': {
                'formula': 'Unable to analyze',
                'breakdown': [],
                'quality': 'none'
            },
            'remaining_unresolved': clue_words
        }


class ExplanationSystemBuilder:
    """
    Wrapper class maintaining backward compatibility with pipeline.
    Processes cases through the CompoundWordplayAnalyzer and ExplanationBuilder.
    """

    def __init__(self):
        self.analyzer = CompoundWordplayAnalyzer()
        self.explainer = ExplanationBuilder()

    def close(self):
        self.analyzer.close()

    def enhance_pipeline_data(self, evidence_enhanced_cases: List[Dict[str, Any]]) -> \
    List[Dict[str, Any]]:
        """
        Process evidence-enhanced cases through compound analysis.
        """
        enhanced = []

        print("Analyzing compound wordplay with database lookups...")

        for i, case in enumerate(evidence_enhanced_cases):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} cases...")

            try:
                result = self.analyzer.analyze_case(case)
                enhanced.append(result)
            except Exception as e:
                print(f"Warning: Could not analyze case {i + 1}: {e}")
                enhanced.append({
                    'clue': case.get('clue', ''),
                    'answer': case.get('answer', ''),
                    'error': str(e)
                })

        print(f"Analyzed {len(enhanced)} cases.")
        return enhanced

    def build_explanations(self, enhanced_cases: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Extract explanations from analyzed cases.
        """
        return [
            {
                'clue': case.get('clue', ''),
                'likely_answer': case.get('likely_answer', ''),
                'db_answer': case.get('db_answer', ''),
                'answer_matches': case.get('answer_matches', False),
                'explanation': case.get('explanation', {}),
                'quality': case.get('explanation', {}).get('quality', 'none')
            }
            for case in enhanced_cases
        ]