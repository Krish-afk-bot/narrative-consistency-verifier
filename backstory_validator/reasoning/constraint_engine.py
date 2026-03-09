# Constraint and causal reasoning engine
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backstory_validator.models.schemas import Claim, Evidence, ClaimEvidencePair
from backstory_validator.llm.gemini_client import GeminiClient
from backstory_validator.config import (
    RelationshipType,
    ClaimImportance,
    CONTRADICTION_THRESHOLD,
    SUPPORT_THRESHOLD
)


@dataclass
class RelationshipAnalysis:
    relationship: RelationshipType
    confidence: float
    supporting_signals: List[str]
    conflicting_signals: List[str]
    causal_consistent: bool
    causal_explanation: str


class ConstraintReasoningEngine:
    # Analyzes claim-evidence pairs using rule-based reasoning
    # Gemini is used ONLY for explanation, NEVER for classification
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini = gemini_client
        
        # Contradiction detection patterns
        self._contradiction_patterns = [
            (r'\bnever\b.*\b(did|was|had|knew)\b', 'explicit_negation'),
            (r'\b(impossible|cannot|could not|never)\b', 'impossibility'),
            (r'\b(died|killed|dead)\b.*\b(before|prior)\b', 'temporal_death'),
            (r'\b(born|birth)\b.*\b(after|later)\b', 'temporal_birth'),
            (r'\b(first|only)\b.*\b(time|instance|occasion)\b', 'uniqueness'),
        ]
        
        # Support detection patterns
        self._support_patterns = [
            (r'\b(always|often|frequently|regularly)\b', 'habitual'),
            (r'\b(known for|famous for|remembered for)\b', 'reputation'),
            (r'\b(learned|taught|trained|studied)\b', 'skill_acquisition'),
            (r'\b(believed|thought|felt|considered)\b', 'belief_alignment'),
        ]
        
        # Constraint detection patterns
        self._constraint_patterns = [
            (r'\b(only|just|merely|simply)\b', 'limitation'),
            (r'\b(except|unless|but not)\b', 'exception'),
            (r'\b(sometimes|occasionally|rarely)\b', 'frequency_limit'),
        ]
    
    def analyze_pair(self, claim: Claim, evidence: Evidence,
                     character: str) -> ClaimEvidencePair:
        # Analyze relationship between claim and evidence
        analysis = self._analyze_relationship(claim, evidence)
        
        # Get Gemini explanation (for explanation only, not classification)
        gemini_explanation = None
        if self.gemini:
            try:
                explanation_data = self.gemini.explain_relationship(
                    claim.text, evidence.chunk.text, character
                )
                gemini_explanation = self._format_gemini_explanation(explanation_data)
            except Exception as e:
                gemini_explanation = f"Explanation unavailable: {e}"
        
        return ClaimEvidencePair(
            claim=claim,
            evidence=evidence,
            relationship=analysis.relationship,
            relationship_confidence=analysis.confidence,
            causal_consistency=analysis.causal_consistent,
            causal_explanation=analysis.causal_explanation,
            gemini_explanation=gemini_explanation
        )
    
    def _analyze_relationship(self, claim: Claim, 
                               evidence: Evidence) -> RelationshipAnalysis:
        # Rule-based relationship classification (deterministic)
        claim_text = claim.text.lower()
        evidence_text = evidence.chunk.text.lower()
        
        supporting_signals = []
        conflicting_signals = []
        
        # Check contradiction patterns
        contradiction_score = 0
        for pattern, signal_name in self._contradiction_patterns:
            if re.search(pattern, evidence_text):
                if self._pattern_relates_to_claim(claim_text, evidence_text, pattern):
                    conflicting_signals.append(signal_name)
                    contradiction_score += 0.3
        
        # Check direct negation
        negation_score = self._check_direct_negation(claim_text, evidence_text)
        if negation_score > 0:
            conflicting_signals.append('direct_negation')
            contradiction_score += negation_score
        
        # Check temporal consistency
        temporal_score = self._check_temporal_consistency(claim, evidence)
        if temporal_score < 0:
            conflicting_signals.append('temporal_inconsistency')
            contradiction_score += abs(temporal_score)
        
        # Check support patterns
        support_score = 0
        for pattern, signal_name in self._support_patterns:
            if re.search(pattern, evidence_text):
                if self._pattern_relates_to_claim(claim_text, evidence_text, pattern):
                    supporting_signals.append(signal_name)
                    support_score += 0.25
        
        # Check keyword overlap
        keyword_overlap = self._compute_keyword_overlap(claim, evidence)
        if keyword_overlap > 0.3:
            supporting_signals.append('keyword_overlap')
            support_score += keyword_overlap * 0.3
        
        # Check constraint patterns
        constraint_score = 0
        for pattern, signal_name in self._constraint_patterns:
            if re.search(pattern, evidence_text):
                constraint_score += 0.2
        
        # Determine final relationship
        relationship, confidence = self._determine_relationship(
            contradiction_score, support_score, constraint_score,
            supporting_signals, conflicting_signals
        )
        
        # Check causal consistency
        causal_consistent, causal_explanation = self._check_causal_consistency(
            claim, evidence, relationship
        )
        
        return RelationshipAnalysis(
            relationship=relationship,
            confidence=confidence,
            supporting_signals=supporting_signals,
            conflicting_signals=conflicting_signals,
            causal_consistent=causal_consistent,
            causal_explanation=causal_explanation
        )
    
    def _pattern_relates_to_claim(self, claim_text: str, evidence_text: str,
                                   pattern: str) -> bool:
        # Check if pattern in evidence relates to claim content
        claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim_text))
        
        match = re.search(pattern, evidence_text)
        if not match:
            return False
        
        # Get context around match
        start = max(0, match.start() - 100)
        end = min(len(evidence_text), match.end() + 100)
        context = evidence_text[start:end]
        
        context_words = set(re.findall(r'\b[a-z]{4,}\b', context))
        overlap = claim_words & context_words
        
        return len(overlap) >= 2
    
    def _check_direct_negation(self, claim_text: str, 
                                evidence_text: str) -> float:
        # Check for direct negation of claim verbs
        claim_verbs = re.findall(r'\b(was|were|had|did|knew|learned|believed)\b', claim_text)
        
        negation_score = 0
        for verb in claim_verbs:
            negated_patterns = [
                f'never {verb}',
                f'did not {verb}',
                f"didn't {verb}",
                f'not {verb}',
                f'no {verb}'
            ]
            for neg_pattern in negated_patterns:
                if neg_pattern in evidence_text:
                    negation_score += 0.4
        
        return min(1.0, negation_score)
    
    def _check_temporal_consistency(self, claim: Claim, 
                                     evidence: Evidence) -> float:
        # Check if claim timeline matches evidence timeline
        claim_temporal = claim.temporal_indicators
        evidence_timeline = evidence.timeline_coverage.value
        
        early_indicators = ['childhood', 'young', 'boy', 'girl', 'birth', 'early']
        late_indicators = ['death', 'final', 'last', 'end', 'eventually']
        
        claim_is_early = any(ind in ' '.join(claim_temporal).lower() 
                            for ind in early_indicators)
        claim_is_late = any(ind in ' '.join(claim_temporal).lower() 
                           for ind in late_indicators)
        
        # Temporal mismatch detection
        if claim_is_early and evidence_timeline == 'late':
            return -0.2
        if claim_is_late and evidence_timeline == 'early':
            return -0.2
        
        # Matching timeline is positive
        if claim_is_early and evidence_timeline == 'early':
            return 0.3
        if claim_is_late and evidence_timeline == 'late':
            return 0.3
        
        return 0.0
    
    def _compute_keyword_overlap(self, claim: Claim, evidence: Evidence) -> float:
        if not claim.keywords:
            return 0.0
        
        evidence_text_lower = evidence.chunk.text.lower()
        matches = sum(1 for kw in claim.keywords if kw.lower() in evidence_text_lower)
        
        return matches / len(claim.keywords)
    
    def _determine_relationship(self, contradiction_score: float,
                                 support_score: float,
                                 constraint_score: float,
                                 supporting_signals: List[str],
                                 conflicting_signals: List[str]) -> Tuple[RelationshipType, float]:
        # Determine relationship type based on scores
        total = contradiction_score + support_score + constraint_score + 0.1
        
        if contradiction_score > CONTRADICTION_THRESHOLD:
            return RelationshipType.CONTRADICTS, min(1.0, contradiction_score)
        
        if support_score > SUPPORT_THRESHOLD and contradiction_score < 0.2:
            return RelationshipType.SUPPORTS, min(1.0, support_score)
        
        if constraint_score > 0.3 and contradiction_score < 0.3:
            return RelationshipType.CONSTRAINS, min(1.0, constraint_score)
        
        # Default to contextual
        confidence = 1.0 - (contradiction_score + support_score + constraint_score)
        return RelationshipType.CONTEXTUAL, max(0.3, confidence)
    
    def _check_causal_consistency(self, claim: Claim, evidence: Evidence,
                                   relationship: RelationshipType) -> Tuple[bool, str]:
        # Check if backstory claim is causally consistent with evidence
        if relationship == RelationshipType.CONTRADICTS:
            return False, "Evidence directly contradicts the claim"
        
        if relationship == RelationshipType.SUPPORTS:
            return True, "Evidence is causally consistent with the claim"
        
        if relationship == RelationshipType.CONSTRAINS:
            if claim.importance == ClaimImportance.CORE:
                return False, "Evidence constrains a core identity claim"
            return True, "Evidence constrains but doesn't invalidate the claim"
        
        return True, "Evidence provides context without causal conflict"
    
    def _format_gemini_explanation(self, explanation_data: Dict[str, Any]) -> str:
        parts = []
        
        if 'relevant_info' in explanation_data:
            parts.append(f"Relevant Information: {explanation_data['relevant_info']}")
        
        if 'evidence_analysis' in explanation_data:
            parts.append(f"Evidence Analysis: {explanation_data['evidence_analysis']}")
        
        if 'causal_implications' in explanation_data:
            parts.append(f"Causal Implications: {explanation_data['causal_implications']}")
        
        if 'key_quotes' in explanation_data and explanation_data['key_quotes']:
            quotes = explanation_data['key_quotes']
            if isinstance(quotes, list):
                parts.append(f"Key Quotes: {'; '.join(quotes[:3])}")
        
        return '\n'.join(parts) if parts else "No detailed explanation available"
    
    def analyze_all_pairs(self, claims: List[Claim], 
                          evidence_map: Dict[str, List[Evidence]],
                          character: str) -> List[ClaimEvidencePair]:
        # Analyze all claim-evidence pairs
        all_pairs = []
        
        for claim in claims:
            evidence_list = evidence_map.get(claim.claim_id, [])
            for evidence in evidence_list:
                pair = self.analyze_pair(claim, evidence, character)
                all_pairs.append(pair)
        
        return all_pairs
