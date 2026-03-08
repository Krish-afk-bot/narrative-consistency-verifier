# Deterministic classifier for backstory consistency
# CRITICAL: Final decision uses ONLY evidence-based logic, NOT Gemini
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from backstory_validator.models.schemas import (
    Claim, ClaimEvidencePair, EvaluationResult
)
from backstory_validator.config import (
    RelationshipType,
    ClaimImportance,
    CORE_CLAIM_FATAL_THRESHOLD
)


@dataclass
class ClassificationDecision:
    label: int  # 0=Contradict, 1=Consistent
    confidence: float
    fatal_contradictions: List[ClaimEvidencePair]
    reasoning: str


class DeterministicClassifier:
    # Decision Logic:
    # - If ANY core claim has fatal contradiction -> label=0 (Contradict)
    # - If ALL core claims remain compatible -> label=1 (Consistent)
    
    def __init__(self):
        self._decision_log: List[str] = []
    
    def classify(self, backstory_id: str, novel_id: str, character: str,
                 backstory_text: str, claims: List[Claim],
                 claim_evidence_pairs: List[ClaimEvidencePair]) -> EvaluationResult:
        self._decision_log = []
        
        # Find fatal contradictions
        fatal_contradictions = self._find_fatal_contradictions(
            claims, claim_evidence_pairs
        )
        
        # Make deterministic decision
        decision = self._make_decision(
            claims, claim_evidence_pairs, fatal_contradictions
        )
        
        return EvaluationResult(
            backstory_id=backstory_id,
            novel_id=novel_id,
            character=character,
            backstory_text=backstory_text,
            predicted_label=decision.label,
            confidence=decision.confidence,
            claims=claims,
            claim_evidence_pairs=claim_evidence_pairs,
            fatal_contradictions=decision.fatal_contradictions,
            reasoning_summary=decision.reasoning
        )
    
    def _find_fatal_contradictions(self, claims: List[Claim],
                                    pairs: List[ClaimEvidencePair]) -> List[ClaimEvidencePair]:
        # Find contradictions that are fatal (core claims with high confidence)
        fatal = []
        claim_importance = {c.claim_id: c.importance for c in claims}
        
        for pair in pairs:
            if pair.relationship != RelationshipType.CONTRADICTS:
                continue
            
            if pair.relationship_confidence < CORE_CLAIM_FATAL_THRESHOLD:
                continue
            
            importance = claim_importance.get(pair.claim.claim_id, ClaimImportance.SIGNIFICANT)
            
            if importance == ClaimImportance.CORE:
                fatal.append(pair)
                self._decision_log.append(
                    f"FATAL: Core claim '{pair.claim.text[:50]}...' contradicted "
                    f"with confidence {pair.relationship_confidence:.2f}"
                )
            elif importance == ClaimImportance.SIGNIFICANT:
                # Significant claims need higher threshold
                if pair.relationship_confidence > 0.9:
                    fatal.append(pair)
                    self._decision_log.append(
                        f"FATAL: Significant claim '{pair.claim.text[:50]}...' "
                        f"strongly contradicted with confidence {pair.relationship_confidence:.2f}"
                    )
        
        return fatal
    
    def _make_decision(self, claims: List[Claim],
                       pairs: List[ClaimEvidencePair],
                       fatal_contradictions: List[ClaimEvidencePair]) -> ClassificationDecision:
        # Rule 1: Any fatal contradiction -> Contradict
        if fatal_contradictions:
            confidence = self._compute_contradiction_confidence(fatal_contradictions)
            reasoning = self._build_contradiction_reasoning(fatal_contradictions)
            
            return ClassificationDecision(
                label=0,
                confidence=confidence,
                fatal_contradictions=fatal_contradictions,
                reasoning=reasoning
            )
        
        # Rule 2: Check overall evidence balance
        support_score, contradict_score = self._compute_evidence_balance(pairs)
        
        if contradict_score > 0.6 and contradict_score > support_score * 1.5:
            strong_contradictions = [
                p for p in pairs 
                if p.relationship == RelationshipType.CONTRADICTS
                and p.relationship_confidence > 0.5
            ]
            
            confidence = contradict_score
            reasoning = self._build_weak_contradiction_reasoning(
                strong_contradictions, support_score, contradict_score
            )
            
            return ClassificationDecision(
                label=0,
                confidence=confidence,
                fatal_contradictions=strong_contradictions,
                reasoning=reasoning
            )
        
        # Rule 3: All core claims compatible -> Consistent
        confidence = self._compute_consistency_confidence(pairs, support_score)
        reasoning = self._build_consistency_reasoning(pairs, support_score)
        
        return ClassificationDecision(
            label=1,
            confidence=confidence,
            fatal_contradictions=[],
            reasoning=reasoning
        )
    
    def _compute_contradiction_confidence(self, 
                                           fatal: List[ClaimEvidencePair]) -> float:
        if not fatal:
            return 0.0
        
        avg_confidence = sum(p.relationship_confidence for p in fatal) / len(fatal)
        count_boost = min(0.2, len(fatal) * 0.05)
        
        return min(1.0, avg_confidence + count_boost)
    
    def _compute_evidence_balance(self, 
                                   pairs: List[ClaimEvidencePair]) -> Tuple[float, float]:
        # Compute ratio of supporting vs contradicting evidence
        if not pairs:
            return 0.5, 0.0
        
        support_sum = 0.0
        contradict_sum = 0.0
        
        for pair in pairs:
            if pair.relationship == RelationshipType.SUPPORTS:
                support_sum += pair.relationship_confidence
            elif pair.relationship == RelationshipType.CONTRADICTS:
                contradict_sum += pair.relationship_confidence
            elif pair.relationship == RelationshipType.CONSTRAINS:
                contradict_sum += pair.relationship_confidence * 0.3
        
        total = support_sum + contradict_sum + 0.1
        
        return support_sum / total, contradict_sum / total
    
    def _compute_consistency_confidence(self, pairs: List[ClaimEvidencePair],
                                         support_score: float) -> float:
        if not pairs:
            return 0.5
        
        base = support_score
        
        # Reduce confidence if any contradictions exist
        contradictions = [p for p in pairs if p.relationship == RelationshipType.CONTRADICTS]
        if contradictions:
            max_contradict = max(p.relationship_confidence for p in contradictions)
            base -= max_contradict * 0.3
        
        # Boost for causal consistency
        causal_consistent = sum(1 for p in pairs if p.causal_consistency)
        causal_ratio = causal_consistent / len(pairs) if pairs else 0
        base += causal_ratio * 0.2
        
        return max(0.3, min(1.0, base))
    
    def _build_contradiction_reasoning(self, 
                                        fatal: List[ClaimEvidencePair]) -> str:
        parts = ["DECISION: Contradict (label=0)", ""]
        parts.append("FATAL CONTRADICTIONS FOUND:")
        
        for i, pair in enumerate(fatal[:5], 1):
            parts.append(f"\n{i}. Claim: {pair.claim.text[:100]}...")
            parts.append(f"   Evidence: {pair.evidence.chunk.text[:150]}...")
            parts.append(f"   Confidence: {pair.relationship_confidence:.2f}")
            parts.append(f"   Causal: {pair.causal_explanation}")
        
        parts.append("\n" + "-" * 50)
        parts.append("Decision made by deterministic logic, NOT by LLM.")
        
        return "\n".join(parts)
    
    def _build_weak_contradiction_reasoning(self, contradictions: List[ClaimEvidencePair],
                                             support_score: float,
                                             contradict_score: float) -> str:
        parts = ["DECISION: Contradict (label=0)", ""]
        parts.append(f"Evidence balance: Support={support_score:.2f}, Contradict={contradict_score:.2f}")
        parts.append("\nSTRONG CONTRADICTIONS:")
        
        for i, pair in enumerate(contradictions[:3], 1):
            parts.append(f"\n{i}. Claim: {pair.claim.text[:100]}...")
            parts.append(f"   Confidence: {pair.relationship_confidence:.2f}")
        
        parts.append("\n" + "-" * 50)
        parts.append("Decision made by deterministic logic, NOT by LLM.")
        
        return "\n".join(parts)
    
    def _build_consistency_reasoning(self, pairs: List[ClaimEvidencePair],
                                      support_score: float) -> str:
        parts = ["DECISION: Consistent (label=1)", ""]
        parts.append(f"Support score: {support_score:.2f}")
        parts.append("No fatal contradictions found in core claims.")
        
        supporting = [p for p in pairs if p.relationship == RelationshipType.SUPPORTS]
        if supporting:
            parts.append("\nSUPPORTING EVIDENCE:")
            for i, pair in enumerate(supporting[:3], 1):
                parts.append(f"\n{i}. Claim: {pair.claim.text[:80]}...")
                parts.append(f"   Support confidence: {pair.relationship_confidence:.2f}")
        
        parts.append("\n" + "-" * 50)
        parts.append("Decision made by deterministic logic, NOT by LLM.")
        
        return "\n".join(parts)
    
    @property
    def decision_log(self) -> List[str]:
        return self._decision_log
