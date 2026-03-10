# Data models for backstory validation pipeline
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from backstory_validator.config import (
    TimelinePosition, 
    RelationshipType, 
    ClaimImportance,
    ClaimCategory
)


@dataclass
class Chunk:
    # A text chunk from a novel with metadata
    chunk_id: str
    text: str
    novel_id: str
    chapter_index: int
    chapter_title: str
    paragraph_start: int
    paragraph_end: int
    timeline_position: TimelinePosition
    character_mentions: List[str] = field(default_factory=list)
    inferred_attitudes: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "novel_id": self.novel_id,
            "chapter_index": self.chapter_index,
            "chapter_title": self.chapter_title,
            "paragraph_start": self.paragraph_start,
            "paragraph_end": self.paragraph_end,
            "timeline_position": self.timeline_position.value,
            "character_mentions": self.character_mentions,
            "inferred_attitudes": self.inferred_attitudes
        }


@dataclass
class Claim:
    # An atomic claim extracted from a backstory
    claim_id: str
    text: str
    category: ClaimCategory
    importance: ClaimImportance
    keywords: List[str] = field(default_factory=list)
    character_references: List[str] = field(default_factory=list)
    temporal_indicators: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "category": self.category.value,
            "importance": self.importance.value,
            "keywords": self.keywords,
            "character_references": self.character_references,
            "temporal_indicators": self.temporal_indicators
        }


@dataclass
class Evidence:
    # Retrieved evidence chunk with relevance scores
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    combined_score: float
    timeline_coverage: TimelinePosition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "timeline_coverage": self.timeline_coverage.value
        }


@dataclass
class ClaimEvidencePair:
    # A claim paired with evidence and relationship analysis
    claim: Claim
    evidence: Evidence
    relationship: RelationshipType
    relationship_confidence: float
    causal_consistency: bool
    causal_explanation: str
    gemini_explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "evidence": self.evidence.to_dict(),
            "relationship": self.relationship.value,
            "relationship_confidence": self.relationship_confidence,
            "causal_consistency": self.causal_consistency,
            "causal_explanation": self.causal_explanation,
            "gemini_explanation": self.gemini_explanation
        }


@dataclass
class EvaluationResult:
    # Final evaluation result for a backstory
    backstory_id: str
    novel_id: str
    character: str
    backstory_text: str
    predicted_label: int  # 0=contradict, 1=consistent
    confidence: float
    claims: List[Claim] = field(default_factory=list)
    claim_evidence_pairs: List[ClaimEvidencePair] = field(default_factory=list)
    fatal_contradictions: List[ClaimEvidencePair] = field(default_factory=list)
    reasoning_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backstory_id": self.backstory_id,
            "novel_id": self.novel_id,
            "character": self.character,
            "backstory_text": self.backstory_text,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "claims": [c.to_dict() for c in self.claims],
            "claim_evidence_pairs": [p.to_dict() for p in self.claim_evidence_pairs],
            "fatal_contradictions": [f.to_dict() for f in self.fatal_contradictions],
            "reasoning_summary": self.reasoning_summary
        }


@dataclass
class BackstoryEntry:
    # A backstory entry from CSV
    id: int
    book_name: str
    character: str
    caption: Optional[str]
    content: str
    label: str
    
    @property
    def label_int(self) -> int:
        return 1 if self.label.lower() == "consistent" else 0


@dataclass
class NovelDocument:
    # A complete novel document
    novel_id: str
    title: str
    full_text: str
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    total_chapters: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "novel_id": self.novel_id,
            "title": self.title,
            "total_chapters": self.total_chapters,
            "chapters": self.chapters
        }
