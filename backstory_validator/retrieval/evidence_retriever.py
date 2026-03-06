# Evidence retrieval with semantic and keyword matching
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backstory_validator.models.schemas import Claim, Chunk, Evidence
from backstory_validator.pipeline.pathway_tables import PathwayDocumentStore
from backstory_validator.config import (
    TimelinePosition,
    TOP_K_EXCERPTS,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT
)


class EvidenceRetriever:
    def __init__(self, store: PathwayDocumentStore,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.store = store
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def retrieve_evidence(self, claim: Claim, novel_id: str,
                          character: str,
                          top_k: int = TOP_K_EXCERPTS,
                          ensure_timeline_coverage: bool = True) -> List[Evidence]:
        # Retrieve relevant evidence chunks for a claim
        all_chunks = self.store.get_chunks_for_novel(novel_id)
        
        if not all_chunks:
            return []
        
        # Score all chunks
        scored_chunks = []
        for chunk_data in all_chunks:
            scores = self._score_chunk(claim, chunk_data, character)
            scored_chunks.append((chunk_data, scores))
        
        # Sort by combined score
        scored_chunks.sort(key=lambda x: x[1]['combined'], reverse=True)
        
        # Select with timeline coverage if requested
        if ensure_timeline_coverage:
            selected = self._select_with_timeline_coverage(
                scored_chunks, top_k
            )
        else:
            selected = scored_chunks[:top_k]
        
        # Convert to Evidence objects
        evidence_list = []
        for chunk_data, scores in selected:
            chunk = self._chunk_from_data(chunk_data)
            evidence = Evidence(
                chunk=chunk,
                semantic_score=scores['semantic'],
                keyword_score=scores['keyword'],
                combined_score=scores['combined'],
                timeline_coverage=TimelinePosition(chunk_data['timeline_position'])
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def _score_chunk(self, claim: Claim, chunk_data: Dict[str, Any],
                     character: str) -> Dict[str, float]:
        # Compute relevance scores for a chunk
        
        # Semantic similarity
        if claim.embedding and chunk_data.get('embedding'):
            claim_emb = np.array(claim.embedding).reshape(1, -1)
            chunk_emb = np.array(chunk_data['embedding']).reshape(1, -1)
            semantic_score = float(cosine_similarity(claim_emb, chunk_emb)[0][0])
        else:
            claim_emb = self.embedding_model.encode(claim.text).reshape(1, -1)
            chunk_emb = self.embedding_model.encode(chunk_data['text']).reshape(1, -1)
            semantic_score = float(cosine_similarity(claim_emb, chunk_emb)[0][0])
        
        # Other scoring factors
        keyword_score = self._compute_keyword_score(claim, chunk_data)
        character_score = self._compute_character_score(character, chunk_data)
        timeline_score = self._compute_timeline_score(claim, chunk_data)
        
        # Weighted combination
        combined = (
            SEMANTIC_WEIGHT * semantic_score +
            KEYWORD_WEIGHT * keyword_score +
            0.15 * character_score +
            0.1 * timeline_score
        )
        
        return {
            'semantic': semantic_score,
            'keyword': keyword_score,
            'character': character_score,
            'timeline': timeline_score,
            'combined': combined
        }
    
    def _compute_keyword_score(self, claim: Claim, 
                                chunk_data: Dict[str, Any]) -> float:
        # Score based on keyword overlap
        if not claim.keywords:
            return 0.0
        
        chunk_text_lower = chunk_data['text'].lower()
        matches = sum(1 for kw in claim.keywords if kw.lower() in chunk_text_lower)
        
        return matches / len(claim.keywords)
    
    def _compute_character_score(self, character: str,
                                  chunk_data: Dict[str, Any]) -> float:
        # Score based on character presence
        character_lower = character.lower()
        
        mentions = chunk_data.get('character_mentions', [])
        if any(character_lower in m.lower() for m in mentions):
            return 1.0
        
        if character_lower in chunk_data['text'].lower():
            return 0.8
        
        # Check partial name match
        name_parts = character.split()
        for part in name_parts:
            if len(part) > 2 and part.lower() in chunk_data['text'].lower():
                return 0.5
        
        return 0.0
    
    def _compute_timeline_score(self, claim: Claim,
                                 chunk_data: Dict[str, Any]) -> float:
        # Score based on timeline alignment
        temporal = claim.temporal_indicators
        chunk_timeline = chunk_data.get('timeline_position', 'mid')
        
        early_indicators = ['childhood', 'young', 'boy', 'girl', 'birth', 'early', 'first']
        late_indicators = ['finally', 'eventually', 'later', 'death', 'end', 'last']
        
        has_early = any(ind in ' '.join(temporal).lower() for ind in early_indicators)
        has_late = any(ind in ' '.join(temporal).lower() for ind in late_indicators)
        
        if has_early and chunk_timeline == 'early':
            return 1.0
        elif has_late and chunk_timeline == 'late':
            return 1.0
        elif not has_early and not has_late and chunk_timeline == 'mid':
            return 0.7
        
        return 0.3
    
    def _select_with_timeline_coverage(self, 
                                        scored_chunks: List[Tuple[Dict, Dict]],
                                        top_k: int) -> List[Tuple[Dict, Dict]]:
        # Select chunks ensuring coverage across timeline positions
        selected = []
        timeline_counts = {'early': 0, 'mid': 0, 'late': 0}
        min_per_timeline = max(1, top_k // 4)
        
        # First pass: ensure minimum coverage per timeline
        for chunk_data, scores in scored_chunks:
            timeline = chunk_data.get('timeline_position', 'mid')
            if timeline_counts[timeline] < min_per_timeline:
                selected.append((chunk_data, scores))
                timeline_counts[timeline] += 1
            
            if len(selected) >= top_k:
                break
        
        # Second pass: fill remaining slots with best scores
        if len(selected) < top_k:
            selected_ids = {c[0]['chunk_id'] for c in selected}
            for chunk_data, scores in scored_chunks:
                if chunk_data['chunk_id'] not in selected_ids:
                    selected.append((chunk_data, scores))
                    if len(selected) >= top_k:
                        break
        
        # Re-sort by score
        selected.sort(key=lambda x: x[1]['combined'], reverse=True)
        
        return selected
    
    def _chunk_from_data(self, chunk_data: Dict[str, Any]) -> Chunk:
        # Convert dict to Chunk object
        return Chunk(
            chunk_id=chunk_data['chunk_id'],
            text=chunk_data['text'],
            novel_id=chunk_data['novel_id'],
            chapter_index=chunk_data['chapter_index'],
            chapter_title=chunk_data['chapter_title'],
            paragraph_start=chunk_data['paragraph_start'],
            paragraph_end=chunk_data['paragraph_end'],
            timeline_position=TimelinePosition(chunk_data['timeline_position']),
            character_mentions=chunk_data.get('character_mentions', []),
            inferred_attitudes=chunk_data.get('attitudes', {}),
            embedding=chunk_data.get('embedding')
        )
    
    def retrieve_by_character_timeline(self, novel_id: str, character: str,
                                        timeline: TimelinePosition,
                                        top_k: int = 5) -> List[Chunk]:
        # Retrieve chunks by character and timeline
        timeline_chunks = self.store.get_chunks_by_timeline(
            novel_id, timeline.value
        )
        
        character_lower = character.lower()
        relevant = []
        for chunk_data in timeline_chunks:
            if character_lower in chunk_data['text'].lower():
                relevant.append(chunk_data)
            elif any(character_lower in m.lower() 
                     for m in chunk_data.get('character_mentions', [])):
                relevant.append(chunk_data)
        
        return [self._chunk_from_data(c) for c in relevant[:top_k]]
