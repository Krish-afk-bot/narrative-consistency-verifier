# Claim extraction from backstory text
import re
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from backstory_validator.models.schemas import Claim
from backstory_validator.pipeline.pathway_tables import PathwayDocumentStore
from backstory_validator.llm.gemini_client import GeminiClient
from backstory_validator.config import ClaimCategory, ClaimImportance


class ClaimExtractor:
    def __init__(self, store: PathwayDocumentStore,
                 gemini_client: Optional[GeminiClient] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.store = store
        self.gemini = gemini_client
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Temporal detection patterns
        self._temporal_patterns = [
            r'\b(as a child|in childhood|when young|as a boy|as a girl)\b',
            r'\b(at age \d+|at \d+ years old|when he was \d+|when she was \d+)\b',
            r'\b(before|after|during|while|until|since)\b',
            r'\b(early|later|eventually|finally|first|then)\b',
            r'\b(youth|boyhood|girlhood|adolescence|adulthood)\b',
            r'\b(\d{4}|18\d{2}|19\d{2})\b',
        ]
        
        # Category detection keywords
        self._category_keywords = {
            ClaimCategory.EARLY_LIFE: ['born', 'childhood', 'parents', 'family', 'grew up', 'raised'],
            ClaimCategory.FORMATIVE: ['learned', 'taught', 'experienced', 'witnessed', 'trauma', 'event'],
            ClaimCategory.BELIEFS: ['believed', 'thought', 'felt', 'conviction', 'faith', 'ideology'],
            ClaimCategory.FEARS_AMBITIONS: ['feared', 'dreamed', 'hoped', 'ambition', 'goal', 'dread'],
            ClaimCategory.AUTHORITY: ['authority', 'power', 'government', 'law', 'rebellion', 'obedience'],
            ClaimCategory.SKILLS: ['skill', 'ability', 'trained', 'mastered', 'learned', 'education'],
            ClaimCategory.RELATIONSHIPS: ['friend', 'enemy', 'lover', 'mentor', 'ally', 'rival'],
            ClaimCategory.EVENTS: ['happened', 'occurred', 'incident', 'battle', 'meeting', 'discovery']
        }
        
        # Importance indicators
        self._core_indicators = [
            'identity', 'fundamental', 'defining', 'core', 'essential',
            'trauma', 'death', 'birth', 'origin', 'belief', 'conviction'
        ]
        self._surface_indicators = [
            'minor', 'small', 'briefly', 'once', 'occasionally', 'sometimes'
        ]
    
    def extract_claims(self, backstory_id: int, backstory_text: str,
                           character: str, use_gemini: bool = True) -> List[Claim]:
        # Extract claims using Gemini or rule-based fallback
        claims = []
        
        if use_gemini and self.gemini:
            raw_claims = self.gemini.extract_claims(backstory_text, character)
            
            for i, raw in enumerate(raw_claims):
                claim = self._process_raw_claim(
                    backstory_id, i, raw, character
                )
                claims.append(claim)
        else:
            # Rule-based extraction fallback
            claims = self._rule_based_extraction(
                backstory_id, backstory_text, character
            )
        
        # Store claims in document store
        for claim in claims:
            self.store.add_claim(
                claim_id=claim.claim_id,
                backstory_id=backstory_id,
                text=claim.text,
                category=claim.category.value,
                importance=claim.importance.value,
                keywords=claim.keywords,
                character_refs=claim.character_references,
                temporal=claim.temporal_indicators,
                embedding=claim.embedding or []
            )
        
        return claims
    
    def _process_raw_claim(self, backstory_id: int, index: int,
                           raw: Dict[str, Any], character: str) -> Claim:
        # Process Gemini output into Claim object
        text = raw.get('text', '')
        
        claim_id = str(uuid.uuid4())
        
        category_str = raw.get('category', 'formative_experiences')
        try:
            category = ClaimCategory(category_str)
        except ValueError:
            category = self._infer_category(text)
        
        importance_str = raw.get('importance', 'significant')
        try:
            importance = ClaimImportance(importance_str)
        except ValueError:
            importance = self._infer_importance(text)
        
        keywords = raw.get('keywords', [])
        if not keywords:
            keywords = self._extract_keywords(text)
        
        temporal = raw.get('temporal_indicators', [])
        if not temporal:
            temporal = self._extract_temporal_indicators(text)
        
        embedding = self.embedding_model.encode(text).tolist()
        
        return Claim(
            claim_id=claim_id,
            text=text,
            category=category,
            importance=importance,
            keywords=keywords,
            character_references=[character],
            temporal_indicators=temporal,
            embedding=embedding
        )
    
    def _rule_based_extraction(self, backstory_id: int, 
                                backstory_text: str,
                                character: str) -> List[Claim]:
        # Extract claims using sentence splitting
        claims = []
        
        sentences = self._split_into_sentences(backstory_text)
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
            
            claim_id = str(uuid.uuid4())
            category = self._infer_category(sentence)
            importance = self._infer_importance(sentence)
            keywords = self._extract_keywords(sentence)
            temporal = self._extract_temporal_indicators(sentence)
            embedding = self.embedding_model.encode(sentence).tolist()
            
            claim = Claim(
                claim_id=claim_id,
                text=sentence,
                category=category,
                importance=importance,
                keywords=keywords,
                character_references=[character],
                temporal_indicators=temporal,
                embedding=embedding
            )
            claims.append(claim)
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        # Split text into sentences, handling abbreviations
        text = re.sub(r'Mr\.', 'Mr', text)
        text = re.sub(r'Mrs\.', 'Mrs', text)
        text = re.sub(r'Dr\.', 'Dr', text)
        text = re.sub(r'St\.', 'St', text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('Mr', 'Mr.').replace('Mrs', 'Mrs.')
                     .replace('Dr', 'Dr.').replace('St', 'St.')
                     for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _infer_category(self, text: str) -> ClaimCategory:
        # Infer claim category from keywords
        text_lower = text.lower()
        
        best_category = ClaimCategory.FORMATIVE
        best_score = 0
        
        for category, keywords in self._category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _infer_importance(self, text: str) -> ClaimImportance:
        # Infer claim importance from indicators
        text_lower = text.lower()
        
        core_count = sum(1 for ind in self._core_indicators if ind in text_lower)
        surface_count = sum(1 for ind in self._surface_indicators if ind in text_lower)
        
        if core_count > surface_count:
            return ClaimImportance.CORE
        elif surface_count > core_count:
            return ClaimImportance.SURFACE
        else:
            return ClaimImportance.SIGNIFICANT
    
    def _extract_keywords(self, text: str) -> List[str]:
        # Extract significant keywords from text
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                     'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were',
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'must', 'shall', 'can', 'this', 'that', 'these', 'those',
                     'he', 'she', 'it', 'they', 'we', 'you', 'his', 'her', 'its',
                     'their', 'our', 'your', 'who', 'which', 'what', 'when', 'where'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        
        return unique[:15]
    
    def _extract_temporal_indicators(self, text: str) -> List[str]:
        # Extract time-related phrases
        indicators = []
        
        for pattern in self._temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return list(set(indicators))
