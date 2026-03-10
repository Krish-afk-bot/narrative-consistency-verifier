# Hierarchical chunking with timeline and character tracking
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from backstory_validator.models.schemas import Chunk, NovelDocument
from backstory_validator.pipeline.pathway_tables import PathwayDocumentStore
from backstory_validator.config import (
    TimelinePosition,
    CHUNK_SIZE_PARAGRAPHS,
    CHUNK_OVERLAP_PARAGRAPHS,
    MIN_CHUNK_LENGTH,
    ATTITUDE_KEYWORDS
)


class HierarchicalChunker:
    def __init__(self, store: PathwayDocumentStore, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.store = store
        self.embedding_model = SentenceTransformer(embedding_model)
        self._character_cache: Dict[str, List[str]] = {}
    
    def chunk_novel(self, novel: NovelDocument, 
                    known_characters: Optional[List[str]] = None) -> List[Chunk]:
        # Chunk a novel into paragraph groups with metadata
        chunks = []
        total_chapters = len(novel.chapters)
        
        for chapter in novel.chapters:
            timeline_pos = self._get_timeline_position(
                chapter['index'], total_chapters
            )
            
            paragraphs = self._extract_paragraphs(chapter['text'])
            
            para_groups = self._create_paragraph_groups(
                paragraphs,
                CHUNK_SIZE_PARAGRAPHS,
                CHUNK_OVERLAP_PARAGRAPHS
            )
            
            for group_idx, (para_start, para_end, group_text) in enumerate(para_groups):
                if len(group_text.strip()) < MIN_CHUNK_LENGTH:
                    continue
                
                characters = self._extract_character_mentions(
                    group_text, known_characters
                )
                
                attitudes = self._infer_attitudes(group_text)
                
                embedding = self.embedding_model.encode(group_text).tolist()
                
                chunk_id = f"{novel.novel_id}_ch{chapter['index']}_p{para_start}_{para_end}"
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=group_text,
                    novel_id=novel.novel_id,
                    chapter_index=chapter['index'],
                    chapter_title=chapter['title'],
                    paragraph_start=para_start,
                    paragraph_end=para_end,
                    timeline_position=timeline_pos,
                    character_mentions=characters,
                    inferred_attitudes=attitudes,
                    embedding=embedding
                )
                chunks.append(chunk)
                
                self.store.add_chunk(
                    chunk_id=chunk_id,
                    text=group_text,
                    novel_id=novel.novel_id,
                    chapter_index=chapter['index'],
                    chapter_title=chapter['title'],
                    paragraph_start=para_start,
                    paragraph_end=para_end,
                    timeline_position=timeline_pos.value,
                    character_mentions=characters,
                    attitudes=attitudes,
                    embedding=embedding
                )
        
        return chunks
    
    def _get_timeline_position(self, chapter_index: int, 
                                total_chapters: int) -> TimelinePosition:
        # Determine timeline position based on chapter location
        if total_chapters <= 3:
            if chapter_index == 0:
                return TimelinePosition.EARLY
            elif chapter_index == total_chapters - 1:
                return TimelinePosition.LATE
            else:
                return TimelinePosition.MID
        
        third = total_chapters / 3
        if chapter_index < third:
            return TimelinePosition.EARLY
        elif chapter_index < 2 * third:
            return TimelinePosition.MID
        else:
            return TimelinePosition.LATE
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _create_paragraph_groups(self, paragraphs: List[str],
                                  group_size: int,
                                  overlap: int) -> List[tuple]:
        # Create overlapping paragraph groups
        groups = []
        step = max(1, group_size - overlap)
        
        for i in range(0, len(paragraphs), step):
            end = min(i + group_size, len(paragraphs))
            group_text = '\n\n'.join(paragraphs[i:end])
            groups.append((i, end - 1, group_text))
            
            if end >= len(paragraphs):
                break
        
        return groups
    
    def _extract_character_mentions(self, text: str,
                                     known_characters: Optional[List[str]] = None) -> List[str]:
        # Extract character names from text
        mentions = set()
        text_lower = text.lower()
        
        # Check known characters first
        if known_characters:
            for char in known_characters:
                if char.lower() in text_lower:
                    mentions.add(char)
        
        # Pattern-based name extraction
        name_patterns = [
            r'\b(Mr\.|Mrs\.|Miss|Dr\.|Lord|Lady|Count|Countess|Baron|Captain)\s+([A-Z][a-z]+)',
            r'\b([A-Z][a-z]{2,})\s+([A-Z][a-z]+)',
            r'\b([A-Z][a-z]{3,})\b'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    name = ' '.join(match).strip()
                else:
                    name = match
                if name.lower() not in ['the', 'and', 'but', 'for', 'with', 'this', 'that']:
                    mentions.add(name)
        
        return list(mentions)
    
    def _infer_attitudes(self, text: str) -> Dict[str, float]:
        # Detect emotional attitudes in text
        attitudes = {}
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return attitudes
        
        for attitude, keywords in ATTITUDE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                score = min(1.0, (count / len(keywords)) * (100 / word_count) * 10)
                if score > 0.1:
                    attitudes[attitude] = round(score, 3)
        
        return attitudes


def extract_characters_from_backstories(backstories: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    # Build character list per novel from backstory data
    characters_by_novel: Dict[str, set] = {}
    
    for bs in backstories:
        book = bs.get('book_name', '')
        char = bs.get('character', '')
        novel_id = book.lower().replace(" ", "_")
        
        if novel_id not in characters_by_novel:
            characters_by_novel[novel_id] = set()
        
        if char:
            characters_by_novel[novel_id].add(char)
    
    return {k: list(v) for k, v in characters_by_novel.items()}
