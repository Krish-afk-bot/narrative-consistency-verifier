# Pathway-compatible document store for novels and chunks
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class NovelSchema:
    novel_id: str
    title: str
    full_text: str
    chapters_json: str
    total_chapters: int


@dataclass
class ChunkSchema:
    chunk_id: str
    text: str
    novel_id: str
    chapter_index: int
    chapter_title: str
    paragraph_start: int
    paragraph_end: int
    timeline_position: str
    character_mentions_json: str
    attitudes_json: str
    embedding_json: str


@dataclass
class BackstorySchema:
    backstory_id: int
    book_name: str
    character: str
    caption: str
    content: str
    label: str


@dataclass
class ClaimSchema:
    claim_id: str
    backstory_id: int
    text: str
    category: str
    importance: str
    keywords_json: str
    character_refs_json: str
    temporal_json: str
    embedding_json: str


class PathwayDocumentStore:
    # In-memory document store (Pathway abstraction for Windows compatibility)
    
    def __init__(self):
        self._novels: Dict[str, Dict[str, Any]] = {}
        self._chunks: Dict[str, Dict[str, Any]] = {}
        self._backstories: Dict[int, Dict[str, Any]] = {}
        self._claims: Dict[str, Dict[str, Any]] = {}
        self._tables_built = False
    
    def add_novel(self, novel_id: str, title: str, full_text: str, 
                  chapters: List[Dict[str, Any]]) -> None:
        self._novels[novel_id] = {
            "novel_id": novel_id,
            "title": title,
            "full_text": full_text,
            "chapters_json": json.dumps(chapters),
            "total_chapters": len(chapters)
        }
    
    def add_chunk(self, chunk_id: str, text: str, novel_id: str,
                  chapter_index: int, chapter_title: str,
                  paragraph_start: int, paragraph_end: int,
                  timeline_position: str, character_mentions: List[str],
                  attitudes: Dict[str, float], embedding: List[float]) -> None:
        self._chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "novel_id": novel_id,
            "chapter_index": chapter_index,
            "chapter_title": chapter_title,
            "paragraph_start": paragraph_start,
            "paragraph_end": paragraph_end,
            "timeline_position": timeline_position,
            "character_mentions_json": json.dumps(character_mentions),
            "attitudes_json": json.dumps(attitudes),
            "embedding_json": json.dumps(embedding)
        }
    
    def add_backstory(self, backstory_id: int, book_name: str, character: str,
                      caption: str, content: str, label: str) -> None:
        self._backstories[backstory_id] = {
            "backstory_id": backstory_id,
            "book_name": book_name,
            "character": character,
            "caption": caption or "",
            "content": content,
            "label": label
        }
    
    def add_claim(self, claim_id: str, backstory_id: int, text: str,
                  category: str, importance: str, keywords: List[str],
                  character_refs: List[str], temporal: List[str],
                  embedding: List[float]) -> None:
        self._claims[claim_id] = {
            "claim_id": claim_id,
            "backstory_id": backstory_id,
            "text": text,
            "category": category,
            "importance": importance,
            "keywords_json": json.dumps(keywords),
            "character_refs_json": json.dumps(character_refs),
            "temporal_json": json.dumps(temporal),
            "embedding_json": json.dumps(embedding)
        }
    
    def build_pathway_tables(self) -> None:
        # Build indexes (simulates Pathway table building)
        self._tables_built = True
    
    def get_novel(self, novel_id: str) -> Optional[Dict[str, Any]]:
        novel = self._novels.get(novel_id)
        if novel:
            result = novel.copy()
            result["chapters"] = json.loads(result["chapters_json"])
            del result["chapters_json"]
            return result
        return None
    
    def get_chunks_for_novel(self, novel_id: str) -> List[Dict[str, Any]]:
        # Get all chunks for a novel, sorted by position
        chunks = []
        for chunk in self._chunks.values():
            if chunk["novel_id"] == novel_id:
                result = chunk.copy()
                result["character_mentions"] = json.loads(result["character_mentions_json"])
                result["attitudes"] = json.loads(result["attitudes_json"])
                result["embedding"] = json.loads(result["embedding_json"])
                del result["character_mentions_json"]
                del result["attitudes_json"]
                del result["embedding_json"]
                chunks.append(result)
        return sorted(chunks, key=lambda x: (x["chapter_index"], x["paragraph_start"]))
    
    def get_chunks_by_character(self, novel_id: str, character: str) -> List[Dict[str, Any]]:
        # Filter chunks by character mention
        all_chunks = self.get_chunks_for_novel(novel_id)
        character_lower = character.lower()
        return [
            c for c in all_chunks 
            if any(character_lower in m.lower() for m in c["character_mentions"])
            or character_lower in c["text"].lower()
        ]
    
    def get_chunks_by_timeline(self, novel_id: str, position: str) -> List[Dict[str, Any]]:
        # Filter chunks by timeline position
        all_chunks = self.get_chunks_for_novel(novel_id)
        return [c for c in all_chunks if c["timeline_position"] == position]
    
    def get_backstory(self, backstory_id: int) -> Optional[Dict[str, Any]]:
        return self._backstories.get(backstory_id)
    
    def get_all_backstories(self) -> List[Dict[str, Any]]:
        return list(self._backstories.values())
    
    def get_claims_for_backstory(self, backstory_id: int) -> List[Dict[str, Any]]:
        claims = []
        for claim in self._claims.values():
            if claim["backstory_id"] == backstory_id:
                result = claim.copy()
                result["keywords"] = json.loads(result["keywords_json"])
                result["character_refs"] = json.loads(result["character_refs_json"])
                result["temporal"] = json.loads(result["temporal_json"])
                result["embedding"] = json.loads(result["embedding_json"])
                del result["keywords_json"]
                del result["character_refs_json"]
                del result["temporal_json"]
                del result["embedding_json"]
                claims.append(result)
        return claims
    
    @property
    def novel_count(self) -> int:
        return len(self._novels)
    
    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
    
    @property
    def backstory_count(self) -> int:
        return len(self._backstories)
    
    @property
    def claim_count(self) -> int:
        return len(self._claims)
