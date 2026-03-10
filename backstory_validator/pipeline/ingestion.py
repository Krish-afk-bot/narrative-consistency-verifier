# Novel and backstory ingestion pipeline
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from backstory_validator.models.schemas import NovelDocument, BackstoryEntry
from backstory_validator.pipeline.pathway_tables import PathwayDocumentStore
from backstory_validator.config import NOVEL_FILES


class NovelIngestionPipeline:
    def __init__(self, store: PathwayDocumentStore):
        self.store = store
        # Chapter detection patterns
        self._chapter_patterns = [
            r'^Chapter\s+(\d+)[\.:\s]+(.+?)$',
            r'^([IVXLCDM]+)[\.:\s]+(.+?)$',
            r'^CHAPTER\s+(\d+|[IVXLCDM]+)\s*$',
            r'^(\d+|[IVXLCDM]+)\.\s*$'
        ]
    
    def ingest_novel(self, file_path: str, novel_id: str) -> NovelDocument:
        # Load and parse a novel file
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        title = self._extract_title(full_text, novel_id)
        chapters = self._extract_chapters(full_text)
        
        doc = NovelDocument(
            novel_id=novel_id,
            title=title,
            full_text=full_text,
            chapters=chapters,
            total_chapters=len(chapters)
        )
        
        self.store.add_novel(
            novel_id=novel_id,
            title=title,
            full_text=full_text,
            chapters=chapters
        )
        
        return doc
    
    def _extract_title(self, text: str, fallback: str) -> str:
        # Extract title from novel text
        lines = text.split('\n')
        for line in lines[:100]:
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                if 'gutenberg' in line.lower():
                    continue
                if 'ebook' in line.lower():
                    continue
                if line.startswith('***'):
                    continue
                if 'title:' in line.lower():
                    return line.split(':', 1)[1].strip()
                if line.isupper() or (line[0].isupper() and not line.startswith('Chapter')):
                    if not any(skip in line.lower() for skip in ['contents', 'volume', 'by ', 'author']):
                        return line
        return fallback
    
    def _extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        # Split novel into chapters
        chapters = []
        lines = text.split('\n')
        
        chapter_starts = []
        current_pos = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern in self._chapter_patterns:
                match = re.match(pattern, stripped, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    chapter_num = groups[0] if groups else str(len(chapter_starts) + 1)
                    chapter_title = groups[1].strip() if len(groups) > 1 and groups[1] else f"Chapter {chapter_num}"
                    
                    chapter_starts.append({
                        'line_index': i,
                        'char_pos': current_pos,
                        'number': chapter_num,
                        'title': chapter_title
                    })
                    break
            current_pos += len(line) + 1
        
        for i, start in enumerate(chapter_starts):
            end_pos = chapter_starts[i + 1]['char_pos'] if i + 1 < len(chapter_starts) else len(text)
            chapter_text = text[start['char_pos']:end_pos]
            
            chapters.append({
                'index': i,
                'number': start['number'],
                'title': start['title'],
                'start_pos': start['char_pos'],
                'end_pos': end_pos,
                'text': chapter_text,
                'line_start': start['line_index']
            })
        
        # Fallback if no chapters found
        if not chapters:
            chapters.append({
                'index': 0,
                'number': '1',
                'title': 'Full Text',
                'start_pos': 0,
                'end_pos': len(text),
                'text': text,
                'line_start': 0
            })
        
        return chapters
    
    def ingest_backstories(self, csv_path: str) -> List[BackstoryEntry]:
        # Load backstories from CSV
        df = pd.read_csv(csv_path)
        entries = []
        
        for _, row in df.iterrows():
            entry = BackstoryEntry(
                id=int(row['id']),
                book_name=str(row['book_name']),
                character=str(row['char']),
                caption=str(row['caption']) if pd.notna(row['caption']) else None,
                content=str(row['content']),
                label=str(row['label'])
            )
            entries.append(entry)
            
            self.store.add_backstory(
                backstory_id=entry.id,
                book_name=entry.book_name,
                character=entry.character,
                caption=entry.caption or "",
                content=entry.content,
                label=entry.label
            )
        
        return entries
    
    def ingest_all_novels(self, data_dir: str = ".") -> Dict[str, NovelDocument]:
        # Load all configured novels
        novels = {}
        data_path = Path(data_dir)
        
        for novel_name, file_name in NOVEL_FILES.items():
            file_path = data_path / file_name
            if file_path.exists():
                novel_id = novel_name.lower().replace(" ", "_")
                doc = self.ingest_novel(str(file_path), novel_id)
                novels[novel_name] = doc
                print(f"Ingested '{novel_name}': {doc.total_chapters} chapters")
            else:
                print(f"Warning: Novel file not found: {file_path}")
        
        return novels


def get_novel_id_for_book(book_name: str) -> str:
    return book_name.lower().replace(" ", "_")
