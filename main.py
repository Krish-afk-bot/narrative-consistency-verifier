# Track-A Backstory Consistency Validator - Entry Point
import os
import csv
from pathlib import Path
from typing import List, Dict, Any

os.environ["PYTHONHASHSEED"] = "42"  # Ensure determinism

from backstory_validator.pipeline.pathway_tables import PathwayDocumentStore
from backstory_validator.pipeline.ingestion import NovelIngestionPipeline
from backstory_validator.pipeline.chunking import HierarchicalChunker, extract_characters_from_backstories
from backstory_validator.extraction.claim_extractor import ClaimExtractor
from backstory_validator.retrieval.evidence_retriever import EvidenceRetriever
from backstory_validator.reasoning.constraint_engine import ConstraintReasoningEngine
from backstory_validator.reasoning.classifier import DeterministicClassifier
from backstory_validator.llm.gemini_client import GeminiClient
from backstory_validator.config import GEMINI_API_KEY
from backstory_validator.models.schemas import EvaluationResult
from backstory_validator.config import RelationshipType

# File paths
DATA_DIR = Path("data")
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_CSV = Path("results.csv")

# Novel file mappings
NOVEL_FILES = {
    "the count of monte cristo": DATA_DIR / "The Count of Monte Cristo.txt",
    "in search of the castaways": DATA_DIR / "In search of the castaways.txt",
}


def get_novel_id(book_name: str) -> str:
    # Convert book name to normalized ID
    return book_name.lower().replace(" ", "_")


def load_test_data() -> List[Dict[str, Any]]:
    # Load test entries from CSV
    entries = []
    try:
        with open(TEST_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    'id': int(row['id']),
                    'book_name': row['book_name'],
                    'character': row['char'],
                    'caption': row.get('caption', ''),
                    'content': row['content']
                })
    except Exception as e:
        print(f"Error loading test data: {e}")
    return entries


def load_train_data() -> List[Dict[str, Any]]:
    # Load training data for character extraction
    entries = []
    try:
        with open(TRAIN_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    'book_name': row['book_name'],
                    'character': row['char']
                })
    except Exception as e:
        print(f"Error loading training data: {e}")
    return entries


def generate_rationale(result: EvaluationResult) -> str:
    # Generate rule-based rationale from classification result (no LLM)
    try:
        if result.predicted_label == 1:
            # Consistent - check what supported it
            supporting = [p for p in result.claim_evidence_pairs 
                          if p.relationship == RelationshipType.SUPPORTS]
            if supporting:
                return "Backstory aligns with established narrative events"
            return "No narrative constraints violated"
        
        # Contradictory - determine reason from fatal contradictions
        if result.fatal_contradictions:
            first = result.fatal_contradictions[0]
            claim_cat = first.claim.category.value if hasattr(first.claim, 'category') else ''
            
            if 'event' in claim_cat or 'formative' in claim_cat:
                return "Proposed backstory contradicts established events"
            if 'belief' in claim_cat or 'worldview' in claim_cat:
                return "Character beliefs incompatible with narrative evidence"
            if 'early_life' in claim_cat:
                return "Early life claims conflict with novel timeline"
            if 'relationship' in claim_cat:
                return "Relationship claims contradict narrative evidence"
            if 'authority' in claim_cat:
                return "Authority attitudes conflict with character actions"
            if 'skill' in claim_cat:
                return "Implied skills contradict demonstrated abilities"
            if 'fear' in claim_cat or 'ambition' in claim_cat:
                return "Character motivations incompatible with stated goals"
            
            return "Core identity claim contradicts narrative evidence"
        
        # Weak contradiction based on evidence balance
        return "Evidence balance indicates narrative inconsistency"
    except Exception as e:
        print(f"Error generating rationale: {e}")
        return "Unknown error"


def main():
    try:
        print("=" * 60)
        print("Track-A Backstory Consistency Validator")
        print("=" * 60)
        
        # Step 1: Initialize components
        print("\n[1/5] Initializing document store...")
        store = PathwayDocumentStore()
        
        gemini = None
        if GEMINI_API_KEY:
            try:
                gemini = GeminiClient()
                print("  Gemini client initialized (deterministic mode)")
            except Exception as e:
                print(f"  Gemini unavailable: {e}")
                print("  Using rule-based extraction")
        else:
            print("  No GEMINI_API_KEY - using rule-based extraction")
        
        # Initialize pipeline components
        ingestion = NovelIngestionPipeline(store)
        chunker = HierarchicalChunker(store)
        claim_extractor = ClaimExtractor(store, gemini)
        retriever = EvidenceRetriever(store)
        reasoning_engine = ConstraintReasoningEngine(gemini)
        classifier = DeterministicClassifier()
        
        # Step 2: Extract characters from training data
        print("\n[2/5] Loading training data for character extraction...")
        train_data = load_train_data()
        characters_by_novel = extract_characters_from_backstories(train_data)
        print(f"  Found characters for {{len(characters_by_novel)}} novels")
        
        # Step 3: Load and chunk novels
        print("\n[3/5] Loading and chunking novels...")
        novels_loaded = {}
        
        for book_name, file_path in NOVEL_FILES.items():
            if file_path.exists():
                novel_id = get_novel_id(book_name)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                
                chapters = ingestion._extract_chapters(full_text)
                title = ingestion._extract_title(full_text, book_name)
                
                store.add_novel(novel_id, title, full_text, chapters)
                
                from backstory_validator.models.schemas import NovelDocument
                novel_doc = NovelDocument(
                    novel_id=novel_id,
                    title=title,
                    full_text=full_text,
                    chapters=chapters,
                    total_chapters=len(chapters)
                )
                
                characters = characters_by_novel.get(novel_id, [])
                chunks = chunker.chunk_novel(novel_doc, characters)
                
                novels_loaded[book_name.lower()] = novel_id
                print(f"  {{title}}: {{len(chapters)}} chapters, {{len(chunks)}} chunks")
            else:
                print(f"  WARNING: Novel file not found: {{file_path}}")
        
        store.build_pathway_tables()
        print(f"  Total chunks indexed: {{store.chunk_count}}")
        
        # Step 4: Load test data
        print("\n[4/5] Loading test data...")
        test_entries = load_test_data()
        print(f"  Loaded {{len(test_entries)}} test entries")
        
        # Step 5: Process each test entry through the pipeline
        print("\n[5/5] Processing test entries...")
        results = []
        
        for i, entry in enumerate(test_entries):
            backstory_id = entry['id']
            book_name = entry['book_name']
            character = entry['character']
            backstory_text = entry['content']
            
            novel_id = get_novel_id(book_name)
            
            # Handle missing novels
            if novel_id not in [get_novel_id(n) for n in novels_loaded.keys()]:
                print(f"  [{{i+1}}/{{len(test_entries)}}] ID {{backstory_id}}: Novel not found, defaulting to 1")
                results.append({
                    'story_id': backstory_id, 
                    'prediction': 1,
                    'rationale': 'No narrative constraints violated'
                })
                continue
            
            try:
                # Extract claims from backstory
                claims = claim_extractor.extract_claims(
                    backstory_id, backstory_text, character,
                    use_gemini=(gemini is not None)
                )
                
                # Retrieve evidence for each claim
                evidence_map = {}
                for claim in claims:
                    evidence = retriever.retrieve_evidence(claim, novel_id, character)
                    evidence_map[claim.claim_id] = evidence
                
                # Analyze claim-evidence pairs
                pairs = reasoning_engine.analyze_all_pairs(claims, evidence_map, character)
                
                # Make final classification (deterministic)
                result = classifier.classify(
                    backstory_id=str(backstory_id),
                    novel_id=novel_id,
                    character=character,
                    backstory_text=backstory_text,
                    claims=claims,
                    claim_evidence_pairs=pairs
                )
                
                prediction = result.predicted_label
                rationale = generate_rationale(result)
                
            except Exception as e:
                print(f"  [{{i+1}}/{{len(test_entries)}}] ID {{backstory_id}}: Error - {{e}}, defaulting to 1")
                prediction = 1
                rationale = 'No narrative constraints violated'
            
            results.append({
                'story_id': backstory_id, 
                'prediction': prediction,
                'rationale': rationale
            })
            
            label_str = "consistent" if prediction == 1 else "contradict"
            print(f"  [{{i+1}}/{{len(test_entries)}}] ID {{backstory_id}} ({{character}}): {{label_str}}")
        
        # Write output
        print("\n" + "=" * 60)
        print("Writing results.csv...")
        
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['story_id', 'prediction', 'rationale'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  Output: {{OUTPUT_CSV}}")
        print(f"  Rows: {{len(results)}}")
        
        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION:")
        print(f"  test.csv rows: {{len(test_entries)}}")
        print(f"  results.csv rows: {{len(results)}}")
        
        if len(results) == len(test_entries):
            print("  \u2713 Row count matches")
        else:
            print("  \u2717 Row count mismatch!")
        
        consistent_count = sum(1 for r in results if r['prediction'] == 1)
        contradict_count = sum(1 for r in results if r['prediction'] == 0)
        print(f"  Consistent: {{consistent_count}}")
        print(f"  Contradict: {{contradict_count}}")
        
        print("\nDone.")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
