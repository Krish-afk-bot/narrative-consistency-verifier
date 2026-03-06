# Configuration settings for backstory validator
import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Gemini API settings
GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_MAX_TOKENS = 8192

# Chunking parameters
CHUNK_SIZE_PARAGRAPHS = 5
CHUNK_OVERLAP_PARAGRAPHS = 2
MIN_CHUNK_LENGTH = 100

# Retrieval parameters
TOP_K_EXCERPTS = 10
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4


# Timeline position in narrative
class TimelinePosition(str, Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"


# Relationship between claim and evidence
class RelationshipType(str, Enum):
    SUPPORTS = "supports"
    CONSTRAINS = "constrains"
    CONTRADICTS = "contradicts"
    CONTEXTUAL = "contextual"


# Importance level of claims
class ClaimImportance(str, Enum):
    CORE = "core"           # Identity-defining claims
    SIGNIFICANT = "significant"  # Important but not fatal
    SURFACE = "surface"     # Minor details


# Categories of backstory claims
class ClaimCategory(str, Enum):
    EARLY_LIFE = "early_life_conditions"
    FORMATIVE = "formative_experiences"
    BELIEFS = "beliefs_worldview"
    FEARS_AMBITIONS = "fears_ambitions"
    AUTHORITY = "authority_attitudes"
    SKILLS = "implied_skills"
    RELATIONSHIPS = "relationships"
    EVENTS = "specific_events"


# Classification thresholds
CONTRADICTION_THRESHOLD = 0.7
SUPPORT_THRESHOLD = 0.6
CORE_CLAIM_FATAL_THRESHOLD = 0.8

# Attitude detection keywords
ATTITUDE_KEYWORDS = {
    "fear": ["afraid", "terrified", "dread", "panic", "horror", "frightened", "scared"],
    "ambition": ["aspire", "desire", "goal", "dream", "strive", "ambition", "determined"],
    "loyalty": ["faithful", "devoted", "loyal", "allegiance", "trust", "bond", "commitment"],
    "trust": ["believe", "faith", "confidence", "rely", "depend", "trust"],
    "distrust": ["suspect", "doubt", "wary", "suspicious", "mistrust", "betray"],
    "anger": ["rage", "fury", "wrath", "hatred", "vengeance", "revenge"],
    "love": ["love", "affection", "adore", "cherish", "beloved", "passion"],
    "grief": ["mourn", "sorrow", "loss", "grief", "lament", "weep"]
}

# Novel file mappings
NOVEL_FILES = {
    "The Count of Monte Cristo": "The Count of Monte Cristo.txt",
    "In Search of the Castaways": "In search of the castaways.txt"
}
