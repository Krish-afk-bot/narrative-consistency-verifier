# Gemini API client for claim extraction and explanation
import os
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from backstory_validator.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        self.client = genai.Client(api_key=self.api_key)
        self._model_name = model
    
    def extract_claims(self, backstory: str, character: str) -> List[Dict[str, Any]]:
        # Extract atomic claims from backstory using Gemini
        prompt = f"""Analyze this character backstory and extract atomic, testable claims.

CHARACTER: {character}

BACKSTORY:
{backstory}

Extract each distinct claim about the character. For each claim, identify:
1. The claim text (a single, specific assertion)
2. Category: one of [early_life_conditions, formative_experiences, beliefs_worldview, fears_ambitions, authority_attitudes, implied_skills, relationships, specific_events]
3. Importance: one of [core, significant, surface]
4. Keywords: key terms for matching
5. Temporal indicators: any time references

Format your response as a JSON array of objects with keys: text, category, importance, keywords, temporal_indicators

Response (JSON array only):"""

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.0  # Deterministic output
                )
            )
            
            response_text = response.text.strip()
            # Clean markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            import json
            claims = json.loads(response_text)
            return claims
            
        except Exception as e:
            print(f"Error extracting claims: {e}")
            # Fallback to single claim
            return [{
                "text": backstory,
                "category": "formative_experiences",
                "importance": "core",
                "keywords": backstory.split()[:10],
                "temporal_indicators": []
            }]
    
    def explain_relationship(self, claim: str, excerpt: str, 
                             character: str) -> Dict[str, Any]:
        # Explain how evidence relates to a claim (NOT for classification)
        prompt = f"""Analyze how this novel excerpt relates to a character backstory claim.

CHARACTER: {character}

BACKSTORY CLAIM:
{claim}

NOVEL EXCERPT:
{excerpt}

Provide analysis of:
1. What specific information in the excerpt is relevant to the claim
2. Whether the excerpt provides evidence that supports, limits, or conflicts with the claim
3. Any causal implications
4. Key quotes from the excerpt

Format your response as JSON with keys: relevant_info, evidence_analysis, causal_implications, key_quotes

Response (JSON only):"""

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.0
                )
            )
            
            response_text = response.text.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            import json
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error explaining relationship: {e}")
            return {
                "relevant_info": "Analysis failed",
                "evidence_analysis": str(e),
                "causal_implications": "Unknown",
                "key_quotes": []
            }
    
    def compare_interpretations(self, claim: str, 
                                 supporting_evidence: List[str],
                                 conflicting_evidence: List[str],
                                 character: str) -> str:
        # Compare supporting vs conflicting evidence
        supporting_text = "\n---\n".join(supporting_evidence[:3])
        conflicting_text = "\n---\n".join(conflicting_evidence[:3])
        
        prompt = f"""Compare the evidence for and against this character backstory claim.

CHARACTER: {character}

CLAIM:
{claim}

SUPPORTING EVIDENCE:
{supporting_text}

CONFLICTING EVIDENCE:
{conflicting_text}

Provide a balanced analysis.

Response:"""

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.0
                )
            )
            return response.text.strip()
            
        except Exception as e:
            print(f"Error comparing interpretations: {e}")
            return f"Comparison failed: {e}"
    
    def synthesize_evidence_dossier(self, character: str, backstory: str,
                                     claim_evidence_pairs: List[Dict[str, Any]]) -> str:
        # Generate evidence summary dossier
        pairs_summary = []
        for pair in claim_evidence_pairs[:10]:
            pairs_summary.append(f"""
CLAIM: {pair.get('claim_text', 'N/A')}
RELATIONSHIP: {pair.get('relationship', 'N/A')}
EXCERPT: {pair.get('excerpt_text', 'N/A')[:500]}...
""")
        
        pairs_text = "\n---\n".join(pairs_summary)
        
        prompt = f"""Create an evidence dossier summarizing the analysis of this character backstory.

CHARACTER: {character}

BACKSTORY:
{backstory}

ANALYZED CLAIM-EVIDENCE PAIRS:
{pairs_text}

Create a structured dossier.

Response:"""

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.0
                )
            )
            return response.text.strip()
            
        except Exception as e:
            print(f"Error synthesizing dossier: {e}")
            return f"Dossier synthesis failed: {e}"
