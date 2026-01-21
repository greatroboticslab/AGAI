"""
Question-Answering RAG for Plant Diagnostic System

This module provides targeted retrieval for follow-up questions
after initial diagnosis. Unlike injecting context into prompts,
this returns specific answers to specific questions.

Usage:
    from knowledge_graph.qa_retriever import DiseaseQA
    
    qa = DiseaseQA()
    answer = qa.answer_question("gray_mold", "how long until recovery?")
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DiseaseQA:
    """Question-answering system for disease-specific queries."""
    
    def __init__(self, kb_path: Optional[str] = None):
        if kb_path is None:
            kb_path = Path(__file__).parent / "disease_knowledge_base.json"
        
        self.kb_path = Path(kb_path)
        self.knowledge_base = self._load_knowledge_base()
        
        # Question patterns mapped to knowledge base sections
        self.question_patterns = {
            "recovery": {
                "patterns": [
                    r"how long", r"recover", r"time", r"days", r"weeks", 
                    r"when will", r"timeline", r"duration"
                ],
                "section": "recovery_timeline",
                "prefix": "Recovery timeline"
            },
            "cause": {
                "patterns": [
                    r"why", r"cause", r"reason", r"how did", r"what caused",
                    r"happen", r"origin", r"source"
                ],
                "section": "causes",
                "prefix": "Common causes"
            },
            "treatment": {
                "patterns": [
                    r"treat", r"fix", r"cure", r"help", r"save", r"remedy",
                    r"what.*do", r"how.*fix", r"solution", r"spray", r"fungicide"
                ],
                "section": "treatments",
                "prefix": "Recommended treatments"
            },
            "prevention": {
                "patterns": [
                    r"prevent", r"avoid", r"stop.*from", r"future", r"protect",
                    r"next time", r"again"
                ],
                "section": "prevention",
                "prefix": "Prevention"
            },
            "symptoms": {
                "patterns": [
                    r"symptom", r"sign", r"look like", r"identify", r"recognize",
                    r"tell if", r"know if", r"indicator"
                ],
                "section": "visual_indicators",
                "prefix": "Key visual indicators"
            },
            "severity": {
                "patterns": [
                    r"serious", r"severe", r"bad", r"dangerous", r"fatal",
                    r"die", r"kill", r"save", r"too late"
                ],
                "section": "severity",
                "prefix": "Severity level"
            },
            "difference": {
                "patterns": [
                    r"difference", r"vs", r"versus", r"compare", r"distinguish",
                    r"tell.*apart", r"similar", r"confuse"
                ],
                "section": "_compare",  # Special handler
                "prefix": "Comparison"
            }
        }
        
        # Label normalization
        self._label_map = {
            "gray": "gray_mold", "grey": "gray_mold", "botrytis": "gray_mold",
            "white": "white_mold", "sclerotinia": "white_mold",
            "root": "root_rot", "phytophthora": "root_rot",
            "frost": "frost_injury", "cold": "frost_injury", "freeze": "frost_injury",
            "dry": "drought", "water": "overwatering", "wet": "overwatering",
            "overwater": "overwatering"
        }
    
    def _load_knowledge_base(self) -> Dict:
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"diseases": {}, "general_best_practices": {}}
    
    def _normalize_label(self, label: str) -> str:
        normalized = label.lower().strip().replace(" ", "_").replace("-", "_")
        return self._label_map.get(normalized, normalized)
    
    def _detect_question_type(self, question: str) -> Tuple[str, float]:
        """Detect what type of question is being asked."""
        question_lower = question.lower()
        
        best_match = None
        best_score = 0
        
        for q_type, config in self.question_patterns.items():
            score = 0
            for pattern in config["patterns"]:
                if re.search(pattern, question_lower):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = q_type
        
        confidence = min(1.0, best_score / 2)  # Normalize
        return best_match or "general", confidence
    
    def _get_disease_section(self, disease_key: str, section: str) -> str:
        """Get a specific section from disease info."""
        diseases = self.knowledge_base.get("diseases", {})
        disease_info = diseases.get(disease_key, {})
        
        content = disease_info.get(section, "")
        
        if isinstance(content, list):
            return "\n• " + "\n• ".join(content[:5])
        return str(content)
    
    def _compare_diseases(self, disease1: str, disease2: str) -> str:
        """Compare two diseases for differential diagnosis."""
        diseases = self.knowledge_base.get("diseases", {})
        
        d1_info = diseases.get(disease1, {})
        d2_info = diseases.get(disease2, {})
        
        if not d1_info or not d2_info:
            return "Unable to compare - disease not found."
        
        d1_name = d1_info.get("display_name", disease1)
        d2_name = d2_info.get("display_name", disease2)
        
        d1_indicators = d1_info.get("visual_indicators", [])
        d2_indicators = d2_info.get("visual_indicators", [])
        
        result = f"**{d1_name} vs {d2_name}**\n\n"
        result += f"**{d1_name}** key signs:\n• " + "\n• ".join(d1_indicators[:4]) + "\n\n"
        result += f"**{d2_name}** key signs:\n• " + "\n• ".join(d2_indicators[:4]) + "\n\n"
        
        # Find distinguishing features
        d1_set = set(d1_indicators)
        d2_set = set(d2_indicators)
        
        unique_d1 = d1_set - d2_set
        unique_d2 = d2_set - d1_set
        
        if unique_d1:
            result += f"**Only in {d1_name}:** " + ", ".join(list(unique_d1)[:3]) + "\n"
        if unique_d2:
            result += f"**Only in {d2_name}:** " + ", ".join(list(unique_d2)[:3])
        
        return result
    
    def answer_question(self, current_disease: str, question: str) -> Dict:
        """
        Answer a follow-up question about a disease.
        
        Args:
            current_disease: The diagnosed disease (context)
            question: User's follow-up question
            
        Returns:
            Dict with answer, confidence, and source section
        """
        disease_key = self._normalize_label(current_disease)
        question_type, confidence = self._detect_question_type(question)
        
        # Handle comparison questions
        if question_type == "difference":
            # Try to extract the second disease from the question
            question_lower = question.lower()
            other_disease = None
            
            for label in self._label_map.keys():
                if label in question_lower and label != disease_key:
                    other_disease = self._normalize_label(label)
                    break
            
            # Also check full disease names
            for dkey in self.knowledge_base.get("diseases", {}).keys():
                if dkey.replace("_", " ") in question_lower and dkey != disease_key:
                    other_disease = dkey
                    break
            
            if other_disease:
                answer = self._compare_diseases(disease_key, other_disease)
                return {
                    "answer": answer,
                    "question_type": "comparison",
                    "confidence": 0.9,
                    "diseases_compared": [disease_key, other_disease]
                }
        
        # Get the relevant section
        config = self.question_patterns.get(question_type, {})
        section = config.get("section", "")
        prefix = config.get("prefix", "Information")
        
        if not section:
            # General fallback - provide overview
            diseases = self.knowledge_base.get("diseases", {})
            disease_info = diseases.get(disease_key, {})
            display_name = disease_info.get("display_name", disease_key)
            severity = disease_info.get("severity", "unknown")
            
            return {
                "answer": f"**{display_name}** (Severity: {severity})\n\nPlease ask about: causes, treatments, recovery time, prevention, or symptoms.",
                "question_type": "general",
                "confidence": 0.3,
                "available_topics": list(self.question_patterns.keys())
            }
        
        content = self._get_disease_section(disease_key, section)
        
        if not content or content == "unknown":
            return {
                "answer": f"I don't have specific information about {section} for this condition.",
                "question_type": question_type,
                "confidence": 0.0,
                "section": section
            }
        
        # Get display name for context
        diseases = self.knowledge_base.get("diseases", {})
        display_name = diseases.get(disease_key, {}).get("display_name", disease_key)
        
        answer = f"**{prefix} for {display_name}:**\n{content}"
        
        return {
            "answer": answer,
            "question_type": question_type,
            "confidence": confidence,
            "section": section,
            "disease": disease_key
        }
    
    def get_quick_facts(self, disease_key: str) -> Dict:
        """Get quick facts for a disease (for UI cards)."""
        disease_key = self._normalize_label(disease_key)
        diseases = self.knowledge_base.get("diseases", {})
        info = diseases.get(disease_key, {})
        
        if not info:
            return {}
        
        return {
            "name": info.get("display_name", disease_key),
            "severity": info.get("severity", "unknown"),
            "recovery": info.get("recovery_timeline", "Unknown"),
            "top_symptoms": info.get("visual_indicators", [])[:3],
            "first_treatment": info.get("treatments", ["Consult an expert"])[0]
        }
    
    def suggest_questions(self, disease_key: str) -> List[str]:
        """Suggest follow-up questions user might want to ask."""
        disease_key = self._normalize_label(disease_key)
        diseases = self.knowledge_base.get("diseases", {})
        info = diseases.get(disease_key, {})
        
        if not info:
            return []
        
        display_name = info.get("display_name", disease_key)
        
        suggestions = [
            f"How long until my plant recovers from {display_name}?",
            f"What caused {display_name}?",
            f"How can I prevent {display_name} in the future?",
        ]
        
        # Add comparison suggestion if applicable
        if disease_key in ["gray_mold", "white_mold"]:
            other = "white_mold" if disease_key == "gray_mold" else "gray_mold"
            other_name = diseases.get(other, {}).get("display_name", other)
            suggestions.append(f"What's the difference between {display_name} and {other_name}?")
        
        if disease_key in ["drought", "overwatering"]:
            other = "overwatering" if disease_key == "drought" else "drought"
            other_name = diseases.get(other, {}).get("display_name", other)
            suggestions.append(f"How do I tell if it's {display_name} vs {other_name}?")
        
        return suggestions


# Test
if __name__ == "__main__":
    qa = DiseaseQA()
    
    print("=== Question-Answering RAG Test ===\n")
    
    # Test different question types
    test_cases = [
        ("gray_mold", "How long until my plant recovers?"),
        ("gray_mold", "What caused this?"),
        ("drought", "How can I prevent this in the future?"),
        ("gray_mold", "What's the difference between gray mold and white mold?"),
        ("root_rot", "Is this serious? Will my plant die?"),
        ("frost_injury", "What should I do to fix this?"),
    ]
    
    for disease, question in test_cases:
        print(f"Disease: {disease}")
        print(f"Question: {question}")
        result = qa.answer_question(disease, question)
        print(f"Type: {result['question_type']} (confidence: {result['confidence']:.1f})")
        print(f"Answer: {result['answer'][:200]}...")
        print("-" * 50)
    
    print("\n=== Suggested Questions ===")
    for disease in ["gray_mold", "drought"]:
        print(f"\n{disease}:")
        for q in qa.suggest_questions(disease):
            print(f"  • {q}")

