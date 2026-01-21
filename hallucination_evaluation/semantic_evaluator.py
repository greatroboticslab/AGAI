#!/usr/bin/env python3
"""
Semantic Evaluator - Advanced hallucination detection using:
1. Sentence embeddings for semantic similarity
2. LLM-as-judge for nuanced evaluation
3. Training data alignment checking

Requires: sentence-transformers, (optional) openai or local LLM
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hallucination_evaluation.config import (
    TRAINING_DATA_PATH, KNOWLEDGE_BASE_PATH, DISEASE_CLASSES
)


@dataclass
class SemanticEvalResult:
    """Results from semantic evaluation."""
    image_path: str
    ground_truth_label: str
    predicted_label: str
    generated_text: str
    
    # Semantic similarity to training data
    best_training_match_score: float = 0.0
    best_training_match_text: str = ""
    training_alignment_score: float = 0.0
    
    # Semantic grounding (similarity to expected symptoms)
    symptom_grounding_score: float = 0.0
    matched_symptoms: List[str] = field(default_factory=list)
    
    # LLM judge scores (if available)
    llm_factuality_score: float = -1.0  # -1 = not evaluated
    llm_hallucination_flags: List[str] = field(default_factory=list)
    
    # Overall
    semantic_quality_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "ground_truth_label": self.ground_truth_label,
            "predicted_label": self.predicted_label,
            "best_training_match_score": self.best_training_match_score,
            "training_alignment_score": self.training_alignment_score,
            "symptom_grounding_score": self.symptom_grounding_score,
            "matched_symptoms": self.matched_symptoms,
            "llm_factuality_score": self.llm_factuality_score,
            "llm_hallucination_flags": self.llm_hallucination_flags,
            "semantic_quality_score": self.semantic_quality_score
        }


class SemanticEvaluator:
    """
    Advanced semantic evaluation using embeddings and LLM judging.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic evaluator.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.embedder = None
        if EMBEDDINGS_AVAILABLE:
            print(f"Loading sentence transformer: {model_name}")
            self.embedder = SentenceTransformer(model_name)
        
        # Load training data and knowledge base
        self.training_data = self._load_training_data()
        self.knowledge_base = self._load_knowledge_base()
        
        # Pre-compute embeddings for training responses
        self.training_embeddings = {}
        self._precompute_training_embeddings()
        
        # Pre-compute embeddings for expected symptoms
        self.symptom_embeddings = {}
        self._precompute_symptom_embeddings()
    
    def _load_training_data(self) -> Dict[str, List[str]]:
        """Load training responses grouped by disease class."""
        training_by_class = {cls: [] for cls in DISEASE_CLASSES}
        
        try:
            with open(TRAINING_DATA_PATH, 'r') as f:
                data = json.load(f)
            
            # Label mappings for training data paths
            label_map = {
                "frost": "frost_injury",
                "gray": "gray_mold",
                "grey": "gray_mold", 
                "white": "white_mold",
                "root": "root_rot",
                "over": "overwatering",
                "water": "overwatering",
            }
            
            for entry in data:
                # Extract class from path
                img_path = entry.get("image", "").lower()
                matched_cls = None
                
                # Try direct match first
                for cls in DISEASE_CLASSES:
                    if cls in img_path or cls.replace("_", "") in img_path:
                        matched_cls = cls
                        break
                
                # Try mapped labels
                if not matched_cls:
                    for key, mapped in label_map.items():
                        if key in img_path:
                            matched_cls = mapped
                            break
                
                # Default fallback
                if not matched_cls and "health" in img_path:
                    matched_cls = "healthy"
                
                if matched_cls:
                    for conv in entry.get("conversations", []):
                        if conv.get("from") == "assistant":
                            training_by_class[matched_cls].append(conv.get("value", ""))
                            break
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
        
        return training_by_class
    
    def _load_knowledge_base(self) -> Dict:
        """Load the disease knowledge base."""
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
            return {"diseases": {}}
    
    def _precompute_training_embeddings(self):
        """Pre-compute embeddings for all training responses."""
        if not self.embedder:
            return
        
        print("Pre-computing training embeddings...")
        for cls, responses in self.training_data.items():
            if responses:
                # Sample up to 50 responses per class to keep memory manageable
                sampled = responses[:50]
                self.training_embeddings[cls] = self.embedder.encode(
                    sampled, convert_to_tensor=True, show_progress_bar=False
                )
        print(f"  Computed embeddings for {len(self.training_embeddings)} classes")
    
    def _precompute_symptom_embeddings(self):
        """
        Pre-compute embeddings for expected symptoms per disease.
        
        IMPORTANT: Uses TRAINING DATA as ground truth, not the knowledge base.
        Extracts symptom sentences from training responses.
        """
        if not self.embedder:
            return
        
        print("Pre-computing symptom embeddings FROM TRAINING DATA...")
        
        # Extract symptom sentences from training data responses
        import re
        
        for disease, responses in self.training_data.items():
            if not responses:
                continue
            
            # Extract symptom-like sentences from training responses
            symptom_sentences = []
            for response in responses[:30]:  # Sample 30 responses
                # Split into sentences
                sentences = re.split(r'[.;]', response)
                for sent in sentences:
                    sent = sent.strip()
                    # Keep sentences that describe visual symptoms (not recommendations)
                    if len(sent) > 20 and len(sent) < 200:
                        # Skip recommendation sentences
                        if any(kw in sent.lower() for kw in ['apply', 'irrigate', 'water', 'add', 'use', 'install', 'schedule']):
                            continue
                        # Keep sentences describing plant state
                        if any(kw in sent.lower() for kw in ['leaf', 'leaves', 'petiole', 'crown', 'soil', 'berry', 'fruit', 'root', 'appear', 'show', 'visible']):
                            symptom_sentences.append(sent)
            
            # Deduplicate similar sentences (keep unique ones)
            if symptom_sentences:
                unique_symptoms = list(set(symptom_sentences))[:50]  # Max 50 per class
                self.symptom_embeddings[disease] = {
                    "texts": unique_symptoms,
                    "embeddings": self.embedder.encode(
                        unique_symptoms, convert_to_tensor=True, show_progress_bar=False
                    )
                }
        
        print(f"  Computed symptom embeddings for {len(self.symptom_embeddings)} diseases from training data")
    
    def compute_training_alignment(
        self, 
        generated_text: str, 
        disease_class: str
    ) -> Tuple[float, str, float]:
        """
        Compute how well the generated text aligns with training data.
        
        Returns:
            - Best match score (cosine similarity)
            - Best matching training text (preview)
            - Overall alignment score
        """
        if not self.embedder or disease_class not in self.training_embeddings:
            return 0.0, "", 0.0
        
        # Encode generated text
        gen_embedding = self.embedder.encode(generated_text, convert_to_tensor=True)
        
        # Compute similarities to training responses
        train_embeddings = self.training_embeddings[disease_class]
        similarities = util.cos_sim(gen_embedding, train_embeddings)[0]
        
        # Get best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_text = self.training_data[disease_class][best_idx][:200] + "..."
        
        # Compute overall alignment (average of top-5 similarities)
        top_k = min(5, len(similarities))
        top_scores = similarities.topk(top_k).values
        alignment_score = top_scores.mean().item()
        
        return best_score, best_text, alignment_score
    
    def compute_symptom_grounding(
        self, 
        generated_text: str, 
        disease_class: str
    ) -> Tuple[float, List[str]]:
        """
        Compute semantic grounding score - how many expected symptoms
        are semantically present in the generated text.
        
        Returns:
            - Grounding score (0-1)
            - List of matched symptoms
        """
        if not self.embedder or disease_class not in self.symptom_embeddings:
            return 0.0, []
        
        symptom_data = self.symptom_embeddings[disease_class]
        symptoms = symptom_data["texts"]
        symptom_embs = symptom_data["embeddings"]
        
        # Split generated text into sentences for finer matching
        sentences = re.split(r'[.!?\n]', generated_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return 0.0, []
        
        # Encode sentences
        sentence_embs = self.embedder.encode(sentences, convert_to_tensor=True)
        
        # For each expected symptom, find best matching sentence
        matched_symptoms = []
        match_scores = []
        
        for i, symptom in enumerate(symptoms):
            similarities = util.cos_sim(symptom_embs[i], sentence_embs)[0]
            best_sim = similarities.max().item()
            
            # Consider matched if similarity > 0.5
            if best_sim > 0.5:
                matched_symptoms.append(f"{symptom} ({best_sim:.2f})")
                match_scores.append(best_sim)
        
        # Compute grounding score
        if symptoms:
            grounding_score = len(matched_symptoms) / len(symptoms)
        else:
            grounding_score = 0.0
        
        return grounding_score, matched_symptoms
    
    def evaluate_with_llm_judge(
        self,
        generated_text: str,
        disease_class: str,
        image_path: str
    ) -> Tuple[float, List[str]]:
        """
        Use an LLM to judge factuality and detect hallucinations.
        
        This is a placeholder - implement with your preferred LLM API.
        Options: OpenAI, local Llama, etc.
        
        Returns:
            - Factuality score (0-1)
            - List of hallucination flags
        """
        # TODO: Implement LLM-as-judge
        # Example prompt:
        # """
        # You are evaluating a plant diagnostic response for hallucinations.
        # 
        # Disease: {disease_class}
        # Generated Response: {generated_text}
        # 
        # Evaluate:
        # 1. Does it claim to see things impossible from a photo? (e.g., smells, underground roots)
        # 2. Does it mention symptoms of a different disease?
        # 3. Is the advice factually correct for this disease?
        # 
        # Return JSON: {"factuality_score": 0.0-1.0, "hallucinations": ["list", "of", "issues"]}
        # """
        
        return -1.0, []  # Not implemented
    
    def evaluate_response(
        self,
        image_path: str,
        predicted_label: str,
        generated_text: str,
        ground_truth_label: Optional[str] = None
    ) -> SemanticEvalResult:
        """
        Perform comprehensive semantic evaluation.
        """
        gt_label = ground_truth_label or predicted_label
        
        # Normalize labels
        gt_label = gt_label.lower().replace(" ", "_").replace("-", "_")
        pred_label = predicted_label.lower().replace(" ", "_").replace("-", "_")
        
        # Training alignment
        best_match, best_text, alignment = self.compute_training_alignment(
            generated_text, pred_label
        )
        
        # Symptom grounding
        grounding, matched = self.compute_symptom_grounding(
            generated_text, pred_label
        )
        
        # LLM judge (if implemented)
        llm_score, llm_flags = self.evaluate_with_llm_judge(
            generated_text, pred_label, image_path
        )
        
        # Compute overall semantic quality
        # Weight: 40% training alignment, 40% grounding, 20% LLM (if available)
        if llm_score >= 0:
            quality = 0.4 * alignment + 0.4 * grounding + 0.2 * llm_score
        else:
            quality = 0.5 * alignment + 0.5 * grounding
        
        return SemanticEvalResult(
            image_path=str(image_path),
            ground_truth_label=gt_label,
            predicted_label=pred_label,
            generated_text=generated_text,
            best_training_match_score=round(best_match, 3),
            best_training_match_text=best_text,
            training_alignment_score=round(alignment, 3),
            symptom_grounding_score=round(grounding, 3),
            matched_symptoms=matched,
            llm_factuality_score=llm_score,
            llm_hallucination_flags=llm_flags,
            semantic_quality_score=round(quality, 3)
        )


def test_semantic_evaluator():
    """Quick test of the semantic evaluator."""
    
    if not EMBEDDINGS_AVAILABLE:
        print("Cannot test - sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return
    
    print("=" * 60)
    print("SEMANTIC EVALUATOR TEST")
    print("=" * 60)
    
    evaluator = SemanticEvaluator()
    
    # Test response
    test_text = """
    Diagnosis: Drought
    
    The strawberry plant shows clear signs of drought stress:
    - Leaves are wilted and curling inward
    - Leaf margins are brown and crispy
    - The soil appears dry and cracked
    - Overall plant appears limp and stressed
    
    Recommendations:
    - Water deeply and consistently
    - Apply mulch to retain moisture
    - Consider drip irrigation
    """
    
    result = evaluator.evaluate_response(
        image_path="/test/drought.jpg",
        predicted_label="drought",
        generated_text=test_text
    )
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Training Alignment: {result.training_alignment_score:.1%}")
    print(f"  Best Match Score: {result.best_training_match_score:.1%}")
    print(f"  Symptom Grounding: {result.symptom_grounding_score:.1%}")
    print(f"  Matched Symptoms: {len(result.matched_symptoms)}")
    for s in result.matched_symptoms[:5]:
        print(f"    - {s}")
    print(f"  Overall Quality: {result.semantic_quality_score:.1%}")


if __name__ == "__main__":
    test_semantic_evaluator()

