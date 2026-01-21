"""
RAG Retriever for Plant Diagnostic System

This module provides retrieval-augmented generation capabilities
by querying a disease-specific knowledge base.

Usage:
    from knowledge_graph.rag_retriever import DiseaseRAG
    
    rag = DiseaseRAG()
    context = rag.get_context("drought")
    # Returns structured knowledge about drought symptoms, treatments, etc.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DiseaseRAG:
    """RAG retriever for strawberry disease knowledge."""
    
    def __init__(self, kb_path: Optional[str] = None):
        """
        Initialize the RAG retriever.
        
        Args:
            kb_path: Path to knowledge base JSON. Defaults to disease_knowledge_base.json
        """
        if kb_path is None:
            kb_path = Path(__file__).parent / "disease_knowledge_base.json"
        
        self.kb_path = Path(kb_path)
        self.knowledge_base = self._load_knowledge_base()
        
        # Build label normalization map
        self._label_map = {
            "overwatered": "overwatering",
            "frost": "frost_injury",
            "graymold": "gray_mold",
            "gray mold": "gray_mold",
            "whitemold": "white_mold",
            "white mold": "white_mold",
            "rootrot": "root_rot",
            "root rot": "root_rot",
            "frost injury": "frost_injury",
            "drought stress": "drought",
        }
        
    def _load_knowledge_base(self) -> Dict:
        """Load the knowledge base from JSON."""
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[RAG] Warning: Knowledge base not found at {self.kb_path}")
            return {"diseases": {}, "general_best_practices": {}}
        except json.JSONDecodeError as e:
            print(f"[RAG] Error parsing knowledge base: {e}")
            return {"diseases": {}, "general_best_practices": {}}
    
    def _normalize_label(self, label: str) -> str:
        """Normalize disease label to match knowledge base keys."""
        normalized = label.lower().strip().replace("-", "_")
        return self._label_map.get(normalized, normalized)
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all disease keys in knowledge base."""
        return list(self.knowledge_base.get("diseases", {}).keys())
    
    def get_disease_info(self, disease_label: str) -> Dict:
        """Get complete disease information dictionary."""
        label = self._normalize_label(disease_label)
        return self.knowledge_base.get("diseases", {}).get(label, {})
    
    def get_context(self, disease_label: str, include_sections: Optional[List[str]] = None) -> str:
        """
        Retrieve context for a specific disease diagnosis.
        
        Args:
            disease_label: The disease label (e.g., "drought", "gray_mold")
            include_sections: Optional list of sections to include. 
                             Options: ["symptoms", "causes", "treatments", "recovery_timeline", "prevention"]
                             Defaults to symptoms and treatments for conciseness.
        
        Returns:
            Formatted context string for prompt injection
        """
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        
        if label not in diseases:
            return ""
        
        disease_info = diseases[label]
        
        if include_sections is None:
            include_sections = ["symptoms", "treatments"]
        
        context_parts = []
        
        if "symptoms" in include_sections and "symptoms" in disease_info:
            # Get first 5 most distinctive symptoms
            symptoms = "; ".join(disease_info["symptoms"][:5])
            context_parts.append(f"Key symptoms: {symptoms}")
        
        if "causes" in include_sections and "causes" in disease_info:
            causes = "; ".join(disease_info["causes"][:3])
            context_parts.append(f"Common causes: {causes}")
        
        if "treatments" in include_sections and "treatments" in disease_info:
            treatments = "; ".join(disease_info["treatments"][:4])
            context_parts.append(f"Recommended treatments: {treatments}")
        
        if "recovery_timeline" in include_sections and "recovery_timeline" in disease_info:
            context_parts.append(f"Recovery: {disease_info['recovery_timeline']}")
        
        if "prevention" in include_sections and "prevention" in disease_info:
            context_parts.append(f"Prevention: {disease_info['prevention']}")
        
        return " | ".join(context_parts)
    
    def get_detailed_context(self, disease_label: str) -> str:
        """Get comprehensive context including all sections."""
        return self.get_context(
            disease_label, 
            include_sections=["symptoms", "causes", "treatments", "recovery_timeline", "prevention"]
        )
    
    def get_symptoms(self, disease_label: str) -> List[str]:
        """Get list of symptoms for a disease."""
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        return diseases.get(label, {}).get("symptoms", [])
    
    def get_treatments(self, disease_label: str) -> List[str]:
        """Get list of treatments for a disease."""
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        return diseases.get(label, {}).get("treatments", [])
    
    def get_causes(self, disease_label: str) -> List[str]:
        """Get list of causes for a disease."""
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        return diseases.get(label, {}).get("causes", [])
    
    def get_visual_indicators(self, disease_label: str) -> List[str]:
        """Get visual indicators for quick symptom matching."""
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        return diseases.get(label, {}).get("visual_indicators", [])
    
    def get_display_name(self, disease_label: str) -> str:
        """Get human-readable display name for a disease."""
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        info = diseases.get(label, {})
        return info.get("display_name", label.replace("_", " ").title())
    
    def get_severity(self, disease_label: str) -> str:
        """Get severity level (none, moderate, severe)."""
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        return diseases.get(label, {}).get("severity", "unknown")
    
    def get_best_practices(self) -> Dict:
        """Get general best practices for plant care."""
        return self.knowledge_base.get("general_best_practices", {})
    
    def get_graph_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get data structured for knowledge graph visualization.
        
        Returns:
            Tuple of (nodes_list, edges_list) for graph construction
        """
        nodes = []
        edges = []
        node_id = 0
        
        diseases = self.knowledge_base.get("diseases", {})
        
        for disease_key, disease_info in diseases.items():
            # Disease node
            disease_node_id = f"disease_{node_id}"
            display_name = disease_info.get("display_name", disease_key.replace("_", " ").title())
            severity = disease_info.get("severity", "unknown")
            
            nodes.append({
                "id": disease_node_id,
                "label": "Disease",
                "name": display_name,
                "severity": severity,
                "key": disease_key
            })
            node_id += 1
            
            # Symptom nodes (limit to top 5 for readability)
            for i, symptom in enumerate(disease_info.get("symptoms", [])[:5]):
                symptom_node_id = f"symptom_{node_id}"
                # Shorten symptom text for display
                short_symptom = symptom[:50] + "..." if len(symptom) > 50 else symptom
                nodes.append({
                    "id": symptom_node_id,
                    "label": "Symptom",
                    "name": short_symptom,
                    "full_text": symptom
                })
                edges.append({
                    "start_id": disease_node_id,
                    "end_id": symptom_node_id,
                    "type": "Shows_Symptom"
                })
                node_id += 1
            
            # Treatment nodes (limit to top 3)
            for i, treatment in enumerate(disease_info.get("treatments", [])[:3]):
                treatment_node_id = f"treatment_{node_id}"
                short_treatment = treatment[:50] + "..." if len(treatment) > 50 else treatment
                nodes.append({
                    "id": treatment_node_id,
                    "label": "Treatment",
                    "name": short_treatment,
                    "full_text": treatment
                })
                edges.append({
                    "start_id": disease_node_id,
                    "end_id": treatment_node_id,
                    "type": "Treated_By"
                })
                node_id += 1
            
            # Cause nodes (limit to top 3)
            for i, cause in enumerate(disease_info.get("causes", [])[:3]):
                cause_node_id = f"cause_{node_id}"
                short_cause = cause[:50] + "..." if len(cause) > 50 else cause
                nodes.append({
                    "id": cause_node_id,
                    "label": "Cause",
                    "name": short_cause,
                    "full_text": cause
                })
                edges.append({
                    "start_id": cause_node_id,
                    "end_id": disease_node_id,
                    "type": "Causes"
                })
                node_id += 1
        
        return nodes, edges
    
    def get_disease_graph_data(self, disease_label: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Get graph data for a single disease (for focused view).
        
        Args:
            disease_label: The disease to get graph data for
            
        Returns:
            Tuple of (nodes_list, edges_list) for graph construction
        """
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        
        if label not in diseases:
            return [], []
        
        disease_info = diseases[label]
        nodes = []
        edges = []
        node_id = 0
        
        # Disease node (central)
        disease_node_id = f"disease_{node_id}"
        display_name = disease_info.get("display_name", label.replace("_", " ").title())
        severity = disease_info.get("severity", "unknown")
        
        nodes.append({
            "id": disease_node_id,
            "label": "Disease",
            "name": display_name,
            "severity": severity,
            "key": label
        })
        node_id += 1
        
        # All symptoms for focused view
        for symptom in disease_info.get("symptoms", []):
            symptom_node_id = f"symptom_{node_id}"
            short_symptom = symptom[:60] + "..." if len(symptom) > 60 else symptom
            nodes.append({
                "id": symptom_node_id,
                "label": "Symptom",
                "name": short_symptom,
                "full_text": symptom
            })
            edges.append({
                "start_id": disease_node_id,
                "end_id": symptom_node_id,
                "type": "Shows_Symptom"
            })
            node_id += 1
        
        # All treatments
        for treatment in disease_info.get("treatments", []):
            treatment_node_id = f"treatment_{node_id}"
            short_treatment = treatment[:60] + "..." if len(treatment) > 60 else treatment
            nodes.append({
                "id": treatment_node_id,
                "label": "Treatment",
                "name": short_treatment,
                "full_text": treatment
            })
            edges.append({
                "start_id": disease_node_id,
                "end_id": treatment_node_id,
                "type": "Treated_By"
            })
            node_id += 1
        
        # All causes
        for cause in disease_info.get("causes", []):
            cause_node_id = f"cause_{node_id}"
            short_cause = cause[:60] + "..." if len(cause) > 60 else cause
            nodes.append({
                "id": cause_node_id,
                "label": "Cause",
                "name": short_cause,
                "full_text": cause
            })
            edges.append({
                "start_id": cause_node_id,
                "end_id": disease_node_id,
                "type": "Causes"
            })
            node_id += 1
        
        # Recovery info as node
        recovery = disease_info.get("recovery_timeline", "")
        if recovery:
            recovery_node_id = f"recovery_{node_id}"
            short_recovery = recovery[:60] + "..." if len(recovery) > 60 else recovery
            nodes.append({
                "id": recovery_node_id,
                "label": "Recovery",
                "name": short_recovery,
                "full_text": recovery
            })
            edges.append({
                "start_id": disease_node_id,
                "end_id": recovery_node_id,
                "type": "Recovery_Time"
            })
            node_id += 1
        
        return nodes, edges
    
    def check_hallucination(self, disease_label: str, generated_text: str) -> Dict:
        """
        Check if generated text mentions symptoms consistent with the disease.
        
        Args:
            disease_label: The diagnosed disease
            generated_text: The model's generated explanation
        
        Returns:
            Dict with hallucination analysis:
            - symptoms_mentioned: List of known symptoms found in text
            - unknown_claims: Potential hallucinations (symptoms from other diseases)
            - grounding_score: Ratio of known symptoms mentioned
        """
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        
        if label not in diseases:
            return {"error": f"Unknown disease: {label}"}
        
        target_symptoms = diseases[label].get("symptoms", [])
        visual_indicators = diseases[label].get("visual_indicators", [])
        text_lower = generated_text.lower()
        
        # Check which known symptoms/indicators are mentioned
        symptoms_mentioned = []
        
        # Check visual indicators first (shorter, more reliable)
        for indicator in visual_indicators:
            if indicator.lower() in text_lower:
                symptoms_mentioned.append(indicator)
        
        # Check full symptoms
        for symptom in target_symptoms:
            # Extract key phrases (3-4 significant words)
            words = [w for w in symptom.lower().split() if len(w) > 4][:4]
            if any(word in text_lower for word in words):
                if symptom not in symptoms_mentioned:
                    symptoms_mentioned.append(symptom[:60])
        
        # Check for symptoms from OTHER diseases (potential hallucinations)
        other_disease_symptoms = []
        for other_label, other_info in diseases.items():
            if other_label != label:
                for indicator in other_info.get("visual_indicators", []):
                    if indicator.lower() in text_lower:
                        other_disease_symptoms.append(f"{other_label}: {indicator}")
        
        total_indicators = len(visual_indicators) + len(target_symptoms)
        grounding_score = len(symptoms_mentioned) / max(1, min(10, total_indicators))
        grounding_score = min(1.0, grounding_score)  # Cap at 1.0
        
        return {
            "symptoms_mentioned": symptoms_mentioned,
            "symptoms_expected": target_symptoms[:5],
            "visual_indicators": visual_indicators,
            "potential_hallucinations": other_disease_symptoms[:5],
            "grounding_score": round(grounding_score, 2),
            "mention_count": len(symptoms_mentioned),
            "disease": label
        }
    
    def check_definite_hallucination(self, disease_label: str, generated_text: str) -> Dict:
        """
        Strict hallucination check - only flags DEFINITE wrong-disease indicators.
        
        This is more careful than check_hallucination():
        - Only checks visual_indicators (short, unique phrases)
        - Ignores vague/shared symptoms like "wilting" or "yellowing"
        - Returns whether regeneration is needed
        
        Args:
            disease_label: The diagnosed disease
            generated_text: The model's generated explanation
            
        Returns:
            Dict with:
            - has_definite_hallucination: bool - True if regeneration needed
            - hallucinations: List of definite wrong-disease indicators found
            - disease_confused_with: Which disease the hallucination belongs to
        """
        label = self._normalize_label(disease_label)
        diseases = self.knowledge_base.get("diseases", {})
        
        if label not in diseases:
            return {"has_definite_hallucination": False, "error": f"Unknown disease: {label}"}
        
        text_lower = generated_text.lower()
        
        # Only check UNIQUE visual indicators from OTHER diseases
        # These are definite hallucinations - no ambiguity
        definite_hallucinations = []
        confused_with = set()
        
        # Highly specific indicators that are UNIQUE to each disease
        # If these appear for wrong disease, it's definitely a hallucination
        unique_indicators = {
            "drought": ["cracked soil", "limp petioles"],
            "overwatering": ["mushy crown", "algae growth", "waterlogged"],
            "root_rot": ["foul odor", "red stele", "brown roots"],
            "frost_injury": ["black eye", "translucent leaves", "glassy tissue"],
            "gray_mold": ["gray fuzz", "dusty spores", "tan mummies", "botrytis"],
            "white_mold": ["white cotton", "black sclerotia", "cottony mycelium"],
            "healthy": []  # Don't flag healthy indicators as hallucinations
        }
        
        for other_disease, indicators in unique_indicators.items():
            if other_disease != label and other_disease != "healthy":
                for indicator in indicators:
                    if indicator.lower() in text_lower:
                        definite_hallucinations.append(f"{indicator} (indicates {other_disease})")
                        confused_with.add(other_disease)
        
        return {
            "has_definite_hallucination": len(definite_hallucinations) > 0,
            "hallucinations": definite_hallucinations,
            "disease_confused_with": list(confused_with),
            "diagnosed_as": label
        }


# Quick test
if __name__ == "__main__":
    rag = DiseaseRAG()
    
    print("=== Disease RAG Retriever Test ===\n")
    
    print("Available diseases:", rag.get_all_diseases())
    
    for disease in ["drought", "gray_mold", "white_mold", "healthy"]:
        print(f"\n--- {rag.get_display_name(disease)} [{rag.get_severity(disease)}] ---")
        context = rag.get_context(disease)
        print(f"Context length: {len(context)} chars")
        print(f"Preview: {context[:200]}...")
        print(f"Visual indicators: {rag.get_visual_indicators(disease)}")
    
    print("\n\n=== Graph Data Test ===")
    nodes, edges = rag.get_graph_data()
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    
    print("\n\n=== Single Disease Graph ===")
    nodes, edges = rag.get_disease_graph_data("gray_mold")
    print(f"Gray mold nodes: {len(nodes)}")
    print(f"Gray mold edges: {len(edges)}")
    
    print("\n\n=== Hallucination Check ===")
    test_text = "The plant shows dry, curled margins and the soil appears cracked and pale. There is also some gray fuzz."
    result = rag.check_hallucination("drought", test_text)
    print(f"Grounding score: {result['grounding_score']}")
    print(f"Symptoms mentioned: {result['symptoms_mentioned']}")
    print(f"Potential hallucinations: {result['potential_hallucinations']}")
