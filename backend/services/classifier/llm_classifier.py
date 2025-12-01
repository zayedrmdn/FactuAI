"""
LLM-based Classifier

Uses LLM for claim classification in cloud mode, replacing the need
for local DistilBERT model. This enables lightweight deployment without
PyTorch/CUDA dependencies.
"""

import json
import re
from typing import Dict, Any, List, Tuple, Union, Optional

from services.llm.base import BaseLLM
from services.classifier.constants import LABELS
from core.logging import logger
from core.exceptions import ClassifierError


class LLMClassifier:
    """
    LLM-based claim classifier.
    
    Uses prompting to classify claims instead of a fine-tuned BERT model.
    This is the "cloud mode" classifier that doesn't require GPU.
    """
    
    # Classification prompt template
    CLASSIFICATION_PROMPT = """You are a fact-checking expert. Analyze the following claim based on the provided evidence and classify it into one of these categories:

Categories (from least to most true):
- false: The claim is completely false
- mostly_false: The claim is mostly false with minor accurate elements  
- barely_true: The claim has a small element of truth but is mostly misleading
- half_true: The claim is partially accurate but leaves out important context
- mostly_true: The claim is accurate but needs minor clarification
- true: The claim is completely accurate

Claim: "{claim}"

Evidence: "{evidence}"

Analyze the claim against the evidence and respond with ONLY a JSON object in this exact format:
{{"label": "<one of the categories above>", "confidence": <0.0 to 1.0>, "reasoning": "<brief explanation>"}}"""

    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize LLM classifier.
        
        Args:
            llm: LLM provider to use. If None, creates one using LLMFactory.
        """
        self._llm = llm
        self._available = False
        
        if llm is None:
            try:
                from services.llm.factory import LLMFactory
                self._llm = LLMFactory.create()
            except Exception as e:
                logger.error(f"[LLM_CLASSIFIER] Failed to create LLM: {e}")
                return
        
        self._available = self._llm is not None and self._llm.is_available()
        if self._available:
            logger.info("[LLM_CLASSIFIER] Initialized successfully")
        else:
            logger.warning("[LLM_CLASSIFIER] LLM not available")
    
    def is_available(self) -> bool:
        """Check if classifier is available."""
        return self._available
    
    def predict(
        self,
        claim: str,
        evidence: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify a claim based on evidence.
        
        Args:
            claim: The claim to classify
            evidence: Evidence text or dict with 'answer' key
            
        Returns:
            Dictionary with label, confidence, and probabilities
        """
        if not self._available:
            return self._empty_result("Classifier not available")
        
        # Extract evidence text
        if isinstance(evidence, dict):
            evidence_text = evidence.get("answer", evidence.get("text", ""))
        else:
            evidence_text = str(evidence)
        
        if not evidence_text.strip():
            return self._empty_result("No evidence provided")
        
        try:
            # Build prompt
            prompt = self.CLASSIFICATION_PROMPT.format(
                claim=claim.strip(),
                evidence=evidence_text.strip()[:2000]  # Limit evidence length
            )
            
            # Get LLM response
            response = self._llm.generate_response(prompt, max_tokens=300)
            logger.debug(f"[LLM_CLASSIFIER] Raw response: {response}")
            
            # Parse response
            result = self._parse_response(response)
            logger.debug(f"[LLM_CLASSIFIER] Parsed result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"[LLM_CLASSIFIER] Classification failed: {e}")
            return self._empty_result(str(e))
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into classification result."""
        # Try to extract JSON from response
        try:
            # Look for JSON object in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                label = data.get("label", "").lower().replace("-", "_").replace(" ", "_")
                
                # Validate label
                if label not in LABELS:
                    # Try to find closest match
                    label = self._fuzzy_match_label(label)
                
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                
                # Build probabilities (approximate based on confidence)
                probs = self._build_probabilities(label, confidence)
                
                return {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": probs,
                    "reasoning": data.get("reasoning", "")
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[LLM_CLASSIFIER] JSON parse failed: {e}")
        
        # Fallback: try to extract label from text
        label = self._extract_label_from_text(response)
        return {
            "label": label,
            "confidence": 0.5,
            "probabilities": self._build_probabilities(label, 0.5),
            "reasoning": response[:200]
        }
    
    def _fuzzy_match_label(self, label: str) -> str:
        """Find the closest matching label."""
        label_lower = label.lower()
        
        # Direct substring matching
        for valid_label in LABELS:
            if valid_label in label_lower or label_lower in valid_label:
                return valid_label
        
        # Keyword matching
        if "false" in label_lower:
            if "mostly" in label_lower:
                return "mostly_false"
            return "false"
        if "true" in label_lower:
            if "mostly" in label_lower:
                return "mostly_true"
            if "barely" in label_lower:
                return "barely_true"
            if "half" in label_lower:
                return "half_true"
            return "true"
        
        return "half_true"  # Default to middle ground
    
    def _extract_label_from_text(self, text: str) -> str:
        """Extract label from unstructured text response."""
        text_lower = text.lower()
        
        for label in LABELS:
            if label.replace("_", " ") in text_lower or label in text_lower:
                return label
        
        return "half_true"  # Default
    
    def _build_probabilities(self, label: str, confidence: float) -> Dict[str, float]:
        """Build probability distribution from label and confidence."""
        probs = {l: 0.0 for l in LABELS}
        
        if label in probs:
            probs[label] = confidence
            # Distribute remaining probability to adjacent labels
            remaining = 1.0 - confidence
            idx = LABELS.index(label)
            
            # Adjacent labels get some probability
            if idx > 0:
                probs[LABELS[idx - 1]] = remaining * 0.3
            if idx < len(LABELS) - 1:
                probs[LABELS[idx + 1]] = remaining * 0.3
            
            # Remaining distributed evenly
            assigned = sum(probs.values())
            if assigned < 1.0:
                unassigned = [l for l in LABELS if probs[l] == 0.0]
                if unassigned:
                    per_label = (1.0 - assigned) / len(unassigned)
                    for l in unassigned:
                        probs[l] = per_label
        
        return probs
    
    def _empty_result(self, reason: str = "") -> Dict[str, Any]:
        """Return empty/unknown result."""
        return {
            "label": "unknown",
            "confidence": 0.0,
            "probabilities": {label: 0.0 for label in LABELS},
            "reasoning": reason
        }
    
    def classify_with_evidence(
        self,
        claim: str,
        evidence: Union[str, Dict[str, Any]],
        return_conf: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Classify and optionally return confidence.
        
        Args:
            claim: Claim to classify
            evidence: Evidence text or dict
            return_conf: Whether to return confidence score
            
        Returns:
            Label string, or tuple of (label, confidence) if return_conf=True
        """
        result = self.predict(claim, evidence)
        if return_conf:
            return result["label"], result["confidence"]
        return result["label"]
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple texts (for interface compatibility).
        Note: For LLM classifier, this processes sequentially.
        """
        return [self.predict(text, "") for text in texts]
    
    def get_supported_labels(self) -> List[str]:
        """Get supported classification labels."""
        return LABELS.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get classifier model info."""
        llm_info = self._llm.get_model_info() if self._llm else {}
        return {
            "type": "llm_classifier",
            "available": self._available,
            "labels": LABELS,
            "llm_provider": llm_info.get("provider", "unknown"),
            "llm_model": llm_info.get("model_name", "unknown"),
        }
