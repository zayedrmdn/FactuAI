# services/classifier/client.py

import json
import logging
from typing import Any, Dict, List, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.logging import logger
from core.exceptions import LLMClientError
from pipeline.config import CLASSIFIER_PATH, DEVICE
from .cleaner import clean_evidence_text
from .justifier import justify
from .constants import LABELS


class ClaimClassifier:
    def __init__(self):
        self.available = False
        try:
            logger.debug(f"[CLASSIFIER] Loading model from {CLASSIFIER_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    CLASSIFIER_PATH, num_labels=len(LABELS)
                )
                .to(DEVICE)
                .eval()
            )
            self.available = True
            logger.debug("[CLASSIFIER] Model ready")
        except Exception as e:
            logger.error(f"[CLASSIFIER] Failed to load model: {e}")
            self.tokenizer = None
            self.model = None

    def is_available(self) -> bool:
        return self.available

    def predict(
        self,
        claim: str,
        evidence: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.available:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": {label: 0.0 for label in LABELS}
            }

        try:
            # extract evidence text
            if isinstance(evidence, dict):
                evidence_text = evidence.get("answer", "")
            else:
                evidence_text = str(evidence)

            # clean it
            cleaned = clean_evidence_text(evidence_text)

            # prepare input
            input_text = f"[STAT] {claim.strip()} [JUST] {cleaned.strip()}"
            logger.debug(f"[CLASSIFIER] Tokenizing input: {input_text}")

            tokens = self.tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**tokens)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

            idx = int(torch.argmax(torch.tensor(probs)))
            label = LABELS[idx]
            confidence = probs[idx]
            logger.debug(f"[CLASSIFIER] Predicted: {label} ({confidence:.3f})")

            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {LABELS[i]: p for i, p in enumerate(probs)}
            }

        except Exception as e:
            logger.error(f"[CLASSIFIER] Prediction error: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": {label: 0.0 for label in LABELS}
            }

    def classify_with_evidence(
        self,
        claim: str,
        evidence: Union[str, Dict[str, Any]],
        return_conf: bool = False
    ) -> Union[str, Tuple[str, float]]:
        result = self.predict(claim, evidence)
        if return_conf:
            return result["label"], result["confidence"]
        return result["label"]

    def justify(
        self,
        label: str,
        claim: str,
        evidence: str,
        llm
    ) -> str:
        if not self.available:
            raise LLMClientError("Classifier not available")
        return justify(label, claim, evidence, llm)