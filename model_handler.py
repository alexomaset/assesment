#!/usr/bin/env python3
"""
Model Handler for REM Waste Accent Analyzer
Manages the Hugging Face accent classification model with optimizations
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import time
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AccentClassifier:
    """Handles accent classification with optimizations"""
    
    def __init__(self, model_name: str = 'm3hrdadfi/wav2vec2-large-xlsr-accent-classification', 
                 use_quantization: bool = True, 
                 cache_dir: Optional[str] = None):
        """
        Initialize the accent classifier.
        
        Args:
            model_name: HuggingFace model name
            use_quantization: Whether to apply model quantization
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.id2label = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the model with optional quantization.
        
        Returns:
            bool: Success status
        """
        try:
            start_time = time.time()
            logging.info(f"Loading model: {self.model_name}")
            
            # Load the processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            
            # Load the model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            
            # Apply quantization if enabled
            if self.use_quantization and torch.cuda.is_available():
                logging.info("Applying model quantization")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            # Get id2label mapping
            self.id2label = self.model.config.id2label
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            # Set to eval mode
            self.model.eval()
            
            self.is_loaded = True
            logging.info(f"Model loaded in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    def classify(self, audio_file: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Classify the accent in an audio file.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Results dictionary if successful
            - Error message if unsuccessful
        """
        if not self.is_loaded and not self.load_model():
            return False, None, "Failed to load model"
        
        try:
            start_time = time.time()
            
            # Load audio
            speech_array, sampling_rate = sf.read(audio_file)
            
            # Process the audio
            inputs = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the most likely accent
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_accent = self.id2label[pred_idx]
            
            # Get the confidence score
            confidence = probs[0][pred_idx].item()
            
            processing_time = time.time() - start_time
            
            return True, {
                "accent": pred_accent,
                "confidence": confidence,
                "processing_time": processing_time
            }, None
            
        except Exception as e:
            logging.error(f"Error during accent classification: {str(e)}")
            return False, None, f"Classification error: {str(e)}"

# Singleton instance for reuse
_classifier_instance = None

def get_classifier(model_name: str = 'm3hrdadfi/wav2vec2-large-xlsr-accent-classification',
                  use_quantization: bool = True,
                  cache_dir: Optional[str] = None) -> AccentClassifier:
    """
    Get or create the accent classifier instance.
    
    Args:
        model_name: HuggingFace model name
        use_quantization: Whether to apply model quantization
        cache_dir: Directory to cache the model
        
    Returns:
        AccentClassifier instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = AccentClassifier(
            model_name=model_name,
            use_quantization=use_quantization,
            cache_dir=cache_dir
        )
    
    return _classifier_instance

def classify_accent(audio_path: str, 
                    model_name: str = 'm3hrdadfi/wav2vec2-large-xlsr-accent-classification',
                    use_quantization: bool = True,
                    cache_dir: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Classify accent in an audio file.
    
    Args:
        audio_path: Path to the audio file
        model_name: HuggingFace model name
        use_quantization: Whether to apply model quantization
        cache_dir: Directory to cache the model
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Results dictionary if successful
        - Error message if unsuccessful
    """
    classifier = get_classifier(model_name, use_quantization, cache_dir)
    return classifier.classify(audio_path)
