#!/usr/bin/env python3
"""
Model utilities for REM Waste Accent Analyzer
"""

import os
import logging
from typing import Dict, Any, Optional
import torch
from model_handler import AccentClassifier

# Dictionary to cache loaded models
_model_cache = {}

def get_or_load_model(model_name: str, use_quantization: bool = True, 
                     cache_dir: Optional[str] = './cache/models') -> AccentClassifier:
    """
    Get a cached model or load it if not already in cache.
    
    Args:
        model_name: Name of the Hugging Face model
        use_quantization: Whether to apply quantization
        cache_dir: Directory to cache the model
        
    Returns:
        Loaded model instance
    """
    global _model_cache
    
    # Create cache directory if it doesn't exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Check if model is already in cache
    if model_name in _model_cache:
        logging.info(f"Using cached model: {model_name}")
        return _model_cache[model_name]
    
    # Load the model
    logging.info(f"Loading model: {model_name}")
    model = AccentClassifier(model_name=model_name, use_quantization=use_quantization, cache_dir=cache_dir)
    
    # Ensure model is loaded
    if not model.is_loaded:
        model.load_model()
    
    # Cache the model
    _model_cache[model_name] = model
    
    return model

def quantize_model(model):
    """
    Apply quantization to a PyTorch model to reduce size and improve inference speed.
    
    Args:
        model: PyTorch model to quantize
        
    Returns:
        Quantized model
    """
    if torch.cuda.is_available():
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    return model

def limit_audio_length(audio_data, sample_rate: int, max_seconds: int = 5):
    """
    Limit audio length to a maximum number of seconds.
    Takes the middle section of audio for consistency.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        max_seconds: Maximum length in seconds
        
    Returns:
        Limited audio data
    """
    max_samples = max_seconds * sample_rate
    
    if len(audio_data) <= max_samples:
        return audio_data
    
    # Take the middle section
    start = (len(audio_data) - max_samples) // 2
    end = start + max_samples
    
    return audio_data[start:end]
