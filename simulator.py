#!/usr/bin/env python3
"""
Simulation module for REM Waste Accent Analyzer
Provides fallback implementations for audio processing dependencies
"""

import os
import logging
import numpy as np
import wave
import tempfile
from typing import Tuple, List, Dict, Any, Optional

class AudioSimulator:
    """Simulates audio processing operations when real libraries aren't available"""
    
    @staticmethod
    def read_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Simulates reading audio data from a file
        
        Args:
            file_path: Path to the audio file
            sample_rate: Target sample rate
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        logging.info(f"Simulating audio read from: {file_path}")
        # Generate 3 seconds of dummy audio
        duration = 3
        samples = duration * sample_rate
        audio_data = np.random.uniform(-0.01, 0.01, samples).astype(np.float32)
        
        # Add some patterns to make it look like speech
        # Create a few "speech-like" formants
        for freq in [100, 200, 500]:
            t = np.arange(samples) / sample_rate
            audio_data += 0.05 * np.sin(2 * np.pi * freq * t)
        
        return audio_data, sample_rate
    
    @staticmethod
    def write_audio(file_path: str, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Simulates writing audio data to a file
        
        Args:
            file_path: Output file path
            audio_data: Audio data as numpy array
            sample_rate: Sample rate
        """
        logging.info(f"Simulating audio write to: {file_path}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write a basic WAV file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            
            # Ensure audio_data is within [-1, 1]
            audio_data = np.clip(audio_data, -1, 1)
            
            # Convert float to int16
            int_data = (audio_data * 32767).astype(np.int16)
            wf.writeframes(int_data.tobytes())
    
    @staticmethod
    def create_audio_chunks(audio_path: str, chunk_duration: int = 5, 
                           target_sample_rate: int = 16000) -> List[str]:
        """
        Simulates creating audio chunks from a longer file
        
        Args:
            audio_path: Path to the source audio file
            chunk_duration: Duration of each chunk in seconds
            target_sample_rate: Sample rate for the chunks
            
        Returns:
            List of paths to the created audio chunks
        """
        logging.info(f"Simulating creation of audio chunks from: {audio_path}")
        
        # Create a temporary directory for chunks
        chunk_dir = tempfile.mkdtemp()
        
        # Create 2-3 chunks (randomly)
        num_chunks = np.random.randint(2, 4)
        chunk_paths = []
        
        for i in range(num_chunks):
            # Create a path for this chunk
            chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
            
            # Generate random audio for this chunk
            duration = chunk_duration
            samples = duration * target_sample_rate
            audio_data = np.random.uniform(-0.01, 0.01, samples).astype(np.float32)
            
            # Add some patterns to make it look like speech
            t = np.arange(samples) / target_sample_rate
            audio_data += 0.1 * np.sin(2 * np.pi * 200 * t)
            
            # Write the chunk
            AudioSimulator.write_audio(chunk_path, audio_data, target_sample_rate)
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    @staticmethod
    def classify_audio(audio_path: str) -> Dict[str, Any]:
        """
        Simulates accent classification on an audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict with classification results
        """
        logging.info(f"Simulating accent classification for: {audio_path}")
        
        # List of possible accents
        accents = [
            "us", "england", "canada", "australia", "indian", 
            "scotland", "ireland", "southatlandtic", "african",
            "philippines", "hongkong", "malaysia", "singapore",
            "bermuda", "wales", "newzealand"
        ]
        
        # Choose a random accent with higher probability for common ones
        weights = [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 
                  0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
        accent = np.random.choice(accents, p=weights)
        
        # Generate a confidence score (higher for common accents)
        base_confidence = 0.7 if accent in ["us", "england", "canada", "australia"] else 0.5
        confidence = min(0.99, base_confidence + np.random.uniform(0, 0.25))
        
        # Create simulated result
        result = {
            "accent": accent,
            "confidence": confidence,
            "processing_time": np.random.uniform(0.5, 2.0)
        }
        
        return result

# Soundfile simulator as a drop-in replacement
class SoundFileSimulator:
    """Drop-in replacement for the soundfile library"""
    
    @staticmethod
    def read(file_path, **kwargs):
        """Read audio data - simulator implementation"""
        return AudioSimulator.read_audio(
            file_path, 
            sample_rate=kwargs.get('samplerate', 16000)
        )
    
    @staticmethod
    def write(file_path, data, samplerate, **kwargs):
        """Write audio data - simulator implementation"""
        AudioSimulator.write_audio(file_path, data, samplerate)
