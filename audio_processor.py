#!/usr/bin/env python3
"""
Audio Processor Module for REM Waste Accent Analyzer
Provides advanced audio processing capabilities:
1. Noise reduction
2. Voice activity detection
3. Sample rate conversion
4. Audio chunking for long videos
"""

import os
import logging
import tempfile
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from typing import List, Tuple, Optional
import webrtcvad
import struct
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioProcessor:
    """Class for advanced audio processing capabilities"""
    
    def __init__(self, target_sample_rate: int = 16000, chunk_duration: int = 5):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate (int): Target sample rate in Hz (default: 16000 Hz)
            chunk_duration (int): Duration of each audio chunk in seconds (default: 5s)
        """
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Aggressiveness mode (0-3), 3 is most aggressive
    
    def process_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Tuple[bool, List[str], Optional[str]]:
        """
        Process audio file with noise reduction, VAD, and chunking.
        
        Args:
            audio_path (str): Path to the input audio file
            output_dir (str, optional): Directory to save processed chunks. If None, uses temp directory.
            
        Returns:
            Tuple[bool, List[str], Optional[str]]: Success status, list of chunk file paths, error message
        """
        try:
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = tempfile.mkdtemp()
            
            # Load audio
            logging.info(f"Loading audio from {audio_path}")
            y, sr = librosa.load(audio_path, sr=None)
            
            # Convert sample rate if needed
            if sr != self.target_sample_rate:
                logging.info(f"Converting sample rate from {sr}Hz to {self.target_sample_rate}Hz")
                y = self._convert_sample_rate(y, sr, self.target_sample_rate)
                sr = self.target_sample_rate
            
            # Apply noise reduction
            logging.info("Applying noise reduction")
            y_reduced = self._reduce_noise(y, sr)
            
            # Detect voice activity
            logging.info("Detecting voice activity")
            y_voice = self._detect_voice_activity(y_reduced, sr)
            
            # Create chunks
            logging.info(f"Creating audio chunks of {self.chunk_duration}s each")
            chunk_paths = self._create_chunks(y_voice, sr, output_dir)
            
            if not chunk_paths:
                return False, [], "No voice detected in audio or chunking failed"
            
            return True, chunk_paths, None
            
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            return False, [], f"Audio processing error: {str(e)}"
    
    def _convert_sample_rate(self, y: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Convert audio sample rate using high-quality resampling"""
        if original_sr == target_sr:
            return y
        
        # Calculate number of samples for the target sample rate
        num_samples = int(len(y) * target_sr / original_sr)
        
        # Use Scipy's high-quality resampling
        y_resampled = signal.resample(y, num_samples)
        
        return y_resampled
    
    def _reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction to audio signal"""
        # Estimate noise from a short segment (first 0.5 seconds or less)
        noise_length = min(int(sr * 0.5), len(y) // 4)
        
        # Apply noise reduction (stationary=True assumes constant noise profile)
        y_reduced = nr.reduce_noise(
            y=y, 
            sr=sr,
            stationary=True,
            prop_decrease=0.75
        )
        
        return y_reduced
    
    def _detect_voice_activity(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect and extract segments with voice activity
        using WebRTC Voice Activity Detection
        """
        # WebRTC VAD works with specific frame durations (10, 20, or 30 ms)
        frame_duration_ms = 20
        samples_per_frame = int(sr * frame_duration_ms / 1000)
        
        # Normalize audio to 16-bit PCM range
        y_norm = np.clip(y * 32768, -32768, 32767).astype(np.int16)
        
        # Create mask for voice frames
        voice_mask = np.zeros_like(y_norm, dtype=bool)
        
        for i in range(0, len(y_norm) - samples_per_frame, samples_per_frame):
            frame = y_norm[i:i+samples_per_frame]
            if len(frame) == samples_per_frame:  # Ensure complete frame
                # Convert to bytes for WebRTC VAD
                frame_bytes = struct.pack("h" * len(frame), *frame)
                # Mark as voice if VAD detects speech
                try:
                    if self.vad.is_speech(frame_bytes, sr):
                        voice_mask[i:i+samples_per_frame] = True
                except Exception:
                    # If VAD fails for a frame, assume it's not speech
                    pass
        
        # Apply smoothing - extend voice segments by 100ms each side
        smoothing_frames = int(sr * 0.1)
        voice_mask_smooth = np.copy(voice_mask)
        
        for i in range(len(voice_mask)):
            if voice_mask[i]:
                start = max(0, i - smoothing_frames)
                end = min(len(voice_mask), i + smoothing_frames)
                voice_mask_smooth[start:end] = True
        
        # Extract only the voice segments
        y_voice = y * voice_mask_smooth
        
        # If almost no voice was detected, return the original audio
        if np.sum(voice_mask_smooth) < 0.1 * len(y):
            logging.warning("Very little voice activity detected, using original audio")
            return y
        
        return y_voice
    
    def _create_chunks(self, y: np.ndarray, sr: int, output_dir: str) -> List[str]:
        """
        Split audio into fixed-duration chunks with meaningful speech
        
        Args:
            y (np.ndarray): Audio signal
            sr (int): Sample rate
            output_dir (str): Directory to save chunks
            
        Returns:
            List[str]: List of file paths to saved chunks
        """
        chunk_paths = []
        chunk_samples = self.chunk_duration * sr
        
        # Skip silent parts (where amplitude is very low)
        silence_threshold = 0.01
        is_silence = np.abs(y) < silence_threshold
        
        # Find contiguous non-silent segments
        non_silent_segments = []
        in_segment = False
        segment_start = 0
        
        for i, silent in enumerate(is_silence):
            if not silent and not in_segment:
                # Start of a non-silent segment
                in_segment = True
                segment_start = i
            elif silent and in_segment and i - segment_start > 0.2 * sr:  # At least 0.2s of speech
                # End of a non-silent segment
                in_segment = False
                non_silent_segments.append((segment_start, i))
        
        # Add the last segment if we're still in one
        if in_segment:
            non_silent_segments.append((segment_start, len(y)))
        
        # Merge very close segments
        merged_segments = []
        gap_threshold = 0.5 * sr  # 0.5 seconds
        
        if non_silent_segments:
            current_start, current_end = non_silent_segments[0]
            
            for start, end in non_silent_segments[1:]:
                if start - current_end < gap_threshold:
                    # Merge with previous segment
                    current_end = end
                else:
                    # Save current segment and start a new one
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            # Add the last segment
            merged_segments.append((current_start, current_end))
        
        # Create fixed-duration chunks from the merged segments
        chunk_idx = 0
        for segment_start, segment_end in merged_segments:
            segment_audio = y[segment_start:segment_end]
            
            # Skip very short segments
            if len(segment_audio) < 0.5 * sr:
                continue
                
            # Create fixed-duration chunks from this segment
            for chunk_start in range(0, len(segment_audio), chunk_samples):
                chunk_end = min(chunk_start + chunk_samples, len(segment_audio))
                chunk_audio = segment_audio[chunk_start:chunk_end]
                
                # Skip very short chunks
                if len(chunk_audio) < 1 * sr:  # At least 1 second
                    continue
                
                # Pad short chunks to full duration
                if len(chunk_audio) < chunk_samples:
                    padding = chunk_samples - len(chunk_audio)
                    chunk_audio = np.pad(chunk_audio, (0, padding), mode='constant')
                
                # Save chunk to file
                chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:03d}.wav")
                sf.write(chunk_path, chunk_audio, sr)
                chunk_paths.append(chunk_path)
                chunk_idx += 1
        
        # If we couldn't create any chunks (e.g., all audio was silence),
        # create at least one chunk from the middle of the audio
        if not chunk_paths and len(y) > 0:
            logging.warning("No suitable chunks found, creating one from the middle of the audio")
            mid_point = len(y) // 2
            start = max(0, mid_point - (chunk_samples // 2))
            end = min(len(y), start + chunk_samples)
            
            chunk_audio = y[start:end]
            if len(chunk_audio) < chunk_samples:
                chunk_audio = np.pad(chunk_audio, (0, chunk_samples - len(chunk_audio)), mode='constant')
            
            chunk_path = os.path.join(output_dir, "chunk_000.wav")
            sf.write(chunk_path, chunk_audio, sr)
            chunk_paths.append(chunk_path)
        
        logging.info(f"Created {len(chunk_paths)} audio chunks")
        return chunk_paths


def process_audio_file(
    input_file: str, 
    target_sample_rate: int = 16000, 
    chunk_duration: int = 5,
    output_dir: Optional[str] = None
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Process an audio file with advanced techniques.
    Convenience function to avoid creating an AudioProcessor instance.
    
    Args:
        input_file (str): Path to input audio file
        target_sample_rate (int): Target sample rate in Hz
        chunk_duration (int): Duration of each chunk in seconds
        output_dir (str, optional): Directory to save processed chunks
        
    Returns:
        Tuple[bool, List[str], Optional[str]]: Success status, list of chunk file paths, error message
    """
    processor = AudioProcessor(target_sample_rate, chunk_duration)
    return processor.process_audio(input_file, output_dir)
