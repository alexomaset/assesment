#!/usr/bin/env python3
"""
Main Processor for REM Waste Accent Analyzer
Integrates all components into a complete processing pipeline
"""

import os
import time
import asyncio
import logging
from typing import Tuple, Optional, Dict, Any, List
import tempfile

# Import our real implementations
from model_handler import classify_accent
from cache_manager import get_cache_manager
from video_downloader import download_video_from_url
from audio_extractor import extract_audio_from_video
from audio_processor import process_audio_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up cache manager
cache_manager = get_cache_manager()

async def process_video_url_async(url: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Async wrapper for process_video_url
    
    Args:
        url: URL of the video to process
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Results dictionary if successful, None otherwise
        - Error message if unsuccessful, None otherwise
    """
    return process_video_url(url)

async def classify_audio_chunk_async(audio_path: str) -> Dict[str, Any]:
    """
    Async wrapper for classify_audio_chunk
    
    Args:
        audio_path: Path to the audio chunk
        
    Returns:
        Classification results
    """
    # Check cache first
    cached_result = cache_manager.get_cached_result(audio_path)
    if cached_result:
        return cached_result
    
    # Classify and cache
    success, result, error = classify_accent(audio_path)
    if success and result:
        cache_manager.cache_result(audio_path, result)
        return result
    else:
        # Return fallback result if classification fails
        return {
            "accent": "unknown",
            "confidence": 0.5,
            "processing_time": 0.0,
            "error": error or "Classification failed"
        }

async def extract_audio_from_video_async(video_path: str, target_sample_rate: int = 16000, 
                                        chunk_duration: int = 5) -> Tuple[bool, List[str], Optional[str]]:
    """
    Async wrapper for extract_audio_from_video
    
    Args:
        video_path: Path to the video file
        target_sample_rate: Target sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - List of paths to audio chunks if successful, empty list otherwise
        - Error message if unsuccessful, None otherwise
    """
    # Check cache for audio path
    cached_audio = cache_manager.get_cached_audio_path(video_path)
    if cached_audio:
        # Check cache for chunks
        cached_chunks = cache_manager.get_cached_chunks(cached_audio)
        if cached_chunks:
            return True, cached_chunks, None
        
        # Process the cached audio
        success, chunks, error = await process_audio_async(cached_audio, target_sample_rate, chunk_duration)
        if success:
            cache_manager.cache_chunks(cached_audio, chunks)
        return success, chunks, error
    
    # Extract audio from video
    success, audio_path, error = extract_audio_from_video(video_path, None, target_sample_rate)
    if not success:
        return False, [], error
    
    # Cache the audio path
    cache_manager.cache_audio_path(video_path, audio_path)
    
    # Process the audio into chunks
    success, chunks, error = await process_audio_async(audio_path, target_sample_rate, chunk_duration)
    if success:
        cache_manager.cache_chunks(audio_path, chunks)
    
    return success, chunks, error

async def process_audio_async(audio_path: str, target_sample_rate: int = 16000, 
                            chunk_duration: int = 5) -> Tuple[bool, List[str], Optional[str]]:
    """
    Process audio file with advanced techniques.
    
    Args:
        audio_path: Path to the audio file
        target_sample_rate: Target sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - List of paths to audio chunks if successful, empty list otherwise
        - Error message if unsuccessful, None otherwise
    """
    return process_audio_file(audio_path, target_sample_rate, chunk_duration)

async def download_video_from_url_async(url: str, max_size_mb: int = 500, 
                                      timeout: int = 60) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Async wrapper for download_video_from_url
    
    Args:
        url: URL of the video
        max_size_mb: Maximum video size in MB
        timeout: Download timeout in seconds
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Path to downloaded video if successful, None otherwise
        - Error message if unsuccessful, None otherwise
    """
    # Check cache first
    cached_video = cache_manager.get_cached_video_path(url)
    if cached_video:
        return True, cached_video, None
    
    # Download and cache
    success, video_path, error = download_video_from_url(url, max_size_mb, timeout)
    if success:
        cache_manager.cache_video_path(url, video_path)
    
    return success, video_path, error

def process_video_url(url: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Process a video URL through the entire pipeline:
    1. Download the video
    2. Extract and process audio (with noise reduction, VAD, chunking)
    3. Classify accent for each audio chunk
    4. Aggregate results
    
    Args:
        url: URL of the video to process
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Results dictionary if successful, None otherwise
        - Error message if unsuccessful, None otherwise
    """
    start_time = time.time()
    
    try:
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Step 1: Download video
        logging.info(f"Downloading video from URL: {url}")
        success, video_path, error = loop.run_until_complete(download_video_from_url_async(url))
        if not success:
            return False, None, f"Video download failed: {error}"
        
        # Step 2: Extract audio and process into chunks
        logging.info(f"Extracting and processing audio from video: {video_path}")
        success, chunk_paths, error = loop.run_until_complete(extract_audio_from_video_async(video_path))
        if not success:
            return False, None, f"Audio extraction failed: {error}"
        
        # Step 3: Classify each chunk in parallel
        logging.info(f"Classifying {len(chunk_paths)} audio chunks")
        classification_tasks = [classify_audio_chunk_async(chunk) for chunk in chunk_paths]
        chunk_results = loop.run_until_complete(asyncio.gather(*classification_tasks))
        
        # Step 4: Aggregate results
        # Find the most common accent with highest average confidence
        accent_counts = {}
        accent_confidences = {}
        
        for result in chunk_results:
            accent = result["accent"]
            confidence = result["confidence"]
            
            if accent not in accent_counts:
                accent_counts[accent] = 0
                accent_confidences[accent] = []
            
            accent_counts[accent] += 1
            accent_confidences[accent].append(confidence)
        
        # Calculate scores
        accent_scores = {}
        for accent in accent_counts:
            count = accent_counts[accent]
            avg_confidence = sum(accent_confidences[accent]) / len(accent_confidences[accent])
            # Weight by both count and confidence
            accent_scores[accent] = count * avg_confidence
        
        # Get the accent with the highest score
        if not accent_scores:
            return False, None, "No accent classifications were successful"
        
        best_accent = max(accent_scores, key=accent_scores.get)
        avg_confidence = sum(accent_confidences[best_accent]) / len(accent_confidences[best_accent])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the results
        results = {
            "accent": best_accent,
            "confidence": avg_confidence,
            "processing_time": processing_time,
            "num_chunks": len(chunk_paths),
            "chunk_results": chunk_results
        }
        
        return True, results, None
        
    except Exception as e:
        logging.error(f"Error processing video URL: {str(e)}")
        return False, None, f"Processing error: {str(e)}"
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass
