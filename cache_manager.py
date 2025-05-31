#!/usr/bin/env python3
"""
Cache Manager for REM Waste Accent Analyzer
Implements multi-level caching (URL, video, and chunk-based)
"""

import os
import json
import hashlib
import time
import logging
import shutil
from typing import Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CacheManager:
    """Manages the multi-level caching system for improved performance"""
    
    def __init__(self, cache_dir: str = './cache', ttl_seconds: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Base directory for caches
            ttl_seconds: Time-to-live in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        
        # Create cache subdirectories
        self.url_cache_dir = os.path.join(cache_dir, 'url_cache')
        self.video_cache_dir = os.path.join(cache_dir, 'video_cache')
        self.audio_cache_dir = os.path.join(cache_dir, 'audio_cache')
        self.chunk_cache_dir = os.path.join(cache_dir, 'chunk_cache')
        self.result_cache_dir = os.path.join(cache_dir, 'result_cache')
        
        # Create directories if they don't exist
        for directory in [self.url_cache_dir, self.video_cache_dir, 
                         self.audio_cache_dir, self.chunk_cache_dir,
                         self.result_cache_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _get_hash(self, value: str) -> str:
        """Create a hash from a string value"""
        return hashlib.md5(value.encode('utf-8')).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a timestamp is expired based on TTL"""
        return (time.time() - timestamp) > self.ttl_seconds
    
    # URL Cache functions
    def get_cached_video_path(self, url: str) -> Optional[str]:
        """
        Get cached video path for a URL if it exists and is not expired.
        
        Args:
            url: URL of the video
            
        Returns:
            Path to the cached video file or None if not found/expired
        """
        url_hash = self._get_hash(url)
        url_cache_file = os.path.join(self.url_cache_dir, f"{url_hash}.json")
        
        if not os.path.exists(url_cache_file):
            return None
        
        try:
            with open(url_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            if self._is_expired(cache_data['timestamp']):
                logging.info(f"Cache expired for URL: {url}")
                os.remove(url_cache_file)
                return None
            
            # Check if the video file still exists
            video_path = cache_data['video_path']
            if not os.path.exists(video_path):
                logging.info(f"Cached video file not found: {video_path}")
                os.remove(url_cache_file)
                return None
            
            logging.info(f"Using cached video for URL: {url}")
            return video_path
            
        except Exception as e:
            logging.error(f"Error reading URL cache: {str(e)}")
            return None
    
    def cache_video_path(self, url: str, video_path: str) -> bool:
        """
        Cache the path to a downloaded video file.
        
        Args:
            url: URL of the video
            video_path: Path to the downloaded video file
            
        Returns:
            bool: Success status
        """
        url_hash = self._get_hash(url)
        url_cache_file = os.path.join(self.url_cache_dir, f"{url_hash}.json")
        
        try:
            cache_data = {
                'url': url,
                'video_path': video_path,
                'timestamp': time.time()
            }
            
            with open(url_cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logging.info(f"Cached video path for URL: {url}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching video path: {str(e)}")
            return False
    
    # Video-to-Audio Cache functions
    def get_cached_audio_path(self, video_path: str) -> Optional[str]:
        """
        Get cached audio path for a video if it exists and is not expired.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the cached audio file or None if not found/expired
        """
        video_hash = self._get_hash(video_path)
        audio_cache_file = os.path.join(self.audio_cache_dir, f"{video_hash}.json")
        
        if not os.path.exists(audio_cache_file):
            return None
        
        try:
            with open(audio_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            if self._is_expired(cache_data['timestamp']):
                logging.info(f"Cache expired for video: {video_path}")
                os.remove(audio_cache_file)
                return None
            
            # Check if the audio file still exists
            audio_path = cache_data['audio_path']
            if not os.path.exists(audio_path):
                logging.info(f"Cached audio file not found: {audio_path}")
                os.remove(audio_cache_file)
                return None
            
            logging.info(f"Using cached audio for video: {video_path}")
            return audio_path
            
        except Exception as e:
            logging.error(f"Error reading audio cache: {str(e)}")
            return None
    
    def cache_audio_path(self, video_path: str, audio_path: str) -> bool:
        """
        Cache the path to an extracted audio file.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the extracted audio file
            
        Returns:
            bool: Success status
        """
        video_hash = self._get_hash(video_path)
        audio_cache_file = os.path.join(self.audio_cache_dir, f"{video_hash}.json")
        
        try:
            cache_data = {
                'video_path': video_path,
                'audio_path': audio_path,
                'timestamp': time.time()
            }
            
            with open(audio_cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logging.info(f"Cached audio path for video: {video_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching audio path: {str(e)}")
            return False
    
    # Audio Chunk Cache functions
    def get_cached_chunks(self, audio_path: str) -> Optional[list]:
        """
        Get cached audio chunks if they exist and are not expired.
        
        Args:
            audio_path: Path to the full audio file
            
        Returns:
            List of paths to the cached audio chunks or None if not found/expired
        """
        audio_hash = self._get_hash(audio_path)
        chunk_cache_file = os.path.join(self.chunk_cache_dir, f"{audio_hash}.json")
        
        if not os.path.exists(chunk_cache_file):
            return None
        
        try:
            with open(chunk_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            if self._is_expired(cache_data['timestamp']):
                logging.info(f"Cache expired for audio chunks: {audio_path}")
                os.remove(chunk_cache_file)
                return None
            
            # Check if all chunk files still exist
            chunk_paths = cache_data['chunk_paths']
            for chunk_path in chunk_paths:
                if not os.path.exists(chunk_path):
                    logging.info(f"Cached chunk file not found: {chunk_path}")
                    os.remove(chunk_cache_file)
                    return None
            
            logging.info(f"Using cached audio chunks for: {audio_path}")
            return chunk_paths
            
        except Exception as e:
            logging.error(f"Error reading chunk cache: {str(e)}")
            return None
    
    def cache_chunks(self, audio_path: str, chunk_paths: list) -> bool:
        """
        Cache the paths to audio chunks.
        
        Args:
            audio_path: Path to the full audio file
            chunk_paths: List of paths to the audio chunks
            
        Returns:
            bool: Success status
        """
        audio_hash = self._get_hash(audio_path)
        chunk_cache_file = os.path.join(self.chunk_cache_dir, f"{audio_hash}.json")
        
        try:
            cache_data = {
                'audio_path': audio_path,
                'chunk_paths': chunk_paths,
                'timestamp': time.time()
            }
            
            with open(chunk_cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logging.info(f"Cached {len(chunk_paths)} chunks for audio: {audio_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching chunks: {str(e)}")
            return False
    
    # Results Cache functions
    def get_cached_result(self, chunk_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached classification result for an audio chunk.
        
        Args:
            chunk_path: Path to the audio chunk
            
        Returns:
            Classification result or None if not found/expired
        """
        chunk_hash = self._get_hash(chunk_path)
        result_cache_file = os.path.join(self.result_cache_dir, f"{chunk_hash}.json")
        
        if not os.path.exists(result_cache_file):
            return None
        
        try:
            with open(result_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            if self._is_expired(cache_data['timestamp']):
                logging.info(f"Cache expired for result: {chunk_path}")
                os.remove(result_cache_file)
                return None
            
            logging.info(f"Using cached result for chunk: {chunk_path}")
            return cache_data['result']
            
        except Exception as e:
            logging.error(f"Error reading result cache: {str(e)}")
            return None
    
    def cache_result(self, chunk_path: str, result: Dict[str, Any]) -> bool:
        """
        Cache a classification result for an audio chunk.
        
        Args:
            chunk_path: Path to the audio chunk
            result: Classification result
            
        Returns:
            bool: Success status
        """
        chunk_hash = self._get_hash(chunk_path)
        result_cache_file = os.path.join(self.result_cache_dir, f"{chunk_hash}.json")
        
        try:
            cache_data = {
                'chunk_path': chunk_path,
                'result': result,
                'timestamp': time.time()
            }
            
            with open(result_cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logging.info(f"Cached result for chunk: {chunk_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching result: {str(e)}")
            return False
    
    def clear_expired_cache(self) -> int:
        """
        Clear all expired cache entries.
        
        Returns:
            int: Number of cleared cache entries
        """
        cleared_count = 0
        
        # Clear URL cache
        for cache_file in os.listdir(self.url_cache_dir):
            if cache_file.endswith('.json'):
                cache_path = os.path.join(self.url_cache_dir, cache_file)
                try:
                    with open(cache_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    if self._is_expired(cache_data['timestamp']):
                        os.remove(cache_path)
                        cleared_count += 1
                except Exception:
                    # If there's an error reading the file, remove it
                    os.remove(cache_path)
                    cleared_count += 1
        
        # Similar logic for other cache types
        # ... (implement for other cache directories)
        
        logging.info(f"Cleared {cleared_count} expired cache entries")
        return cleared_count

# Singleton instance
_cache_manager_instance = None

def get_cache_manager(cache_dir: str = './cache', ttl_seconds: int = 86400) -> CacheManager:
    """
    Get or create the cache manager instance.
    
    Args:
        cache_dir: Base directory for caches
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        CacheManager instance
    """
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager(
            cache_dir=cache_dir,
            ttl_seconds=ttl_seconds
        )
    
    return _cache_manager_instance
