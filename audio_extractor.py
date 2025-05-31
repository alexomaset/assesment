#!/usr/bin/env python3
"""
Audio Extractor for REM Waste Accent Analyzer
Handles extracting audio from video files without external dependencies
"""

import os
import logging
import tempfile
import subprocess
from typing import Tuple, Optional
# Import individual components instead of the full editor
from moviepy.video.io.VideoFileClip import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioExtractor:
    """Extracts audio from video files"""
    
    @staticmethod
    def extract_audio(video_path: str, output_path: Optional[str] = None, 
                     sample_rate: int = 16000) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Path for the output audio file (optional)
            sample_rate: Target sample rate in Hz
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Path to extracted audio if successful, None otherwise
            - Error message if unsuccessful, None otherwise
        """
        try:
            logging.info(f"Extracting audio from: {video_path}")
            
            # If no output path is provided, create a temporary file
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
            
            # Use moviepy to extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                return False, None, "No audio track found in video"
            
            # Write audio to file with specified sample rate
            audio.write_audiofile(output_path, fps=sample_rate, 
                                 nbytes=2,  # 16-bit
                                 ffmpeg_params=["-ac", "1"])  # Mono
            
            # Close resources
            audio.close()
            video.close()
            
            logging.info(f"Successfully extracted audio to: {output_path}")
            return True, output_path, None
            
        except Exception as e:
            logging.error(f"Error extracting audio: {str(e)}")
            
            # Try fallback method if moviepy fails
            return AudioExtractor._extract_audio_fallback(video_path, output_path, sample_rate)
    
    @staticmethod
    def _extract_audio_fallback(video_path: str, output_path: Optional[str] = None, 
                              sample_rate: int = 16000) -> Tuple[bool, Optional[str], Optional[str]]:
        """Fallback method using subprocess if moviepy fails"""
        try:
            logging.info(f"Using fallback method to extract audio from: {video_path}")
            
            # If no output path is provided, create a temporary file
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
            
            # Try using ffmpeg directly
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                output_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                return False, None, f"FFmpeg extraction error: {error_msg}"
            
            logging.info(f"Successfully extracted audio using fallback to: {output_path}")
            return True, output_path, None
            
        except Exception as e:
            logging.error(f"Fallback audio extraction failed: {str(e)}")
            return False, None, f"Audio extraction failed: {str(e)}"

# Function for easier usage
def extract_audio_from_video(video_path: str, output_path: Optional[str] = None, 
                           sample_rate: int = 16000) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path for the output audio file (optional)
        sample_rate: Target sample rate in Hz
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Path to extracted audio if successful, None otherwise
        - Error message if unsuccessful, None otherwise
    """
    return AudioExtractor.extract_audio(video_path, output_path, sample_rate)
