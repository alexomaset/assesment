#!/usr/bin/env python3
"""
Video Downloader for REM Waste Accent Analyzer
Handles downloading videos from URLs, with special handling for Loom videos
"""

import os
import re
import json
import logging
import tempfile
import requests
from typing import Tuple, Optional, Dict, Any
from urllib.parse import urlparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoDownloader:
    """Handles video downloads from various sources with special Loom handling"""
    
    def __init__(self, max_size_mb: int = 500, timeout: int = 60):
        """
        Initialize the video downloader.
        
        Args:
            max_size_mb: Maximum video size in MB
            timeout: Download timeout in seconds
        """
        self.max_size_mb = max_size_mb
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def download_video(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Download a video from a URL.
        
        Args:
            url: URL of the video
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Path to downloaded video if successful, None otherwise
            - Error message if unsuccessful, None otherwise
        """
        # Check if this is a Loom video
        if 'loom.com' in url:
            return self._download_loom_video(url)
        
        # Handle direct MP4 URLs
        if url.lower().endswith('.mp4'):
            return self._download_direct_mp4(url)
        
        # Default handler for other URLs
        return self._download_generic_video(url)
    
    def _download_direct_mp4(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Download a direct MP4 URL"""
        try:
            logging.info(f"Downloading direct MP4 from URL: {url}")
            
            # Create a temporary file with .mp4 extension
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            
            # Stream the download to handle large files
            with requests.get(url, stream=True, headers=self.headers, timeout=self.timeout) as r:
                r.raise_for_status()
                
                # Check content size if available
                content_length = int(r.headers.get('content-length', 0))
                if content_length > self.max_size_mb * 1024 * 1024:
                    return False, None, f"Video is too large ({content_length/(1024*1024):.1f}MB > {self.max_size_mb}MB)"
                
                # Download the file
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logging.info(f"Successfully downloaded MP4 to: {temp_path}")
            return True, temp_path, None
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading MP4: {str(e)}")
            return False, None, f"Download error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return False, None, f"Unexpected error: {str(e)}"
    
    def _download_loom_video(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Download a video from Loom"""
        try:
            logging.info(f"Processing Loom video URL: {url}")
            
            # Extract the Loom video ID from the URL
            loom_id_match = re.search(r'loom\.com/(?:share|embed)/([a-zA-Z0-9]+)', url)
            if not loom_id_match:
                return False, None, "Invalid Loom URL format"
            
            loom_id = loom_id_match.group(1)
            logging.info(f"Extracted Loom ID: {loom_id}")
            
            # Get the Loom API data to find the direct MP4 URL
            api_url = f"https://www.loom.com/api/campaigns/sessions/{loom_id}"
            
            response = requests.get(api_url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                # Try alternative API endpoint
                api_url = f"https://www.loom.com/api/videos/{loom_id}"
                response = requests.get(api_url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code != 200:
                    return False, None, f"Failed to get Loom video data: HTTP {response.status_code}"
            
            video_data = response.json()
            
            # Extract the direct video URL from the response
            try:
                if 'data' in video_data:
                    # Try first data structure
                    mp4_url = video_data['data']['video_url']
                elif 'video_url' in video_data:
                    # Try alternative data structure
                    mp4_url = video_data['video_url']
                else:
                    # Try to extract from the video object
                    if 'video' in video_data:
                        video_obj = video_data['video']
                        if 'mp4_url' in video_obj:
                            mp4_url = video_obj['mp4_url']
                        elif 'url' in video_obj:
                            mp4_url = video_obj['url']
                        else:
                            return False, None, "Could not find video URL in Loom API response"
                    else:
                        return False, None, "Could not find video URL in Loom API response"
            
            except (KeyError, TypeError) as e:
                return False, None, f"Error extracting MP4 URL from Loom API: {str(e)}"
            
            logging.info(f"Found Loom MP4 URL: {mp4_url}")
            
            # Now download the direct MP4
            return self._download_direct_mp4(mp4_url)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error processing Loom video: {str(e)}")
            return False, None, f"Loom processing error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected error with Loom video: {str(e)}")
            return False, None, f"Unexpected Loom error: {str(e)}"
    
    def _download_generic_video(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Generic fallback for other video URLs"""
        try:
            logging.info(f"Attempting to download video from generic URL: {url}")
            
            # Try to determine the content type
            head_response = requests.head(url, headers=self.headers, timeout=self.timeout)
            content_type = head_response.headers.get('content-type', '')
            
            if 'video' in content_type or 'mp4' in content_type:
                # It's a video, download directly
                return self._download_direct_mp4(url)
            else:
                # Not a direct video link
                return False, None, f"URL does not point to a video file. Content-Type: {content_type}"
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error with generic URL: {str(e)}")
            return False, None, f"Download error: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return False, None, f"Unexpected error: {str(e)}"

# Create a function to use this class more easily
def download_video_from_url(url: str, max_size_mb: int = 500, timeout: int = 60) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Download a video from a URL.
    
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
    downloader = VideoDownloader(max_size_mb=max_size_mb, timeout=timeout)
    return downloader.download_video(url)
