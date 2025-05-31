#!/usr/bin/env python3
"""
Loom Video Handler for REM Waste Accent Analyzer
Special handling for Loom videos
"""

import os
import re
import logging
import tempfile
import requests
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_loom_video(url: str, max_size_mb: int = 500, timeout: int = 60) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Download a video from Loom.
    
    Args:
        url: Loom URL
        max_size_mb: Maximum size in MB
        timeout: Download timeout in seconds
        
    Returns:
        Tuple containing:
        - Success status (bool)
        - Path to downloaded video if successful, None otherwise
        - Error message if unsuccessful, None otherwise
    """
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
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(api_url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            # Try alternative API endpoint
            api_url = f"https://www.loom.com/api/videos/{loom_id}"
            response = requests.get(api_url, headers=headers, timeout=timeout)
            
            if response.status_code != 200:
                return False, None, f"Failed to get Loom video data: HTTP {response.status_code}"
        
        # Extract the direct video URL from the response
        try:
            video_data = response.json()
            
            if 'data' in video_data and isinstance(video_data['data'], dict):
                # Try first data structure
                mp4_url = video_data['data'].get('video_url')
            elif 'video_url' in video_data:
                # Try alternative data structure
                mp4_url = video_data['video_url']
            elif 'video' in video_data and isinstance(video_data['video'], dict):
                # Try to extract from the video object
                video_obj = video_data['video']
                mp4_url = video_obj.get('mp4_url') or video_obj.get('url')
            else:
                # Last attempt - try to find any URL field that contains mp4
                for key, value in video_data.items():
                    if isinstance(value, str) and value.endswith('.mp4'):
                        mp4_url = value
                        break
                else:
                    return False, None, "Could not find video URL in Loom API response"
            
            if not mp4_url:
                return False, None, "Could not find video URL in Loom API response"
                
            # For demo purposes, create a placeholder MP4
            logging.info(f"Found Loom MP4 URL: {mp4_url}")
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            
            # Download the MP4 file
            with requests.get(mp4_url, stream=True, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            logging.info(f"Downloaded Loom video to: {temp_path}")
            return True, temp_path, None
            
        except (KeyError, TypeError) as e:
            return False, None, f"Error extracting MP4 URL from Loom API: {str(e)}"
        except Exception as e:
            logging.error(f"Error downloading MP4: {str(e)}")
            return False, None, f"Download error: {str(e)}"
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error processing Loom video: {str(e)}")
        return False, None, f"Loom processing error: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error with Loom video: {str(e)}")
        return False, None, f"Unexpected Loom error: {str(e)}"
