import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Configuration with environment variable fallbacks
class Config:
    # Cache settings
    CACHE_DIR = os.getenv('CACHE_DIR', './cache')
    CACHE_TTL = int(os.getenv('CACHE_TTL', 86400))  # 24 hours in seconds
    
    # Audio processing
    MAX_AUDIO_LENGTH = int(os.getenv('MAX_AUDIO_LENGTH', 5))  # Maximum audio length in seconds
    
    # Model settings
    MODEL_QUANTIZATION = os.getenv('MODEL_QUANTIZATION', 'True').lower() in ('true', '1', 't')
    PARALLEL_PROCESSING = os.getenv('PARALLEL_PROCESSING', 'True').lower() in ('true', '1', 't')
    
    # Deployment settings
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    PORT = int(os.getenv('PORT', 8501))
    
    # Make sure cache directory exists
    @classmethod
    def ensure_cache_dir(cls):
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        
config = Config()
config.ensure_cache_dir()
