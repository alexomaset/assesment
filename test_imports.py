#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

print("Testing imports...")
try:
    import moviepy
    print(f"✅ Successfully imported moviepy {moviepy.__version__}")
except ImportError as e:
    print(f"❌ Failed to import moviepy: {str(e)}")

try:
    from moviepy.editor import VideoFileClip
    print("✅ Successfully imported VideoFileClip")
except ImportError as e:
    print(f"❌ Failed to import VideoFileClip: {str(e)}")

print("\nEnvironment information:")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
