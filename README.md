# RemWaste Accent Analyzer

A Python tool for analyzing accents in speech data.

## Description

RemWaste Accent Analyzer is a tool designed to process and analyze speech data to identify and classify different accents. This project uses audio processing techniques and machine learning to extract features from speech samples and perform accent analysis.

## Features

- Video download from public URLs (MP4 and Loom formats)
- Audio extraction from video files
- Conversion to 16kHz mono WAV format
- Intelligent sampling of middle audio segments
- Accent classification using state-of-the-art Hugging Face models
- Support for 16 different accent categories
- Confidence scoring and processing time metrics
- GPU acceleration (when available)

## Installation

### Prerequisites

This project requires FFmpeg for video and audio processing. Install it before proceeding:

**Ubuntu/Debian:**
```
sudo apt update
sudo apt install ffmpeg
```

**macOS (using Homebrew):**
```
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add to your PATH.

### Project Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/remwaste-accent-analyzer.git
   cd remwaste-accent-analyzer
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```
python main.py
```

## Requirements

- Python 3.8+
- FFmpeg (for video processing)
- PyTorch
- Transformers
- MoviePy
- Other dependencies listed in `requirements.txt`

Note: The first time you run accent classification, the model will be downloaded from Hugging Face (approximately 1.2GB).

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
