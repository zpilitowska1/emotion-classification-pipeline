# Polish Emotion Classification Pipeline

An end-to-end NLP pipeline for automatic emotion detection in Polish video content. This pipeline processes Polish video/audio content (from YouTube or local files) and generates timestamped emotion annotations. It's designed for analyzing conversational content like TV shows, interviews, or podcasts.

**Pipeline Flow:**
```
Video/Audio → Audio Extract → Transcription → Translation → Emotion Classification → CSV Output
   (PL)         (FFmpeg)      (AssemblyAI)    (Marian MT)    (XLM-RoBERTa)          (Results)
```

##  Features

- **Multi-source Input**: Process YouTube URLs or local video/audio files
- **Accurate Transcription**: Polish speech-to-text via AssemblyAI
- **Neural Translation**: Helsinki-NLP Marian MT model (Polish → English)
- **7-Class Emotion Detection**: happiness, sadness, anger, surprise, fear, disgust, neutral
- **Timestamped Output**: CSV with start/end times, original text, translation, emotions
- **Batch Processing**: Efficient processing of long-form content with configurable batch sizes
- **GPU Support**: Automatic GPU detection for faster inference
- **Progress Tracking**: Real-time progress bars for each pipeline stage

## Requirements

- Python 3.8+
- AssemblyAI API key (for transcription)
- FFmpeg (for audio extraction)
- CUDA-capable GPU (optional)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/zpilitowska1/emotion-classification-pipeline.git
cd emotion-classification-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### 4. Set up API key

Create a `.env` file in the project root:
```bash
ASSEMBLYAI_API_KEY='your_api_key_here'
```

Get a free API key at [AssemblyAI](https://www.assemblyai.com/)

### 5. Prepare the emotion model

Link to the model: 
https://edubuas-my.sharepoint.com/:u:/g/personal/245205_buas_nl/EYiwjQMP0oNNu01EFNqbA_IBPxpkzzWyrgQFNvBeOjP6hg?e=62qW3i 

Ensure trained emotion model is placed at:
```
models/polish_emotion_model.h5
```


##  Usage

### Basic Usage

### Options

```bash
python pipeline.py --input "https://youtu.be/OQJklpdGgk8?si=ZeD2_B1_AQse0JFt" --output test.csv --api-key your_api_key
```

**Process a YouTube video:**
```bash
python pipeline.py --input "https://youtube.com/watch?v=VIDEO_ID" --output results.csv
```

**Process a local file:**
```bash
python pipeline.py --input path/to/video.mp4 --output results.csv
```



**Arguments:**
- `--input, -i`: YouTube URL or local file path (required)
- `--output, -o`: Output CSV file path (required)
- `--api-key`: AssemblyAI API key 
- `--keep-audio`: Keep extracted audio file after processing

### Supported Input Formats

- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`
- **Audio**: `.wav`, `.mp3`, `.m4a`, `.webm`

###  Output

Output will be saved in test.csv 

Example output file: https://github.com/zpilitowska1/emotion-classification-pipeline/blob/main/test.csv

```csv
Start Time,End Time,Sentence,Translation,Emotion
00:00:00,00:00:03,To jest MasterChef.,This is MasterChef.,neutral
00:00:03,00:00:08,Szansę na tytuł najlepszego kucharza...,Only 12 people have a chance...,happiness
00:00:12,00:00:15,Jestem dinozaurem który chce walczyć.,I'm a dinosaur who wants to fight.,anger,happiness
```

##  Project Structure

```
emotion-classification-pipeline/
├── pipeline.py                 # Main execution script
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── README.md                   # This file
│
├── modules/                    # Pipeline modules
│   ├── __init__.py            # Package initialization
│   ├── youtube_download.py    # YouTube audio download
│   ├── audio_extraction.py    # FFmpeg audio extraction
│   ├── transcription.py       # AssemblyAI integration
│   ├── translation.py         # Marian MT translation
│   ├── emotion_classification.py  # Emotion classification
│   └── utils.py               # Helper functions
│
├── models/                     # Model files
│   ├── polish_emotion_model.h5    # Trained emotion classifier
│   ├── model_card.md          # Model documentation
│   └── model_config.json      # Model configuration
│
└── notebooks/                  # Development notebooks
    └── machine_translation_best_model.ipynb
```

##  Models & Architecture

### 1. Audio Processing
- **Tool**: FFmpeg
- **Output**: 16kHz mono WAV
- **Codec**: PCM 16-bit

### 2. Transcription
- **Service**: AssemblyAI
- **Language**: Polish (`pl`)
- **Features**: 
  - Automatic punctuation
  - Text formatting
  - Word-level timestamps
  - Sentence segmentation (max 15 words or punctuation-based)

### 3. Translation
- **Model**: [Helsinki-NLP/opus-mt-pl-en](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en)
- **Architecture**: Marian Neural Machine Translation
- **Training Data**: OPUS parallel corpus
- **Performance**: State-of-the-art for Polish→English
- **Batch Size**: 16 (configurable)
- **Max Length**: 512 tokens

### 4. Emotion Classification
- **Base Model**: XLM-RoBERTa
- **Input**: Polish text
- **Output**: 7 emotion classes (multi-label)
- **Architecture**: Custom transformer-based classifier
- **Loading**: CPU-compatible unpickler for cross-device compatibility

**Emotion Labels & Thresholds:**
```python
{
    'happiness': 0.6,
    'sadness': 0.7,
    'anger': 0.6,
    'surprise': 0.7,
    'fear': 0.8,
    'disgust': 0.5,
    'neutral': 0.5
}
```

**Multi-label Output**: Segments can have multiple emotions (e.g., `anger,happiness`)

## Optional Configuration

Edit `config.py` to customize pipeline behavior:

### Audio Settings
```python
AUDIO_CONFIG = {
    'sample_rate': 16000,      # Hz
    'channels': 1,             # Mono
    'codec': 'pcm_s16le'      # 16-bit PCM
}
```

### Transcription Settings
```python
TRANSCRIPTION_CONFIG = {
    'language_code': 'pl',
    'speaker_labels': False,
    'punctuate': True,
    'format_text': True
}
```

### Translation Settings
```python
TRANSLATION_CONFIG = {
    'model': 'Helsinki-NLP/opus-mt-pl-en',
    'batch_size': 16,
    'max_length': 512
}
```

### Emotion Classification Settings
```python
EMOTION_CONFIG = {
    'model_path': 'models/polish_emotion_model.h5',
    'batch_size': 16,
    'max_length': 512,
    'thresholds': {...}  # Adjust per emotion
}
```

##  Use Case: MasterChef Analysis

This pipeline was developed to analyze emotional dynamics in Polish MasterChef episodes. The system helps:

- **Track Emotional Arcs**: Monitor how emotions evolve throughout episodes
- **Identify High-Tension Moments**: Detect peaks in anger, fear, or surprise
- **Episode Segmentation**: Use emotion shifts to identify scene changes


##  Known Limitations

### Translation Quality

1. **Speech-to-Text Errors**: 

Most translation mistakes stem from incorrect Polish transcriptions rather than translation errors. The Marian MT model translates what it receives accurately, but if the transcription is wrong, the translation will be wrong.

2. **Word Sense Disambiguation**: 

Polish words with multiple meanings may be translated incorrectly:
   - `kropka` → "dot" or "period" (context-dependent)
   - Conversational idioms may not translate literally

3. **Colloquialisms**: 

Informal speech, slang, and TV-specific jargon may not translate perfectly

4. **Grammar**: 

Translation grammar is generally correct and handles Polish inflection well

### Emotion Classification

1. **Context Dependency**: Short sentences may lack context for accurate emotion detection
2. **Multi-label Output**: Some segments receive multiple emotion labels, which may need interpretation
3. **Domain Specificity**: Model trained on general Polish text may not capture TV-specific emotional expressions perfectly


### Recommendations

- **Review Transcription Quality**: Validate a sample before processing large datasets
- **Post-Processing**: Consider domain-specific vocabulary corrections
- **Multi-label Handling**: Decide how to handle segments with multiple emotions for your analysis
- **Batch Processing**: For multiple videos, process overnight or use GPU acceleration

##  Troubleshooting

### Installation Issues

**"ERROR: API key required!"**
```bash
# Check .env file exists and is formatted correctly
# Should show be like: ASSEMBLYAI_API_KEY='your_key_here'

# Or pass key directly
python pipeline.py --input video.mp4 --output results.csv --api-key YOUR_KEY
```

**"FFmpeg not found"**
```bash
# Test FFmpeg installation
ffmpeg -version

# macOS
brew install ffmpeg

# Add to PATH on Windows
setx PATH "%PATH%;C:\path\to\ffmpeg\bin"
```

### Model Issues

**"Model file not found"**
```bash
# Check model exists
ls models/polish_emotion_model.h5

# Verify file permissions
chmod 644 models/polish_emotion_model.h5
```

### Runtime Issues

**Out of memory errors**
```python
# In config.py, reduce batch sizes:
TRANSLATION_CONFIG = {
    'batch_size': 8,  # Reduced from 16
    ...
}

EMOTION_CONFIG = {
    'batch_size': 8,  # Reduced from 16
    ...
}
```

**YouTube download fails**
```bash
# Update pytubefix
pip install --upgrade pytubefix

# Try alternative URL format
# Instead of: youtube.com/watch?v=VIDEO_ID
# Try: youtu.be/VIDEO_ID
```
