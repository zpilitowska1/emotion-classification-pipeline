"""
Configuration settings for the emotion classification pipeline
"""

# Audio extraction settings
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'channels': 1,
    'codec': 'pcm_s16le'
}

# AssemblyAI transcription settings
TRANSCRIPTION_CONFIG = {
    'language_code': 'pl',
    'speaker_labels': False,
    'punctuate': True,
    'format_text': True
}

# Marian MT translation settings
TRANSLATION_CONFIG = {
    'model': 'Helsinki-NLP/opus-mt-pl-en',
    'batch_size': 16,
    'max_length': 512
}

# Emotion classification settings
EMOTION_CONFIG = {
    'model_path': 'models/polish_emotion_model.h5',  # Note: models/ folder
    'batch_size': 16,
    'max_length': 512,
    'thresholds': {
        'neutral': 0.5,
        'disgust': 0.5,
        'happiness': 0.6,
        'anger': 0.6,
        'surprise': 0.7,
        'sadness': 0.7,
        'fear': 0.8
    }
}

# Emotion labels
EMOTION_LABELS = ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral']