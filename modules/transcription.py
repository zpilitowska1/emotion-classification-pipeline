"""
Speech-to-text transcription using AssemblyAI
"""

import assemblyai as aai
import time
import sys
from config import TRANSCRIPTION_CONFIG

def transcribe_audio(audio_path, api_key):
    """
    Transcribe audio to Polish text using AssemblyAI
    
    Args:
        audio_path: Path to audio file (.wav)
        api_key: AssemblyAI API key
        
    Returns:
        List of segments, each with:
            - 'start': start time in milliseconds
            - 'end': end time in milliseconds  
            - 'text': transcribed Polish text
    """
    print(f"[2/4] Transcribing with AssemblyAI...")
    
    # Set API key
    aai.settings.api_key = api_key
    
    # Configure transcription
    config = aai.TranscriptionConfig(
        language_code=TRANSCRIPTION_CONFIG['language_code'],
        speaker_labels=TRANSCRIPTION_CONFIG.get('speaker_labels', False),
        punctuate=TRANSCRIPTION_CONFIG.get('punctuate', True),
        format_text=TRANSCRIPTION_CONFIG.get('format_text', True)
    )
    
    # Create transcriber
    transcriber = aai.Transcriber(config=config)
    
    print(f"    Uploading audio file...")
    
    # Start transcription
    transcript = transcriber.transcribe(audio_path)
    
    # Poll for completion
    print(f"    Processing transcription...")
    while transcript.status not in [aai.TranscriptStatus.completed, aai.TranscriptStatus.error]:
        time.sleep(3)
        transcript = aai.Transcript.get_by_id(transcript.id)
    
    # Check for errors
    if transcript.status == aai.TranscriptStatus.error:
        print(f"ERROR: Transcription failed")
        print(f"Reason: {transcript.error}")
        sys.exit(1)
    
    print(f"    Transcription completed")
    
    # Build segments from word-level timestamps
    segments = []
    
    if transcript.words:
        current_sentence = []
        current_start = None
        
        for word in transcript.words:
            if current_start is None:
                current_start = word.start
            
            current_sentence.append(word.text)
            
            # End sentence on punctuation or max words
            if word.text.endswith(('.', '!', '?')) or len(current_sentence) >= 15:
                segments.append({
                    'start': current_start,
                    'end': word.end,
                    'text': ' '.join(current_sentence)
                })
                current_sentence = []
                current_start = None
        
        # Add remaining words as final segment
        if current_sentence:
            segments.append({
                'start': current_start,
                'end': transcript.words[-1].end,
                'text': ' '.join(current_sentence)
            })
    else:
        # Fallback: single segment with full text
        segments = [{
            'start': 0,
            'end': 0,
            'text': transcript.text
        }]
    
    print(f"    Created {len(segments)} text segments")
    
    return segments