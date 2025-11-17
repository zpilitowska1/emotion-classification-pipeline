#!/usr/bin/env python3
"""
Main Pipeline - With Timing and Forced Output Display
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import time  # ADD THIS
from datetime import timedelta  # ADD THIS


sys.stdout.reconfigure(line_buffering=True)


import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from modules import (
    download_youtube_audio,
    is_youtube_url,
    extract_audio,
    transcribe_audio,
    translate_segments,
    classify_emotions,
    format_timestamp,
    validate_file
)

def format_duration(seconds):
    """Format seconds into readable duration"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def process_video(input_path, output_path, assembly_api_key, keep_audio=False):
    """Process video through pipeline with timing"""
    
    # TIMING: Start overall timer
    pipeline_start = time.time()
    timings = {}
    
    print(f"\n{'='*70}", flush=True)
    print(f"POLISH EMOTION PIPELINE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Input:  {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    try:
        # STEP 1: Audio extraction
        step_start = time.time()
        audio_path = None
        temp_audio = None
        
        if is_youtube_url(input_path):
            print("Detected YouTube URL", flush=True)
            temp_audio = download_youtube_audio(input_path)
            audio_path = temp_audio
        else:
            print("Detected local file", flush=True)
            validate_file(input_path)
            audio_path = extract_audio(input_path)
            temp_audio = audio_path
        
        timings['audio_extraction'] = time.time() - step_start
        print(f"Audio ready: {audio_path} ({format_duration(timings['audio_extraction'])})", flush=True)
        
        # STEP 2: Transcription
        step_start = time.time()
        print("\nStarting transcription...", flush=True)
        segments = transcribe_audio(audio_path, assembly_api_key)
        timings['transcription'] = time.time() - step_start
        print(f"Got {len(segments)} segments ({format_duration(timings['transcription'])})", flush=True)
        
        # STEP 3: Translation
        step_start = time.time()
        print("\nStarting translation...", flush=True)
        translations = translate_segments(segments)
        timings['translation'] = time.time() - step_start
        print(f"Translated {len(translations)} segments ({format_duration(timings['translation'])})", flush=True)
        
        # STEP 4: Emotion Classification
        step_start = time.time()
        print("\nStarting emotion classification...", flush=True)
        polish_texts = [seg['text'] for seg in segments]
        emotions = classify_emotions(polish_texts)
        timings['emotion_classification'] = time.time() - step_start
        print(f"Classified {len(emotions)} segments ({format_duration(timings['emotion_classification'])})", flush=True)
        
        # STEP 5: Combine and save
        step_start = time.time()
        print("\nCombining results...", flush=True)
        results = []
        for i, seg in enumerate(segments):
            results.append({
                'Start Time': format_timestamp(seg['start']),
                'End Time': format_timestamp(seg['end']),
                'Sentence': seg['text'].strip(),
                'Translation': translations[i].strip(),
                'Emotion': emotions[i]
            })
        
        df = pd.DataFrame(results)
        
        # Save
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        timings['output_generation'] = time.time() - step_start
        
        print(f"Saved to: {output_path} ({format_duration(timings['output_generation'])})", flush=True)
        
        # Cleanup
        if temp_audio and not keep_audio:
            try:
                os.remove(temp_audio)
            except:
                pass
        
        # TIMING: Calculate total
        timings['total'] = time.time() - pipeline_start
        
        # Summary with timing breakdown
        print(f"\n{'='*70}", flush=True)
        print(f"COMPLETED", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Segments processed: {len(results)}", flush=True)
        
        print(f"\n{'='*70}", flush=True)
        print(f"TIMING BREAKDOWN", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Audio Extraction:        {format_duration(timings['audio_extraction']):>12s}", flush=True)
        print(f"Transcription:           {format_duration(timings['transcription']):>12s}", flush=True)
        print(f"Translation:             {format_duration(timings['translation']):>12s}", flush=True)
        print(f"Emotion Classification:  {format_duration(timings['emotion_classification']):>12s}", flush=True)
        print(f"Output Generation:       {format_duration(timings['output_generation']):>12s}", flush=True)
        print(f"{'-'*70}", flush=True)
        print(f"TOTAL PIPELINE TIME:     {format_duration(timings['total']):>12s}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Performance stats
        print(f"\nPERFORMANCE STATISTICS:", flush=True)
        print(f"  Segments/second: {len(results) / timings['total']:.2f}", flush=True)
        print(f"  Average time per segment: {timings['total'] / len(results):.2f}s", flush=True)
        
        print(f"\n{'='*70}", flush=True)
        print(f"EMOTION DISTRIBUTION", flush=True)
        print(f"{'='*70}", flush=True)
        for emotion, count in df['Emotion'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"  {emotion:12s}: {count:3d} ({percentage:5.1f}%)", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        return df, timings
        
    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Polish Emotion Pipeline')
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--keep-audio', action='store_true')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('ASSEMBLYAI_API_KEY')
    
    if not api_key:
        print("ERROR: API key required!", flush=True)
        print("Use: --api-key YOUR_KEY", flush=True)
        sys.exit(1)
    
    print(f"Starting pipeline...", flush=True)
    df, timings = process_video(args.input, args.output, api_key, args.keep_audio)

if __name__ == "__main__":
    main()