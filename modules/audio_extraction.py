
import subprocess
import tempfile
import sys
from config import AUDIO_CONFIG

def extract_audio(video_path):
    """
    Extract audio from video using FFmpeg
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Path to extracted audio file (.wav)
    """
    print(f"[1/4] Extracting audio from: {video_path}")
    
    # Create temporary audio file
    audio_path = tempfile.mktemp(suffix='.wav')
    
    # FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', AUDIO_CONFIG['codec'],
        '-ar', str(AUDIO_CONFIG['sample_rate']),
        '-ac', str(AUDIO_CONFIG['channels']),
        '-y',  # Overwrite output
        audio_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"    Audio extracted successfully")
        return audio_path
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FFmpeg failed")
        print(f"Details: {e.stderr.decode()}")
        sys.exit(1)