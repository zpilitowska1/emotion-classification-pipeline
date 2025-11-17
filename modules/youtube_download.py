"""
YouTube audio download - Direct upload to AssemblyAI (no conversion needed)
"""

from pytubefix import YouTube
import os
import sys
import tempfile

def is_youtube_url(url):
    """Check if string is YouTube URL"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
    return any(domain in url.lower() for domain in youtube_domains)

def download_youtube_audio(url, output_path=None):
    """
    Download audio from YouTube
    AssemblyAI accepts m4a/mp4/webm directly - NO conversion needed!
    
    Args:
        url: YouTube URL
        output_path: Optional output path
        
    Returns:
        Path to downloaded audio file
    """
    print(f"[0/4] Downloading audio from YouTube...")
    print(f"    URL: {url}")
    
    try:
        # Create YouTube object
        yt = YouTube(url)
        
        print(f"    Title: {yt.title}")
        print(f"    Duration: {yt.length}s")
        print(f"    Downloading audio stream...")
        
        # Get best audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if audio_stream is None:
            raise ValueError("No audio stream available for this video")
        
        # Set destination
        if output_path is None:
            destination = tempfile.mkdtemp()
        else:
            destination = os.path.dirname(output_path)
            if not destination:
                destination = os.getcwd()
        
        os.makedirs(destination, exist_ok=True)
        
        # Download audio
        out_file = audio_stream.download(output_path=destination)
        
        print(f"    Audio downloaded: {os.path.basename(out_file)}")
        print(f"    Format: {os.path.splitext(out_file)[1]}")
        print(f"    Size: {os.path.getsize(out_file) / (1024*1024):.1f} MB")
        print(f"    AssemblyAI will process this format directly")
        
        return out_file
        
    except Exception as e:
        print(f"ERROR: Failed to download YouTube audio")
        print(f"Reason: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check URL is correct and video is public")
        print(f"  2. Check internet connection")
        print(f"  3. Try: pip install --upgrade pytubefix")
        sys.exit(1)