"""
Utility functions for the pipeline
"""

def format_timestamp(milliseconds):
    """
    Convert milliseconds to HH:MM:SS,mmm format
    
    Args:
        milliseconds: Time in milliseconds
        
    Returns:
        Formatted timestamp string (e.g., "00:01:23,456")
    """
    total_seconds = milliseconds / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int(milliseconds % 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def validate_file(file_path):
    """
    Check if file exists and has valid extension
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If extension not supported
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wav', '.mp3']
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in valid_extensions:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return True