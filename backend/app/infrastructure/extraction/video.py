"""
Video Processing for FactuAI.

Extract text from videos using:
1. ffmpeg for audio extraction
2. speech_recognition for speech-to-text
"""

import os
import subprocess
import tempfile

import speech_recognition as sr

from utils.logging import get_logger

logger = get_logger(__name__)


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file (WAV)
        
    Returns:
        True if successful, False otherwise
    """
    logger.debug(f"[VIDEO] Extracting audio: {video_path} -> {audio_path}")
    
    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            "-loglevel",
            "error",
            audio_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"[VIDEO] FFmpeg failed: {result.stderr}")
            return False
        
        logger.debug("[VIDEO] Audio extraction successful")
        return True
        
    except Exception as e:
        logger.error(f"[VIDEO] Audio extraction failed: {e}")
        return False


def audio_to_text(audio_path: str) -> str:
    """
    Convert audio file to text using speech recognition.
    
    Args:
        audio_path: Path to audio file (WAV)
        
    Returns:
        Transcribed text string
    """
    logger.debug(f"[VIDEO] Converting audio to text: {audio_path}")
    
    try:
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio, language="en-US")
        logger.debug(f"[VIDEO] Transcribed {len(text)} characters")
        return text.strip()
        
    except sr.UnknownValueError:
        logger.warning("[VIDEO] Could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"[VIDEO] Speech recognition service error: {e}")
        return ""
    except Exception as e:
        logger.error(f"[VIDEO] Audio to text failed: {e}")
        return ""


def extract_text_from_video(video_path: str, audio_path: str | None = None) -> str:
    """
    Extract text from video by converting speech to text.
    
    Args:
        video_path: Path to video file
        audio_path: Optional path for temporary audio file
                   If None, creates a temporary file
        
    Returns:
        Extracted text string
    """
    logger.info(f"[VIDEO] Processing video: {video_path}")
    
    cleanup_audio = False
    
    if audio_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_path = tmp.name
        cleanup_audio = True
    
    try:
        if not extract_audio_from_video(video_path, audio_path):
            return ""
        
        text = audio_to_text(audio_path)
        
        return text
        
    finally:
        if cleanup_audio and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"[VIDEO] Failed to cleanup temp audio file: {e}")


__all__ = [
    "extract_audio_from_video",
    "audio_to_text",
    "extract_text_from_video",
]
