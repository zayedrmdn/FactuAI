"""
Video-to-text endpoint.

POST /api/extract-video-text
Form-data field name: video (MP4/AVI/MOV/...)
Returns: { "text": "...extracted text..." }
"""

from flask import Blueprint, request, jsonify
from core.logging import logger
import tempfile
import os
import subprocess
import speech_recognition as sr

bp_video = Blueprint("video", __name__, url_prefix="/api")

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using ffmpeg"""
    try:
        # Use ffmpeg to extract audio as WAV
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            '-loglevel', 'error',  # Reduce ffmpeg output
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return False

def audio_to_text(audio_path: str) -> str:
    """Convert audio to text using speech recognition"""
    try:
        recognizer = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source)
        
        # Convert to text using Google Speech Recognition
        text = recognizer.recognize_google(audio, language='en-US')
        return text.strip()
        
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Audio to text conversion failed: {e}")
        return ""

@bp_video.post("/extract-video-text")
def extract_video_text():
    """Extract text from video by converting speech to text"""
    video_path = None
    audio_path = None
    
    try:
        # Handle file upload only
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Create temporary file for uploaded video
        video_ext = os.path.splitext(file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp_video:
            video_path = tmp_video.name
            file.save(video_path)
            logger.debug(f"Saved uploaded video to: {video_path}")
        
        # Extract audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            audio_path = tmp_audio.name
        
        if not extract_audio_from_video(video_path, audio_path):
            return jsonify({"error": "Failed to extract audio from video"}), 500
        
        # Convert audio to text
        text = audio_to_text(audio_path)
        
        if not text:
            return jsonify({"error": "No speech found in video"}), 400
        
        logger.debug(f"Extracted {len(text)} characters from video")
        return jsonify({"text": text})
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
        
    finally:
        # Clean up temp files
        for path in [video_path, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Cleaned up: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {path}: {e}")
