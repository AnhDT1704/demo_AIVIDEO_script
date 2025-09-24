import os
import tempfile
import streamlit as st
import ffmpeg
import uuid
import logging  # Add missing import
from typing import Dict, Any, Optional
from config import MAX_FILE_SIZE, MIN_DURATION, MAX_DURATION, SUPPORTED_FORMATS, TEMP_DIR, UPLOAD_DIR

class VideoProcessor:
    """Handle video ingestion, validation and basic processing with FFmpeg"""

    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.upload_dir = UPLOAD_DIR

    def validate_video(self, uploaded_file, skip_duration_check=True):
        """
        Validate uploaded video file with optional duration check skip
        Default: Accept any duration (skip_duration_check=True)
        """
        try:
            # Save uploaded file to temp location
            temp_path = self._save_uploaded_file(uploaded_file)

            # Debug: check if file exists
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temp file not found after saving: {temp_path}")

            # Get video metadata using FFmpeg
            metadata = self._get_video_metadata(temp_path)

            # Validate constraints
            validation_result = self._validate_constraints(metadata, uploaded_file, skip_duration_check)

            # If valid and has audio, extract optimized audio for diarization
            if validation_result['valid'] and metadata.get('has_audio', False):
                diar_audio = self.extract_audio_for_diarization(temp_path)
                validation_result['diarization_audio'] = diar_audio
            else:
                validation_result['diarization_audio'] = None

            # Add file path to result
            validation_result['temp_path'] = temp_path
            validation_result['metadata'] = metadata
            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': f"Video validation failed: {str(e)}",
                'temp_path': None,
                'metadata': None,
                'diarization_audio': None
            }

    def extract_audio_for_diarization(self, video_path: str) -> Optional[str]:
        """
        Extract audio optimized cho Picovoice Falcon:
          - PCM 16-bit
          - Mono
          - 16kHz
        Tráº£ vá» Ä‘Æ°á»ng dáº«n WAV hoáº·c None náº¿u lá»—i.
        """
        try:
            temp_id = str(uuid.uuid4())[:8]
            audio_path = os.path.join(self.temp_dir, f"diarization_audio_{temp_id}.wav")

            stream = ffmpeg.input(video_path)
            audio = ffmpeg.output(
                stream, audio_path,
                vn=None,                    # no video
                acodec='pcm_s16le',         # 16-bit PCM
                ar='16000',                 # 16 kHz
                ac=1,                       # mono
                af='highpass=f=80,lowpass=f=8000'  # lá»c bÄƒng thÃ´ng
            )
            ffmpeg.run(audio, overwrite_output=True, quiet=True)

            if not os.path.exists(audio_path):
                raise RuntimeError("Audio extraction for diarization failed")

            return audio_path

        except Exception as e:
            st.error(f"âŒ Diarization audio extraction failed: {e}")
            return None

    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary directory"""
        os.makedirs(self.temp_dir, exist_ok=True)
        import time
        temp_filename = f"temp_{int(time.time())}_{uploaded_file.name}"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        # st.write(f"Saving to: {temp_path}")  # Hidden debug message
        # st.write(f"Temp dir exists: {os.path.exists(self.temp_dir)}")  # Hidden debug message
        # st.write(f"Full temp path: {os.path.abspath(temp_path)}")  # Hidden debug message

        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # st.write(f"File saved successfully: {os.path.exists(temp_path)}")  # Hidden debug message
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            raise

        return temp_path

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using FFmpeg probe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

            if not video_stream:
                raise ValueError("No video stream found")

            metadata = {
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'has_audio': audio_stream is not None,
                'audio_codec': audio_stream.get('codec_name', 'none') if audio_stream else None,
                'bitrate': int(probe['format'].get('bit_rate', 0))
            }
            return metadata

        except Exception as e:
            raise ValueError(f"Failed to extract video metadata: {str(e)}")

    def _validate_constraints(self, metadata: Dict[str, Any], uploaded_file, skip_duration_check=False) -> Dict[str, Any]:
        """Validate video against defined constraints"""
        errors = []
        warnings = []

        # File size validation
        if uploaded_file.size > MAX_FILE_SIZE:
            errors.append(f"File size ({uploaded_file.size/(1024*1024):.1f}MB) exceeds maximum ({MAX_FILE_SIZE/(1024*1024):.0f}MB)")

        # Duration validation
        duration = metadata.get('duration', 0)
        if not skip_duration_check and duration < MIN_DURATION:
            errors.append(f"Video duration ({duration:.1f}s) is too short (minimum: {MIN_DURATION}s)")
        elif skip_duration_check and duration < MIN_DURATION:
            warnings.append(f"Short video detected ({duration:.1f}s) - processing anyway")
            # Always log duration for reference
            logging.info(f"Video duration: {duration:.1f}s (skip_duration_check: {skip_duration_check})")
        elif duration > MAX_DURATION:  # Fix: Remove duplicate elif, combine with proper structure
            errors.append(f"Video duration ({duration:.1f}s) is too long (maximum: {MAX_DURATION}s)")

        # Format validation
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            errors.append(f"Unsupported format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}")

        # Resolution warnings
        width, height = metadata.get('width', 0), metadata.get('height', 0)
        if width < 640 or height < 480:
            warnings.append(f"Low resolution ({width}x{height}) may affect analysis quality")

        # Audio check
        if not metadata.get('has_audio', False):
            warnings.append("No audio track found - transcription will be skipped")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'duration': duration,
            'resolution': f"{width}x{height}",
            'has_audio': metadata.get('has_audio', False)
        }

    def cleanup_temp_file(self, temp_path: Optional[str]):
        """Clean up temporary file"""
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                st.warning(f"Failed to cleanup temp file: {str(e)}")