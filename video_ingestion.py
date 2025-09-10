import os
import tempfile
import streamlit as st
import ffmpeg
from typing import Dict, Any, Optional
from config import (
    MAX_FILE_SIZE, MIN_DURATION, MAX_DURATION,
    SUPPORTED_FORMATS, TEMP_DIR, UPLOAD_DIR
)

class VideoProcessor:
    """Handle video ingestion, validation and basic processing with FFmpeg"""

    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.upload_dir = UPLOAD_DIR

    def validate_video(self, uploaded_file) -> Dict[str, Any]:
        """
        Validate uploaded video file using FFmpeg
        Args:
            uploaded_file: Streamlit uploaded file object
        Returns:
            Dict with validation results and video metadata
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
            validation_result = self._validate_constraints(metadata, uploaded_file)

            # Add file path to result
            validation_result['temp_path'] = temp_path
            validation_result['metadata'] = metadata

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': f"Video validation failed: {str(e)}",
                'temp_path': None,
                'metadata': None
            }

    def validate_video_by_path(self, video_path: str) -> Dict[str, Any]:
        """
        Validate video file by path (for API use)
        Args:
            video_path: Path to video file on disk
        Returns:
            Dict with validation results
        """
        try:
            # Get video metadata using FFmpeg
            metadata = self._get_video_metadata(video_path)
            
            # Create mock uploaded file object for validation
            class MockUploadedFile:
                def __init__(self, size, name):
                    self.size = size
                    self.name = name
            
            mock_file = MockUploadedFile(metadata['size'], os.path.basename(video_path))
            validation_result = self._validate_constraints(metadata, mock_file)
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'duration': 0,
                'resolution': '0x0',
                'has_audio': False
            }

    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary directory"""
        # Create temp directory if not exists
        os.makedirs(self.temp_dir, exist_ok=True)

        # Generate unique filename with timestamp
        import time
        temp_filename = f"temp_{int(time.time())}_{uploaded_file.name}"
        temp_path = os.path.join(self.temp_dir, temp_filename)

        # Debug info
        st.write(f"Saving to: {temp_path}")
        st.write(f"Temp dir exists: {os.path.exists(self.temp_dir)}")
        st.write(f"Full temp path: {os.path.abspath(temp_path)}")

        # Write file content
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"File saved successfully: {os.path.exists(temp_path)}")
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            raise

        return temp_path

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using FFmpeg probe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )

            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )

            if not video_stream:
                raise ValueError("No video stream found")

            # Extract key metadata
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

    def _validate_constraints(self, metadata: Dict[str, Any], uploaded_file) -> Dict[str, Any]:
        """Validate video against defined constraints"""
        errors = []
        warnings = []

        # File size validation
        if uploaded_file.size > MAX_FILE_SIZE:
            errors.append(f"File size ({uploaded_file.size / (1024*1024):.1f}MB) exceeds maximum ({MAX_FILE_SIZE / (1024*1024):.0f}MB)")

        # Duration validation
        duration = metadata.get('duration', 0)
        if duration < MIN_DURATION:
            errors.append(f"Video duration ({duration:.1f}s) is too short (minimum: {MIN_DURATION}s)")
        elif duration > MAX_DURATION:
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

    def normalize_video(self, input_path: str, output_path: str) -> bool:
        """
        Normalize video format/codec using FFmpeg for consistent processing
        Args:
            input_path: Path to input video
            output_path: Path to output normalized video
        Returns:
            bool: Success status
        """
        try:
            # Normalize to MP4 with H264/AAC for consistency
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                vcodec='libx264',
                acodec='aac',
                preset='fast', # Balance between speed and compression
                crf=23, # Good quality
                movflags='+faststart' # Web optimization
            )

            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return True

        except Exception as e:
            st.error(f"Video normalization failed: {str(e)}")
            return False
