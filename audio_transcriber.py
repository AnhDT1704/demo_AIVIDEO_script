import os
import requests
import streamlit as st
import traceback
import ffmpeg
import librosa
import soundfile as sf
import numpy as np
import shutil
import time
from typing import Dict, List, Any, Optional
from config import TEMP_DIR, ASSEMBLYAI_API_KEY

class AssemblyAITranscriber:
    """AssemblyAI with Clean, Minimal Output"""

    def __init__(self):
        self.api_key = ASSEMBLYAI_API_KEY
        self.base_url = "https://api.assemblyai.com/v2"
        self.temp_dir = TEMP_DIR
        self.timeline_offset = 0.0

        # Test API connection silently
        try:
            headers = {"authorization": self.api_key}
            response = requests.get(f"{self.base_url}/transcript", headers=headers, timeout=60)
            # Only show error if connection fails
            if response.status_code != 200:
                # st.warning(f"⚠️ AssemblyAI API response: {response.status_code}")
                pass
        except Exception as e:
            st.error(f"❌ AssemblyAI API connection failed: {e}")
            pass

    def extract_audio_with_calibration(self, video_path: str, unique_suffix: str = None) -> Optional[str]:
        """Extract audio with minimal output and unique naming"""
        try:
            # ✅ FIX: Add unique suffix to avoid filename conflicts
            if unique_suffix:
                audio_filename = f"audio_assemblyai_{unique_suffix}_{int(time.time())}.wav"
            else:
                audio_filename = f"audio_assemblyai_{int(time.time())}.wav"
            audio_path = os.path.join(self.temp_dir, audio_filename)

            # Try multiple extraction methods for compatibility
            extraction_success = False

            # Method 1: Accurate seek
            try:
                stream = ffmpeg.input(video_path, accurate_seek=True)
                audio = stream.audio
                out = ffmpeg.output(
                    audio,
                    audio_path,
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16000",
                    avoid_negative_ts="disabled"
                )
                ffmpeg.run(out, overwrite_output=True, quiet=True)
                extraction_success = True
                # No success message - keep quiet
            except Exception as e1:
                # Method 2: Basic extraction
                try:
                    stream = ffmpeg.input(video_path)
                    audio = stream.audio
                    out = ffmpeg.output(
                        audio,
                        audio_path,
                        acodec="pcm_s16le",
                        ac=1,
                        ar="16000"
                    )
                    ffmpeg.run(out, overwrite_output=True, quiet=True)
                    extraction_success = True
                    # No success message - keep quiet
                except Exception as e2:
                    st.error(f"❌ Audio extraction failed: {str(e2)}")
                    return None

            if not extraction_success:
                return None

            # Verify file silently - only warn for serious issues
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                try:
                    # Get durations for validation
                    probe = ffmpeg.probe(audio_path)
                    audio_duration = float(probe['format']['duration'])
                    video_probe = ffmpeg.probe(video_path)
                    video_duration = float(video_probe['format']['duration'])
                    duration_diff = abs(audio_duration - video_duration)

                    # Set timeline offset for calibration
                    if duration_diff > 0.5:
                        # st.warning(f"⚠️ Timeline mismatch detected: {duration_diff:.2f}s")
                        self.timeline_offset = duration_diff
                    else:
                        self.timeline_offset = 0.0

                    return audio_path
                except Exception as probe_error:
                    # Still return file if extraction succeeded
                    self.timeline_offset = 0.0
                    return audio_path
            else:
                st.error("❌ Audio extraction failed - file not created or empty")
                return None

        except Exception as e:
            st.error(f"❌ Audio extraction failed: {str(e)}")
            return None

    def upload_audio_to_assemblyai(self, audio_path: str) -> Optional[str]:
        """Upload audio file to AssemblyAI - silent"""
        try:
            headers = {"authorization": self.api_key}
            with open(audio_path, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    headers=headers,
                    data=f,
                    timeout=300
                )

            if response.status_code == 200:
                upload_url = response.json()["upload_url"]
                # Silent success - no message
                return upload_url
            else:
                st.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            return None

    def transcribe_with_speaker_diarization(self, upload_url: str, speakers_expected: int = None) -> Dict[str, Any]:
        """Transcribe with AssemblyAI - minimal output"""
        try:
            headers = {
                "authorization": self.api_key,
                "content-type": "application/json"
            }

            # Enhanced configuration
            data = {
                "audio_url": upload_url,
                "speaker_labels": True,
                "language_code": "vi",
                "punctuate": True,
                "format_text": True,
                "dual_channel": False,
                "speech_model": "best"
            }

            if speakers_expected and speakers_expected > 0:
                data["speakers_expected"] = speakers_expected

            # Submit transcription request
            response = requests.post(
                f"{self.base_url}/transcript",
                json=data,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                return {"error": f"Transcription request failed: {response.status_code} - {response.text}"}

            transcript_id = response.json()["id"]
            return self._poll_for_completion(transcript_id, headers)

        except Exception as e:
            return {"error": f"AssemblyAI transcription failed: {str(e)}"}

    def _poll_for_completion(self, transcript_id: str, headers: Dict) -> Dict[str, Any]:
        """Poll for completion with minimal progress tracking"""
        try:
            polling_url = f"{self.base_url}/transcript/{transcript_id}"
            max_wait = 600
            start_time = time.time()

            # Minimal progress display
            progress_bar = st.progress(0)
            status_text = st.empty()

            while time.time() - start_time < max_wait:
                response = requests.get(polling_url, headers=headers, timeout=30)
                if response.status_code != 200:
                    return {"error": f"Polling failed: {response.status_code}"}

                result = response.json()
                status = result.get("status")

                # Update progress
                elapsed = time.time() - start_time
                progress = min(elapsed / 120, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Status: {status} | Elapsed: {elapsed:.1f}s")

                if status == "completed":
                    # Clear progress display
                    progress_bar.empty()
                    status_text.empty()
                    return self._parse_assemblyai_result(result)
                elif status == "error":
                    error_msg = result.get("error", "Unknown error")
                    return {"error": f"AssemblyAI error: {error_msg}"}

                time.sleep(3)

            return {"error": "Transcription timeout (10 minutes)"}

        except Exception as e:
            return {"error": f"Polling failed: {str(e)}"}

    def _parse_assemblyai_result(self, result: Dict) -> Dict[str, Any]:
        """Parse AssemblyAI result with minimal output"""
        try:
            # Get main transcript
            full_text = result.get("text", "")

            # Parse utterances (speaker segments)
            utterances = result.get("utterances", [])
            speaker_segments = []

            # Apply timeline offset if needed
            for utterance in utterances:
                original_start = utterance.get("start", 0) / 1000.0
                original_end = utterance.get("end", 0) / 1000.0

                if self.timeline_offset > 0.1:
                    calibrated_start = max(0, original_start - self.timeline_offset)
                    calibrated_end = max(calibrated_start + 0.1, original_end - self.timeline_offset)
                else:
                    calibrated_start = original_start
                    calibrated_end = original_end

                segment = {
                    "speaker": utterance.get("speaker", "Unknown"),
                    "text": utterance.get("text", "").strip(),
                    "start": calibrated_start,
                    "end": calibrated_end,
                    "original_start": original_start,
                    "original_end": original_end,
                    "confidence": utterance.get("confidence", 0.0),
                    "words": utterance.get("words", [])
                }

                if segment["text"]:
                    speaker_segments.append(segment)

            # Parse words for detailed mapping
            words = result.get("words", [])
            word_segments = []
            for word in words:
                original_start = word.get("start", 0) / 1000.0
                original_end = word.get("end", 0) / 1000.0

                if self.timeline_offset > 0.1:
                    calibrated_start = max(0, original_start - self.timeline_offset)
                    calibrated_end = max(calibrated_start + 0.01, original_end - self.timeline_offset)
                else:
                    calibrated_start = original_start
                    calibrated_end = original_end

                word_segments.append({
                    "word": word.get("text", "").strip(),
                    "start_time": calibrated_start,
                    "end_time": calibrated_end,
                    "confidence": word.get("confidence", 0.0),
                    "speaker": word.get("speaker", "Unknown")
                })

            # Get unique speakers
            unique_speakers = list(set([s["speaker"] for s in speaker_segments if s["speaker"] != "Unknown"]))

            return {
                "full_text": full_text,
                "speaker_segments": speaker_segments,
                "words": word_segments,
                "unique_speakers": unique_speakers,
                "language": "vi",
                "timeline_offset": self.timeline_offset,
                "confidence": result.get("confidence", 0.0),
                "audio_duration": result.get("audio_duration", 0.0),
                "error": None
            }

        except Exception as e:
            st.error(f"❌ Result parsing failed: {str(e)}")
            return {"error": f"Result parsing failed: {str(e)}"}

    def map_to_scenes_with_precision(self, assemblyai_result: Dict, scenes: List[Dict]) -> List[Dict]:
        """Strict Timeline Boundary Mapping - Clean Version"""
        try:
            if assemblyai_result.get("error"):
                st.error(f"❌ AssemblyAI error: {assemblyai_result['error']}")
                return [
                    {**scene,
                     "transcript": "",
                     "speaker_transcript": "",
                     "word_count": 0,
                     "unique_speakers": [],
                     "has_speech": False,
                     "transcription_error": assemblyai_result["error"]
                    } for scene in scenes
                ]

            speaker_segments = assemblyai_result.get("speaker_segments", [])
            timeline_offset = assemblyai_result.get("timeline_offset", 0.0)
            scenes_with_text = []

            # Sort segments by start time
            speaker_segments = sorted(speaker_segments, key=lambda x: x["start"])

            for scene_idx, scene in enumerate(scenes):
                scene_start = scene["start_time"]
                scene_end = scene["end_time"]
                scene_duration = scene_end - scene_start

                # Find segments and CUT them at exact scene boundaries
                scene_segments = []
                scene_speakers = set()

                # STRICT: Only 100ms tolerance
                TOLERANCE = 0.1

                for seg_idx, segment in enumerate(speaker_segments):
                    seg_start = segment["start"]
                    seg_end = segment["end"]
                    seg_duration = seg_end - seg_start

                    # Calculate overlap
                    overlap_start = max(seg_start, scene_start)
                    overlap_end = min(seg_end, scene_end)
                    overlap_duration = overlap_end - overlap_start

                    # Only process if there's actual overlap
                    if overlap_duration > 0.05:
                        # Calculate text cutting ratios
                        text_start_ratio = 0.0
                        text_end_ratio = 1.0

                        if seg_start < scene_start:
                            time_before_scene = scene_start - seg_start
                            text_start_ratio = min(0.8, time_before_scene / seg_duration)

                        if seg_end > scene_end:
                            time_after_scene = seg_end - scene_end
                            text_end_ratio = 1.0 - min(0.8, time_after_scene / seg_duration)

                        # Precise text cutting
                        original_text = segment["text"].strip()
                        words = original_text.split()
                        if len(words) > 1:
                            start_word_idx = int(len(words) * text_start_ratio)
                            end_word_idx = int(len(words) * text_end_ratio)
                            start_word_idx = max(0, min(start_word_idx, len(words) - 1))
                            end_word_idx = max(start_word_idx + 1, min(end_word_idx, len(words)))
                            precise_text = " ".join(words[start_word_idx:end_word_idx])
                        else:
                            precise_text = original_text

                        # Calculate precise time boundaries
                        precise_start = max(seg_start, scene_start)
                        precise_end = min(seg_end, scene_end)

                        # Only include if we have meaningful text
                        if precise_text and len(precise_text.strip()) > 2:
                            scene_segment = {
                                **segment,
                                "text": precise_text.strip(),
                                "start": precise_start,
                                "end": precise_end,
                                "original_text": original_text,
                                "text_start_ratio": text_start_ratio,
                                "text_end_ratio": text_end_ratio,
                                "overlap_duration": overlap_duration,
                                "was_cut": text_start_ratio > 0.05 or text_end_ratio < 0.95
                            }

                            scene_segments.append(scene_segment)
                            scene_speakers.add(segment["speaker"])

                # Create final transcript
                if scene_segments:
                    # Sort by precise start time within scene
                    scene_segments.sort(key=lambda x: x["start"])

                    # Build transcript parts
                    full_text_parts = []
                    speaker_lines = []

                    for seg in scene_segments:
                        text = seg["text"].strip()
                        if text:
                            full_text_parts.append(text)
                            speaker_lines.append(f"{seg['speaker']}: {text}")

                    full_text = " ".join(full_text_parts)
                    speaker_transcript = "\n".join(speaker_lines)
                    word_count = len(full_text.split()) if full_text else 0
                    unique_speakers_list = sorted(list(scene_speakers))
                else:
                    # No speech in this scene
                    full_text = ""
                    speaker_transcript = ""
                    word_count = 0
                    unique_speakers_list = []

                # Create scene result
                scene_with_text = {
                    **scene,
                    "transcript": full_text,
                    "speaker_transcript": speaker_transcript,
                    "word_count": word_count,
                    "unique_speakers": unique_speakers_list,
                    "has_speech": bool(full_text.strip()),
                    "language": "vi",
                    "transcription_error": None,
                    "segments": scene_segments,
                    "timeline_offset": timeline_offset,
                    "mapping_quality": "strict_boundary"
                }

                scenes_with_text.append(scene_with_text)

            # Final statistics - minimal output
            total_words = sum(s["word_count"] for s in scenes_with_text)
            scenes_with_speech = sum(1 for s in scenes_with_text if s["has_speech"])
            total_cut_segments = sum(sum(1 for seg in s.get("segments", []) if seg.get("was_cut", False)) for s in scenes_with_text)

            # Only show final result
            # st.success(f"✔ STRICT BOUNDARY MAPPING COMPLETE!")

            return scenes_with_text

        except Exception as e:
            st.error(f"❌ Strict boundary mapping failed: {str(e)}")
            traceback.print_exc()
            return scenes

    def process_complete_video(self, video_path: str, scenes: List[Dict], speakers_expected: int = None) -> List[Dict]:
        """Complete processing pipeline - Clean Version with unique audio naming"""
        try:
            # ✅ FIX: Generate unique suffix from video path and scenes
            video_filename = os.path.basename(video_path).replace('.', '_')
            unique_suffix = f"{video_filename}_{len(scenes)}scenes"
            
            # Audio extraction with unique naming
            audio_path = self.extract_audio_with_calibration(video_path, unique_suffix)
            if not audio_path:
                st.error("❌ Audio extraction failed - cannot proceed")
                return scenes

            # Upload (silent)
            upload_url = self.upload_audio_to_assemblyai(audio_path)
            if not upload_url:
                st.error("❌ Upload failed - cannot proceed")
                self.cleanup_audio_file(audio_path)
                return scenes

            # Transcription
            assemblyai_result = self.transcribe_with_speaker_diarization(upload_url, speakers_expected)
            if assemblyai_result.get("error"):
                st.error(f"❌ Transcription failed: {assemblyai_result['error']}")
                self.cleanup_audio_file(audio_path)
                return scenes

            # Precision mapping
            scenes_with_text = self.map_to_scenes_with_precision(assemblyai_result, scenes)

            # Cleanup (silent)
            self.cleanup_audio_file(audio_path)

            return scenes_with_text

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            traceback.print_exc()
            return scenes

    def get_transcription_stats(self, scenes_with_text: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate transcription statistics"""
        try:
            total_words = sum(s.get("word_count", 0) for s in scenes_with_text)
            scenes_with_speech = sum(1 for s in scenes_with_text if s.get("has_speech", False))
            all_speakers = {sp for s in scenes_with_text for sp in s.get("unique_speakers", [])}
            scenes_multi = sum(1 for s in scenes_with_text if len(s.get("unique_speakers", [])) > 1)

            # Speaker distribution
            speaker_counts = {}
            for scene in scenes_with_text:
                for speaker in scene.get("unique_speakers", []):
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            return {
                "total_words": total_words,
                "scenes_with_speech": scenes_with_speech,
                "scenes_without_speech": len(scenes_with_text) - scenes_with_speech,
                "speech_coverage": (scenes_with_speech / len(scenes_with_text) * 100) if scenes_with_text else 0,
                "total_unique_speakers": len(all_speakers),
                "scenes_with_multiple_speakers": scenes_multi,
                "speaker_distribution": speaker_counts,
                "average_words_per_scene": (total_words / scenes_with_speech) if scenes_with_speech else 0
            }

        except Exception as e:
            st.error(f"❌ Stats calculation failed: {str(e)}")
            return {}

    def cleanup_audio_file(self, audio_path: str):
        """Clean up temporary audio file - silent"""
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                # Silent cleanup - no message
            except Exception as e:
                # st.warning(f"⚠️ Cleanup failed: {e}")
                pass