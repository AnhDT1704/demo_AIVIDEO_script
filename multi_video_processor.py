import streamlit as st
import os
from typing import Dict, List, Any, Optional
from video_ingestion import VideoProcessor
from scene_detector import SceneDetector
from audio_transcriber import AssemblyAITranscriber
from gemini_frame_analyzer import GeminiFrameAnalyzer

class MultiVideoProcessor:
    """🎬 Helper class for processing multiple videos efficiently"""

    def __init__(self):
        """Initialize processors"""
        self.video_processor = VideoProcessor()
        self.scene_detector = None  # Will be initialized with threshold
        self.transcriber = AssemblyAITranscriber()
        self.frame_analyzer = GeminiFrameAnalyzer()

    def process_all_videos(self, uploaded_files: List, settings: Dict) -> Dict[str, Dict]:
        """
        🚀 Process multiple videos through stages 1-4

        Args:
            uploaded_files: List of uploaded video files
            settings: Dict with processing settings (scene_threshold, etc.)

        Returns:
            Dict[filename] = {
                'validation': validation_result,
                'scenes': scenes_list,
                'transcription': transcription_result,
                'visual_analysis': visual_analysis_list,
                'status': 'completed'/'failed',
                'error': error_message if failed
            }
        """
        results = {}
        total_files = len(uploaded_files)

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            progress_percentage = (i / total_files)

            status_text.text(f"Processing {filename} ({i+1}/{total_files})...")
            progress_bar.progress(progress_percentage)

            try:
                video_result = self._process_single_video(
                    uploaded_file, filename, settings
                )
                results[filename] = video_result

            except Exception as e:
                st.error(f"❌ Failed to process {filename}: {str(e)}")
                results[filename] = {
                    'validation': None,
                    'scenes': None,
                    'transcription': None,
                    'visual_analysis': None,
                    'status': 'failed',
                    'error': str(e)
                }

            # Update progress
            progress_bar.progress((i + 1) / total_files)

        status_text.text("✅ All videos processed!")
        progress_bar.progress(1.0)

        return results

    def _process_single_video(self, uploaded_file, filename: str, settings: Dict) -> Dict:
        """Process a single video through all stages"""

        with st.expander(f"🎬 Processing {filename}", expanded=False):

            # STAGE 1: Video Validation
            with st.spinner(f"Stage 1: Validating {filename}..."):
                validation_result = self.video_processor.validate_video(uploaded_file)

                if not validation_result['valid']:
                    st.error(f"❌ Video validation failed: {validation_result.get('error', 'Unknown error')}")
                    raise Exception(f"Video validation failed: {validation_result.get('error')}")

                st.success("✅ Video validation passed!")

            # STAGE 2: Scene Detection
            with st.spinner(f"Stage 2: Detecting scenes in {filename}..."):
                scene_threshold = settings.get('scene_threshold', 0.3)
                self.scene_detector = SceneDetector(threshold=scene_threshold)

                scenes = self.scene_detector.detect_scenes(validation_result['temp_path'])

                if not scenes:
                    st.warning("⚠️ No scenes detected")
                    scenes = []
                else:
                    st.success(f"✅ Detected {len(scenes)} scenes")

            # STAGE 3: AssemblyAI Transcription & Speaker Diarization  
            with st.spinner(f"Stage 3: Transcribing {filename} with AssemblyAI..."):
                enable_speaker_diarization = settings.get('enable_speaker_diarization', True)

                transcription_result = self.transcriber.transcribe_with_speaker_diarization(
                    validation_result['temp_path'],
                    enable_speaker_diarization=enable_speaker_diarization
                )

                if transcription_result.get('success'):
                    st.success("✅ AssemblyAI transcription completed")
                else:
                    st.warning("⚠️ Transcription failed or incomplete")

            # STAGE 4: Visual Analysis with Gemini
            with st.spinner(f"Stage 4: Visual analysis of {filename}..."):
                visual_analysis_results = []

                if scenes and transcription_result.get('success'):
                    for j, scene in enumerate(scenes):
                        try:
                            visual_analysis = self.frame_analyzer.analyze_scene_frames(
                                validation_result['temp_path'],
                                scene,
                                transcription_result
                            )

                            if visual_analysis:
                                # Add scene metadata
                                visual_analysis['scene_id'] = j + 1
                                visual_analysis['source_video'] = filename
                                visual_analysis_results.append(visual_analysis)

                        except Exception as e:
                            st.warning(f"⚠️ Scene {j+1} analysis failed: {str(e)}")
                            continue

                    st.success(f"✅ Analyzed {len(visual_analysis_results)} scenes")
                else:
                    st.warning("⚠️ Skipped visual analysis (no scenes or transcription failed)")

            return {
                'validation': validation_result,
                'scenes': scenes,
                'transcription': transcription_result,
                'visual_analysis': visual_analysis_results,
                'status': 'completed',
                'error': None
            }

    def get_multi_video_summary(self, results: Dict[str, Dict]) -> Dict:
        """Generate summary statistics across all videos"""

        total_videos = len(results)
        successful_videos = sum(1 for r in results.values() if r['status'] == 'completed')
        failed_videos = total_videos - successful_videos

        total_scenes = sum(
            len(r['visual_analysis']) for r in results.values() 
            if r['visual_analysis']
        )

        total_speakers = set()
        for result in results.values():
            if result['visual_analysis']:
                for scene in result['visual_analysis']:
                    speakers = scene.get('unique_speakers', [])
                    total_speakers.update(speakers)

        return {
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'total_scenes': total_scenes,
            'unique_speakers': len(total_speakers),
            'speaker_list': list(total_speakers)
        }

    def cleanup_temp_files(self, results: Dict[str, Dict]):
        """Clean up temporary files from all video processing"""
        for filename, result in results.items():
            if result['validation'] and result['validation'].get('temp_path'):
                temp_path = result['validation']['temp_path']
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        st.info(f"🗑️ Cleaned up temp file for {filename}")
                except Exception as e:
                    st.warning(f"⚠️ Could not cleanup {temp_path}: {e}")
