import asyncio
import concurrent.futures
from typing import List, Dict, Any
import streamlit as st
import time
from video_ingestion import VideoProcessor
from scene_detector import SceneDetector
from audio_transcriber import AssemblyAITranscriber
from gemini_frame_analyzer import GeminiFrameAnalyzer

class OptimizedVideoProcessor:
    """🎬 Optimized Video Processor with Async API calls and Parallel CPU processing"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.transcriber = AssemblyAITranscriber()
        self.frame_analyzer = GeminiFrameAnalyzer()
        
        # Optimal concurrency limits based on API tiers and 64GB RAM
        self.assemblyai_concurrency = 10  # Safe limit under 200 concurrent requests
        self.gemini_concurrency = 8  # Safe limit under 150 RPM (Tier 1)
        self.cpu_workers = 6  # Optimal for CPU-intensive tasks

    async def process_videos_optimized(self, uploaded_files: List, settings: Dict):
        """🚀 Main async processing method - handles all 4 stages optimally"""
        
        # Stage 1: Video Validation (CPU-intensive) - Parallel
        st.info("🎬 Video Validation")
        validation_results = await self._validate_videos_parallel(uploaded_files, settings)
        
        valid_videos = {
            name: result for name, result in validation_results.items()
            if result.get('valid', False)
        }
        
        if not valid_videos:
            st.error("❌ No valid videos found")
            return {}

        # Stage 2: Scene Detection (CPU-intensive) - Parallel
        st.info("🎪 Scene Detection")
        scene_results = await self._detect_scenes_parallel(valid_videos, settings)
        
        # Stage 3: Transcription (API-intensive) - Async with concurrency control
        st.info("🎤 Advanced STT + Speaker Recognition")
        transcription_results = await self._transcribe_videos_async(valid_videos, scene_results, settings)
        
        # Stage 4: Visual Analysis (API-intensive) - Async with concurrency control
        st.info("🎭 Visual Analysis (AI)")
        visual_results = await self._analyze_visuals_async(valid_videos, transcription_results)        # Combine final results
        final_results = {}
        for video_name in valid_videos.keys():
            final_results[video_name] = {
                'validation': valid_videos[video_name],
                'scenes': scene_results.get(video_name, []),
                'transcription': transcription_results.get(video_name, {}),
                'visual_analysis': visual_results.get(video_name, []),
                'status': 'completed',
                'processing_complete': True
            }
        
        return final_results

    async def _validate_videos_parallel(self, uploaded_files: List, settings: Dict) -> Dict:
        """Stage 1: Parallel video validation (CPU-bound) - FIXED VERSION"""
        loop = asyncio.get_event_loop()
        
        # ✅ FIX: Create tasks with proper file name tracking
        tasks_with_names = []
        for uploaded_file in uploaded_files:
            task = loop.run_in_executor(
                None,  # Use default executor
                self._validate_single_video_sync,
                uploaded_file, settings
            )
            tasks_with_names.append((uploaded_file.name, task))
        
        # Progress tracking
        progress_bar = st.progress(0)
        results = {}
        completed = 0
        
        # ✅ FIX: Proper async completion with correct file mapping
        for file_name, task in tasks_with_names:
            try:
                result = await task
                results[file_name] = result
                completed += 1
                progress_bar.progress(completed / len(tasks_with_names))
                # st.write(f"✅ Validated: {file_name}")
            except Exception as e:
                results[file_name] = {'valid': False, 'error': str(e)}
                completed += 1
                progress_bar.progress(completed / len(tasks_with_names))
                # st.write(f"❌ Failed: {file_name}")
        
        return results

    async def _detect_scenes_parallel(self, valid_videos: Dict, settings: Dict) -> Dict:
        """Stage 2: Parallel scene detection (CPU-bound) - FIXED VERSION"""
        loop = asyncio.get_event_loop()
        video_items = list(valid_videos.items())
        
        # ✅ FIX: Create tasks with proper video_name tracking
        tasks_with_names = []
        for video_name, video_data in video_items:
            task = loop.run_in_executor(
                None,  # Use default executor
                self._detect_scenes_single_video_sync,
                video_data['temp_path'],
                video_data['duration'],
                settings
            )
            tasks_with_names.append((video_name, task))
        
        # Progress tracking
        progress_bar = st.progress(0)
        results = {}
        completed = 0
        
        # ✅ FIX: Proper async completion handling with correct video mapping
        for video_name, task in tasks_with_names:
            try:
                result = await task
                results[video_name] = result
                completed += 1
                progress_bar.progress(completed / len(tasks_with_names))
                # st.write(f"✅ Detected {len(result)} scenes in: {video_name}")
            except Exception as e:
                results[video_name] = []
                completed += 1
                progress_bar.progress(completed / len(tasks_with_names))
                # st.write(f"❌ Scene detection failed: {video_name}")
        
        return results

    async def _transcribe_videos_async(self, valid_videos: Dict, scene_results: Dict, settings: Dict) -> Dict:
        """Stage 3: Async transcription with AssemblyAI (API-bound)"""
        # Create semaphore to limit concurrent AssemblyAI requests
        semaphore = asyncio.Semaphore(self.assemblyai_concurrency)

        async def transcribe_single_video(video_name: str, video_data: Dict, scenes: List):
            async with semaphore:
                # Show current processing
                # st.write(f"🎤 Transcribing: {video_name}")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # Use default thread pool
                    self._transcribe_single_video_sync,
                    video_name, video_data, scenes, settings  # ✅ FIX: Pass video_name for isolation
                )
                return video_name, result  # ✅ FIX: Return video_name with result

        # Create tasks for all videos with audio
        tasks = []
        for video_name, video_data in valid_videos.items():
            if video_data.get('has_audio', False):
                tasks.append(transcribe_single_video(
                    video_name,
                    video_data,
                    scene_results.get(video_name, [])
                ))
            else:
                # Handle videos without audio immediately
                scenes = scene_results.get(video_name, [])
                tasks.append(asyncio.create_task(self._create_no_audio_result(video_name, scenes)))

        if not tasks:
            return {}

        # Progress tracking
        progress_bar = st.progress(0)
        completed = 0
        results = {}

        # ✅ FIX: Properly handle results with video names
        for coro in asyncio.as_completed(tasks):
            try:
                video_name, result = await coro
                results[video_name] = result
                completed += 1
                progress_bar.progress(completed / len(tasks))
                # st.write(f"✅ Transcription complete: {video_name}")
            except Exception as e:
                # st.write(f"❌ Transcription failed: {str(e)}")
                pass

        return results

    async def _analyze_visuals_async(self, valid_videos: Dict, transcription_results: Dict) -> Dict:
        """Stage 4: Async visual analysis with Gemini (API-bound) - FIXED VERSION"""
        # Create semaphore to limit concurrent Gemini requests
        semaphore = asyncio.Semaphore(self.gemini_concurrency)

        async def analyze_single_video(video_name: str, video_data: Dict, transcription_data: Dict):
            async with semaphore:
                # st.write(f"🎭 Analyzing visuals: {video_name}")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._analyze_visual_single_video_sync,
                    video_name, video_data['temp_path'], transcription_data  # ✅ FIX: Pass video_name to ensure correct mapping
                )
                return video_name, result  # ✅ FIX: Return video_name with result

        # Create tasks for all videos
        tasks = [
            analyze_single_video(
                video_name,
                video_data,
                transcription_results.get(video_name, {})
            )
            for video_name, video_data in valid_videos.items()
        ]

        # Progress tracking
        progress_bar = st.progress(0)
        completed = 0
        results = {}

        # ✅ FIX: Properly handle async results with correct video mapping
        for coro in asyncio.as_completed(tasks):
            try:
                video_name, result = await coro
                results[video_name] = result  # ✅ Correct mapping now!
                completed += 1
                progress_bar.progress(completed / len(tasks))
                # st.write(f"✅ Visual analysis complete: {video_name}")
            except Exception as e:
                # st.write(f"❌ Visual analysis failed: {str(e)}")
                # Note: Can't provide fallback without knowing which video failed
                pass

        return results

    # Synchronous helper methods
    def _validate_single_video_sync(self, uploaded_file, settings):
        """Sync video validation"""
        return self.video_processor.validate_video(
            uploaded_file,
            skip_duration_check=settings.get('skip_min_duration_check', True)
        )

    def _detect_scenes_single_video_sync(self, video_path: str, duration: float, settings: Dict):
        """Sync scene detection"""
        scene_detector = SceneDetector(threshold=settings.get('scene_threshold', 27.0))
        
        # Override config values temporarily
        import config
        original_min_duration = config.MIN_SCENE_DURATION
        original_merge_threshold = config.MERGE_THRESHOLD_DURATION
        config.MIN_SCENE_DURATION = settings.get('min_scene_duration', 3.0)
        config.MERGE_THRESHOLD_DURATION = settings.get('merge_threshold_duration', 5.0)
        
        try:
            scenes = scene_detector.detect_scenes(video_path)
            
            # If no scenes detected, create single scene for entire video
            if not scenes:
                scenes = [{
                    'scene_id': 1,
                    'start_time': 0.0,
                    'end_time': duration,
                    'duration': duration,
                    'timestamp_str': f"00:00:00.000 - {int(duration//3600):02d}:{int((duration%3600)//60):02d}:{duration%60:06.3f}"
                }]
            
            return scenes
        finally:
            # Restore original config values
            config.MIN_SCENE_DURATION = original_min_duration
            config.MERGE_THRESHOLD_DURATION = original_merge_threshold

    def _transcribe_single_video_sync(self, video_name: str, video_data: Dict, scenes: List, settings: Dict):
        """Sync transcription - FIXED with video isolation and unique audio naming"""
        if not video_data.get('has_audio', False):
            return {
                'transcription': {'language': 'none'},
                'scenes_with_text': scenes,
                'speaker_enabled': False
            }
        
        # ✅ CRITICAL FIX: Add video_name to each scene BEFORE processing with unique identifier
        scenes_with_video_id = []
        for scene in scenes:
            scene_copy = scene.copy()
            scene_copy['source_video'] = video_name  # Ensure video tracking
            scene_copy['unique_video_id'] = f"{video_name}_{scene.get('scene_id', 'unknown')}"  # Unique identifier
            scenes_with_video_id.append(scene_copy)
        
        # ✅ CRITICAL FIX: Create isolated transcriber instance for this video
        from audio_transcriber import AssemblyAITranscriber
        isolated_transcriber = AssemblyAITranscriber()
        
        # Process with isolated transcriber using scenes with video ID
        scenes_with_text = isolated_transcriber.process_complete_video(
            video_data['temp_path'],
            scenes_with_video_id,  # ✅ Use scenes with unique video ID
            settings.get('speakers_expected') if settings.get('enable_speaker_diarization') else None
        )
        
        # ✅ FIX: Triple-check that all returned scenes have correct video source
        for scene in scenes_with_text:
            if 'source_video' not in scene or scene['source_video'] != video_name:
                scene['source_video'] = video_name
                # st.write(f"🔧 Fixed source_video for scene {scene.get('scene_id')} -> {video_name}")
        
        return {
            'transcription': {'language': 'vi'},
            'scenes_with_text': scenes_with_text,
            'speaker_enabled': settings.get('enable_speaker_diarization', False)
        }

    def _analyze_visual_single_video_sync(self, video_name: str, video_path: str, transcription_data: Dict):
        """Sync visual analysis - FIXED to ensure correct video-specific transcription mapping"""
        scenes_with_text = transcription_data.get('scenes_with_text', [])
        
        # ✅ FIX: Add video_name to each scene to ensure correct mapping
        for scene in scenes_with_text:
            if 'source_video' not in scene:
                scene['source_video'] = video_name
        
        # Extract frames from scenes
        scenes_with_frames = self.frame_analyzer.extract_frames_from_scenes(
            video_path, scenes_with_text
        )
        
        if scenes_with_frames:
            # ✅ FIX: Ensure each scene maintains video source information
            for scene in scenes_with_frames:
                if 'source_video' not in scene:
                    scene['source_video'] = video_name
                    
            # Analyze frames with AI
            enriched_scenes = self.frame_analyzer.analyze_scenes_batch(scenes_with_frames)
            
            # ✅ FIX: Final check - ensure all scenes have correct video source
            for scene in enriched_scenes:
                if 'source_video' not in scene:
                    scene['source_video'] = video_name
            
            # Cleanup frames
            self.frame_analyzer.cleanup_frame_files(scenes_with_frames)
            return enriched_scenes
        else:
            # ✅ FIX: Ensure scenes without frames also have correct video source
            for scene in scenes_with_text:
                if 'source_video' not in scene:
                    scene['source_video'] = video_name
            return scenes_with_text

    async def _create_no_audio_result(self, video_name: str, scenes: List):
        """Create result for videos without audio"""
        result = {
            'transcription': {'language': 'none'},
            'scenes_with_text': scenes,
            'speaker_enabled': False
        }
        return video_name, result  # ✅ FIX: Return video_name with result

    def get_processing_stats(self, results: Dict) -> Dict:
        """Get aggregated statistics from processing results"""
        if not results:
            return {}
            
        total_duration = 0
        total_scenes = 0
        total_words = 0
        total_videos_with_audio = 0
        total_unique_speakers = set()
        
        for video_name, video_data in results.items():
            if video_data.get('status') == 'completed':
                # Duration
                if 'validation' in video_data:
                    total_duration += video_data['validation'].get('duration', 0)
                
                # Scenes
                total_scenes += len(video_data.get('scenes', []))
                
                # Audio stats
                transcription_data = video_data.get('transcription', {})
                if transcription_data.get('transcription', {}).get('language') != 'none':
                    total_videos_with_audio += 1
                    
                    # Calculate words and speakers
                    scenes_with_text = transcription_data.get('scenes_with_text', [])
                    for scene in scenes_with_text:
                        if scene.get('transcript'):
                            total_words += len(scene['transcript'].split())
                        if scene.get('unique_speakers'):
                            total_unique_speakers.update(scene['unique_speakers'])
        
        return {
            'total_videos': len(results),
            'successful_videos': len([r for r in results.values() if r.get('status') == 'completed']),
            'total_duration': total_duration,
            'total_scenes': total_scenes,
            'total_words': total_words,
            'videos_with_audio': total_videos_with_audio,
            'unique_speakers': len(total_unique_speakers)
        }
