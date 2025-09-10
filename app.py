import streamlit as st
import os
from config import SUPPORTED_FORMATS, MAX_FILE_SIZE
from video_ingestion import VideoProcessor
from scene_detector import SceneDetector
from audio_transcriber import AudioTranscriber
from gemini_frame_analyzer import GeminiFrameAnalyzer # NEW IMPORT

st.set_page_config(
    page_title="Hybrid Video Scene Extraction",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Hybrid Video Scene Extraction")
st.markdown("Extract relevant scenes from videos based on your query")

# Initialize session state
if 'validation_result' not in st.session_state:
    st.session_state.validation_result = None
if 'scene_results' not in st.session_state:
    st.session_state.scene_results = None
if 'transcription_results' not in st.session_state:
    st.session_state.transcription_results = None
if 'visual_analysis_results' not in st.session_state: # NEW
    st.session_state.visual_analysis_results = None

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.info("Phase 1: Fast Preprocessing")
    st.markdown("**Current Step:** 1.1 Video Ingestion")

# Main interface
uploaded_file = st.file_uploader(
    "Upload your video",
    type=SUPPORTED_FORMATS,
    help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)} | Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
)

if uploaded_file is not None:
    # Check file size first
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File too large! Max size: {MAX_FILE_SIZE // (1024*1024)}MB")
    else:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        # Display file info
        st.info(f"📁 Size: {uploaded_file.size / (1024*1024):.1f}MB | Type: {uploaded_file.type}")
        
        # Process button - Combined processing
        if st.button("🚀 Start Phase 1 - Complete Processing", type="primary"):
            # Step 1: Video Validation
            with st.spinner("🔍 Step 1: Validating video with FFmpeg..."):
                processor = VideoProcessor()
                result = processor.validate_video(uploaded_file)
                st.session_state.validation_result = result
            
            if result['valid']:
                st.success("✅ Video validation passed!")
                
                # Display metadata in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{result['duration']:.1f}s")
                with col2:
                    st.metric("Resolution", result['resolution'])
                with col3:
                    st.metric("Has Audio", "✅ Yes" if result['has_audio'] else "❌ No")
                
                # Show warnings if any
                if result['warnings']:
                    st.warning("⚠️ **Warnings:**")
                    for warning in result['warnings']:
                        st.warning(f"• {warning}")
                
                # Step 2: Scene Detection
                with st.spinner("🎬 Step 2: Detecting scenes..."):
                    scene_detector = SceneDetector()
                    scenes = scene_detector.detect_scenes(result['temp_path'])
                    st.session_state.scene_results = scenes
                
                if scenes:
                    st.success(f"✅ Detected {len(scenes)} scenes!")
                    
                    # Show scene statistics
                    stats = scene_detector.get_scene_stats(scenes)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Scenes", stats['total_scenes'])
                    with col2:
                        st.metric("Avg Duration", f"{stats['avg_duration']:.1f}s")
                    with col3:
                        st.metric("Shortest", f"{stats['min_duration']:.1f}s")
                    with col4:
                        st.metric("Longest", f"{stats['max_duration']:.1f}s")
                    
                    # Step 3: Audio Transcription
                    scenes_with_text = scenes # Default fallback
                    if result['has_audio']:
                        with st.spinner("🎤 Step 3: Transcribing audio..."):
                            transcriber = AudioTranscriber()
                            
                            # Extract complete audio
                            audio_path = transcriber.extract_audio(result['temp_path'])
                            if audio_path:
                                # Extract audio segments for each scene
                                scene_audio_files = transcriber.extract_scene_audio_segments(audio_path, scenes)
                                
                                if scene_audio_files:
                                    # Transcribe each scene separately
                                    scenes_with_text = transcriber.transcribe_scenes_batch(scenes, scene_audio_files)
                                    st.session_state.transcription_results = {
                                        'transcription': {'language': 'vi'},
                                        'scenes_with_text': scenes_with_text
                                    }
                                    
                                    # Show transcription stats
                                    trans_stats = transcriber.get_transcription_stats(scenes_with_text)
                                    
                                    st.success(f"✅ Audio transcribed successfully with Blaze.vn STT!")
                                    st.info(f"🗣️ **Language detected:** vi")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Words", trans_stats['total_words'])
                                    with col2:
                                        st.metric("Scenes with Speech", trans_stats['scenes_with_speech'])
                                    with col3:
                                        st.metric("Silent Scenes", trans_stats['scenes_without_speech'])
                                    with col4:
                                        st.metric("Speech Coverage", f"{trans_stats['speech_coverage']:.1f}%")
                                    
                                    # Cleanup audio files
                                    transcriber.cleanup_scene_audio_files(scene_audio_files)
                                    transcriber.cleanup_audio_file(audio_path)
                                else:
                                    st.error("❌ Scene audio extraction failed")
                            else:
                                st.error("❌ Audio extraction failed")
                    else:
                        st.warning("⚠️ No audio track found - skipping transcription")
                    
                    # Step 4: Visual Analysis with Gemini 2.5 Flash (NEW STEP)
                    with st.spinner("🎬 Step 4: Analyzing frames with Gemini 2.5 Flash..."):
                        frame_analyzer = GeminiFrameAnalyzer()
                        
                        # Extract frames from scenes
                        scenes_with_frames = frame_analyzer.extract_frames_from_scenes(
                            result['temp_path'],
                            scenes_with_text
                        )
                        
                        if scenes_with_frames:
                            # Analyze frames with Gemini
                            enriched_scenes = frame_analyzer.analyze_scenes_batch(scenes_with_frames)
                            
                            # Update session state
                            st.session_state.transcription_results['scenes_with_text'] = enriched_scenes
                            st.session_state.visual_analysis_results = enriched_scenes
                            
                            # Show visual analysis stats
                            visual_stats = frame_analyzer.get_visual_analysis_stats(enriched_scenes)
                            
                            st.success("✅ Visual analysis completed with Gemini 2.5 Flash!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Analyzed Scenes", visual_stats['analyzed_scenes'])
                            with col2:
                                st.metric("Scene Types Found", len(visual_stats['scene_types']))
                            with col3:
                                st.metric("Avg Objects/Scene", f"{visual_stats['avg_objects_per_scene']:.1f}")
                            with col4:
                                st.metric("Avg Actions/Scene", f"{visual_stats['avg_actions_per_scene']:.1f}")
                            
                            # FIXED: Sample Analysis with expander, scenes 1-12, visual only
                            with st.expander("🎬 Sample Analysis", expanded=True):
                                # Sort scenes by scene_id and take first 12
                                all_scenes = sorted(enriched_scenes, key=lambda x: x.get('scene_id', 0))
                                sample_scenes = [s for s in all_scenes if not s.get('visual_analysis_error')][:12]
                                
                                for scene in sample_scenes:
                                    st.write(f"**Scene {scene['scene_id']}** ({scene['timestamp_str']}):")
                                    st.write(f"🎭 **Visual**: {scene.get('visual_description', 'N/A')}")
                                    st.write("---")
                                
                                if len(sample_scenes) == 12:
                                    remaining_scenes = len([s for s in all_scenes if not s.get('visual_analysis_error')]) - 12
                                    if remaining_scenes > 0:
                                        st.info(f"... and {remaining_scenes} more analyzed scenes")
                            
                            # Cleanup frames
                            frame_analyzer.cleanup_frame_files(scenes_with_frames)
                        else:
                            st.error("❌ Frame extraction failed")
                    
                    st.success("🎯 **Phase 1 Complete! Ready for Phase 2: Semantic Filtering**")
                    st.balloons()
                    
                else:
                    st.error("❌ No scenes detected. Try adjusting threshold.")
                
            else:
                st.error("❌ **Video validation failed!**")
                
                # Handle both single error and multiple errors
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    for error in result['errors']:
                        st.error(f"• {error}")
                
                # Cleanup temp file on failure
                processor.cleanup_temp_file(result.get('temp_path'))

        # Display previous results if they exist
        if st.session_state.validation_result is not None and st.session_state.validation_result['valid']:
            result = st.session_state.validation_result
            
            st.success("✅ Previous processing results:")
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{result['duration']:.1f}s")
            with col2:
                st.metric("Resolution", result['resolution'])
            with col3:
                st.metric("Has Audio", "✅ Yes" if result['has_audio'] else "❌ No")
            
            # Show detailed metadata option
            if st.checkbox("Show detailed metadata"):
                st.json(result['metadata'])
            
            # Display scene results if available
            if st.session_state.scene_results:
                scenes = st.session_state.scene_results
                scene_detector = SceneDetector()
                stats = scene_detector.get_scene_stats(scenes)
                
                st.success(f"✅ Scene Detection Results: {len(scenes)} scenes detected")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Scenes", stats['total_scenes'])
                with col2:
                    st.metric("Avg Duration", f"{stats['avg_duration']:.1f}s")
                with col3:
                    st.metric("Shortest", f"{stats['min_duration']:.1f}s")
                with col4:
                    st.metric("Longest", f"{stats['max_duration']:.1f}s")
                
                # Display transcription results if available
                if st.session_state.transcription_results:
                    transcription_data = st.session_state.transcription_results
                    transcriber = AudioTranscriber()
                    trans_stats = transcriber.get_transcription_stats(transcription_data['scenes_with_text'])
                    
                    st.success("✅ Audio Transcription Results (Blaze.vn STT - Per Scene):")
                    st.info(f"🗣️ **Language:** {transcription_data['transcription']['language']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Words", trans_stats['total_words'])
                    with col2:
                        st.metric("Scenes with Speech", trans_stats['scenes_with_speech'])
                    with col3:
                        st.metric("Silent Scenes", trans_stats['scenes_without_speech'])
                    with col4:
                        st.metric("Speech Coverage", f"{trans_stats['speech_coverage']:.1f}%")
                    
                    with st.expander("📝 Scene Transcripts"):
                        speech_scenes = [s for s in transcription_data['scenes_with_text'] if s['has_speech']][:10]
                        for scene in speech_scenes:
                            st.write(f"**Scene {scene['scene_id']}** ({scene['timestamp_str']}):")
                            st.write(f"_{scene['transcript']}_")
                            st.write("---")
                
                # FIXED: Complete Scene Analysis with visual description, no mood
                if st.session_state.visual_analysis_results:
                    st.success("✅ Visual Analysis Results (Gemini 2.5 Flash):")
                    
                    with st.expander("🎬 Complete Scene Analysis"):
                        # Sort scenes by scene_id for consistent display
                        all_scenes = sorted(st.session_state.visual_analysis_results, key=lambda x: x.get('scene_id', 0))
                        
                        # Hiển thị tất cả các cảnh
                        for scene in all_scenes:
                            st.write(f"**Scene {scene['scene_id']}** ({scene['timestamp_str']}):")
                            st.write(f"🎭 **Visual**: {scene.get('visual_description', 'N/A')}")
                            st.write(f"📝 **Audio**: {scene.get('transcript', 'No speech')}")
                            st.write(f"🎯 **Type**: {scene.get('scene_type', 'Unknown')}")
                            st.write("---")
                
                with st.expander("📋 Scene List"):
                    for scene in scenes[:10]:
                        st.write(f"**Scene {scene['scene_id']}**: {scene['timestamp_str']} ({scene['duration']:.1f}s)")
                    
                    if len(scenes) > 10:
                        st.info(f"... and {len(scenes) - 10} more scenes")
            
            # Cleanup option
            if st.button("🗑️ Clean up temporary files"):
                processor = VideoProcessor()
                processor.cleanup_temp_file(result.get('temp_path'))
                st.session_state.validation_result = None
                st.session_state.scene_results = None
                st.session_state.transcription_results = None
                st.session_state.visual_analysis_results = None
                st.success("Temporary files cleaned up!")
                st.rerun()

else:
    st.info("👆 Please upload a video file to start processing")
    
    # Show supported formats
    st.markdown("### 📋 Supported Formats")
    cols = st.columns(len(SUPPORTED_FORMATS))
    for i, format_name in enumerate(SUPPORTED_FORMATS):
        cols[i].info(f"**{format_name.upper()}**")
