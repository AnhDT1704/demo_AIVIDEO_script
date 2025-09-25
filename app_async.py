import warnings
import logging
import os

# Suppress ALL Streamlit warnings về ScriptRunContext
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*client.caching.*")

# Tắt tất cả warnings từ streamlit runtime
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.CRITICAL) 
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Set environment variable để tắt warnings
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

from optimized_video_processor import OptimizedVideoProcessor
# app_enhanced_assemblyai.py - Enhanced with AssemblyAI STT + Speaker Diarization

import streamlit as st
import os
import json
import re
from datetime import datetime
from config import SUPPORTED_FORMATS, MAX_FILE_SIZE
from video_ingestion import VideoProcessor
from scene_detector import SceneDetector
from audio_transcriber import AssemblyAITranscriber
from gemini_frame_analyzer import GeminiFrameAnalyzer
from script_generator import DirectScriptGenerator

# Function to parse final script and convert to JSON format
def parse_script_to_acts(final_script_text, selected_scenes):
    """
    Parse the final script text and create separate JSON files for each act
    Returns a dictionary with act data for creating separate files
    """
    acts_data = {}
    
    # Extract main title from the script
    script_title = "UNKNOWN_SCRIPT"
    title_match = re.search(r'KỊCH BẢN[:\s]*(.+?)(?:\n|\r)', final_script_text, re.IGNORECASE)
    if title_match:
        script_title = title_match.group(1).strip()
    
    # Split script by acts (looking for "Cảnh X:" patterns)
    act_sections = re.split(r'\n\s*Cảnh\s+(\d+):', final_script_text)
    
    # Process each act section
    for i in range(1, len(act_sections), 2):
        if i + 1 < len(act_sections):
            act_number = int(act_sections[i])
            act_content = act_sections[i + 1]
            
            # Extract act title (first line after act number)
            lines = act_content.strip().split('\n')
            act_title = lines[0].strip() if lines else f"Act {act_number}"
            
            # Extract voiceover transcription (text after "Voiceover cho Cảnh")
            voice_over = ""
            voiceover_match = re.search(r'🎙️\s*Voiceover cho Cảnh[^:]*:\s*(.+?)(?=\n\n|\n\s*Cảnh|\n\s*🎙️|\Z)', act_content, re.DOTALL)
            if voiceover_match:
                voice_over = voiceover_match.group(1).strip()
            
            # Extract scenes from the act content
            scene_data = []
            scene_count = 0
            
            # Find all scene sections within this act
            scene_matches = re.findall(r'Scene\s+(\d+):\s*\(([^)]+)\)\s*-\s*From:\s*([^\n]+)', act_content)
            
            for scene_match in scene_matches:
                scene_num, time_range, filename = scene_match
                
                # Parse time range (00:00:11.800 - 00:00:27.199)
                time_parts = time_range.split(' - ')
                start_time = time_parts[0].strip() if len(time_parts) > 0 else "00:00:00.000"
                end_time = time_parts[1].strip() if len(time_parts) > 1 else "00:00:00.000"
                
                scene_count += 1
                scene_info = {
                    "order": scene_count,
                    "start": start_time,
                    "end": end_time,
                    "Video_filename": filename.strip()
                }
                scene_data.append(scene_info)
            
            # Create act data structure
            acts_data[f"act{act_number}"] = {
                "act_number": act_number,
                "act_data": {
                    "act_title": act_title,
                    "voice_over_transcription": voice_over,
                    "scene_number": scene_count,
                    "scenes": scene_data
                }
            }
    
    return script_title, acts_data

def create_acts_folder_and_files(script_title, acts_data):
    """
    Create a folder with script title and individual JSON files for each act
    Returns the folder path and list of created files
    """
    import os
    import json
    from datetime import datetime
    
    # Clean script title for folder name
    clean_title = re.sub(r'[<>:"/\\|?*]', '_', script_title)
    folder_name = f"EXPORTED_{clean_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder_path = os.path.join('data/outputs', folder_name)
    
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    created_files = []
    
    # Create individual JSON files for each act
    for act_key, act_data in acts_data.items():
        filename = f"{act_key}.json"
        filepath = os.path.join(folder_path, filename)
        
        # Write act data to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(act_data, f, ensure_ascii=False, indent=2)
        
        created_files.append(filename)
    
    return folder_path, created_files

def create_downloadable_json(json_data, filename):
    """
    Create a downloadable JSON file
    """
    json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
    return json_string.encode('utf-8')

st.set_page_config(
    page_title="Enhanced Video Scene Extraction",
    page_icon="🎬",  # Sửa: Thay ký tự Unicode bị lỗi bằng emoji chính xác
    layout="wide"
)

st.title("🎬 Enhanced Video Scene Extraction")
st.markdown("Extract relevant scenes from videos with **Advanced STT + Speaker Recognition**")

# Add custom CSS for better Final Script display
st.markdown("""
<style>
    /* Enhanced styling for Final Script Output */
    .final-script-output {
        background-color: #1e1e1e !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        color: #ffffff !important;
    }
    
    /* Make text areas darker */
    .stTextArea textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #4CAF50 !important;
    }
    
    /* Success message styling */
    .success-header {
        background: linear-gradient(90deg, #4CAF50, #45a049) !important;
        padding: 10px !important;
        border-radius: 5px !important;
        color: white !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ✅ HIDE Processing Pipeline steps - commented out
# st.markdown("""
# ### 📋 Processing Pipeline
# ✔ **Phase 1**: Enhanced Preprocessing with AI  
# 🎬 **Step 1**: Video Validation  
# 🎪 **Step 2**: Scene Detection  
# 🎤 **Step 3**: Advanced STT + Speaker Recognition  
# 🎭 **Step 4**: Visual Analysis (AI)  
# 📝 **Step 5**: Final Script with Voiceover
# """, unsafe_allow_html=True)

# Initialize session state
if 'validation_result' not in st.session_state:
    st.session_state.validation_result = None
if 'scene_results' not in st.session_state:
    st.session_state.scene_results = None
if 'transcription_results' not in st.session_state:
    st.session_state.transcription_results = None
if 'visual_analysis_results' not in st.session_state:
    st.session_state.visual_analysis_results = None
if 'script_generator_results' not in st.session_state:
    st.session_state.script_generator_results = None
# NEW: Store results for multiple videos
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = {}

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.info("Phase 1: Enhanced Preprocessing with AI")
    st.markdown("**Current Features:**")
    st.markdown("- ✔ Video Validation")
    st.markdown("- ✔ Scene Detection") 
    st.markdown("- ✔ **Advanced STT + Speaker Recognition**")
    st.markdown("- ✔ Visual Analysis (AI)")
    st.markdown("- ✔ **Final Script Generation**")

    # Scene Detection Settings
    st.subheader("🎬 Scene Detection")
    st.markdown("**Scene Threshold Control:**")

    # Preset options
    preset_option = st.selectbox(
        "Choose Preset or Custom:",
        ["Custom", "Fast Cuts (15)", "Balanced (27)", "Conservative (35)", "Very Conservative (50)"],
        index=2,  # Default to Balanced (27)
        help="Presets based on video content type"
    )

    if preset_option == "Custom":
        scene_threshold = st.slider(
            "Scene Detection Threshold",
            min_value=5.0,
            max_value=95.0,
            value=27.0,  # PySceneDetect default
            step=1.0,
            help="Lower = More sensitive (more scenes), Higher = Less sensitive (fewer scenes)"
        )
    else:
        # Extract threshold from preset name
        threshold_mapping = {
            "Fast Cuts (15)": 15.0,
            "Balanced (27)": 27.0,
            "Conservative (35)": 35.0,
            "Very Conservative (50)": 50.0
        }
        scene_threshold = threshold_mapping[preset_option]
        st.info(f"**Using Threshold:** {scene_threshold}")

    # Threshold recommendations
    with st.expander("💡 Threshold Guide", expanded=False):
        st.markdown("""
        **Threshold Recommendations:**
        - **5-15**: Very sensitive, many micro-cuts
        - **15-20**: Good for fast-cut videos (music videos, action)
        - **25-30**: Balanced, works for most content ✔
        - **35-45**: Conservative, only major scene changes
        - **50+**: Very conservative, minimal scenes

        **Video Type Guide:**
        - **Talk shows/Interviews**: 25-35
        - **Movies/Drama**: 20-30
        - **Action/Music videos**: 15-25
        - **Documentary**: 30-40
        - **Low quality video**: 20-25
        """)

    # Advanced Scene Settings
    with st.expander("⚙️ Advanced Scene Settings", expanded=False):
        min_scene_duration = st.slider(
            "Minimum Scene Duration (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Scenes shorter than this will be merged"
        )

        merge_threshold_duration = st.slider(
            "Merge Threshold Duration (seconds)",
            min_value=3.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Adjacent short scenes will be merged if below this duration"
        )

        # NEW: Add option to skip minimum video duration check - Default to True
        skip_min_duration_check = st.checkbox(
            "Accept videos of any duration",
            value=True,
            help="Process videos regardless of duration (recommended for flexibility)"
        )
        
        if not skip_min_duration_check:
            st.warning("⚠️ Only videos ≥ 30 seconds will be processed")
        else:
            st.info("✔ All video durations accepted")

    st.divider()

    # ✅ HIDE Speech Recognition Settings - set default values
    # st.subheader("🎤 Speech Recognition Settings")
    enable_speaker_diarization = True  # Default to enabled
    speakers_expected = None  # Default to automatic detection
    
    # Hidden speech settings (commented out)
    # enable_speaker_diarization = st.checkbox(
    #     "Enable Speaker Recognition",
    #     value=True,
    #     help="Uses advanced AI to identify different speakers"
    # )
    # 
    # if enable_speaker_diarization:
    #     st.info("🎭 **Speaker Format**: Speaker A: ..., Speaker B: ...")
    #     speakers_expected = st.number_input(
    #         "Expected Number of Speakers (optional)",
    #         min_value=0,
    #         max_value=10,
    #         value=0,
    #         help="Leave 0 for automatic detection"
    #     )
    #     if speakers_expected == 0:
    #         speakers_expected = None

# Main interface
st.subheader("📤 Upload your videos")
uploaded_files = st.file_uploader(
    "Upload your videos (multiple files supported)",
    accept_multiple_files=True,
    type=SUPPORTED_FORMATS,
    help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)} | Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
)

# Display uploaded files information
if uploaded_files is not None and len(uploaded_files) > 0:
    st.success(f"✔ {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Show all uploaded files in a compact format
    for idx, uploaded_file in enumerate(uploaded_files):
        # Check file size first
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ {uploaded_file.name}: File too large! Max size: {MAX_FILE_SIZE // (1024*1024)}MB")
        else:
            col1, col2 = st.columns(2)
            with col1:
                pass  # Hidden message
            with col2:
                pass  # Hidden message

    # Single process button for all videos
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Enhanced Phase 1 - Process All Videos Async", type="primary", key="start_processing_all_videos", use_container_width=True):
            # Initialize optimized processor
            async_processor = OptimizedVideoProcessor()

            # Collect settings
            settings = {
                'scene_threshold': scene_threshold,
                'skip_min_duration_check': skip_min_duration_check,
                'enable_speaker_diarization': enable_speaker_diarization,
                'speakers_expected': speakers_expected,
                'min_scene_duration': min_scene_duration,
                'merge_threshold_duration': merge_threshold_duration
            }

            # Run async processing
            with st.spinner("🔄 Processing all videos with optimized async pipeline..."):
                try:
                    # Create and run async processing
                    import asyncio

                    # Check if we're already in an async context
                    try:
                        # Try to get current loop
                        loop = asyncio.get_running_loop()
                        # If we get here, we're in async context - use ThreadPoolExecutor
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, 
                                async_processor.process_videos_optimized(uploaded_files, settings)
                            )
                            all_results = future.result()
                    except RuntimeError:
                        # No running loop - safe to use asyncio.run
                        all_results = asyncio.run(
                            async_processor.process_videos_optimized(uploaded_files, settings)
                        )

                    # Update session state with all results
                    st.session_state.processed_videos.update(all_results)

                    # Get processing statistics
                    stats = async_processor.get_processing_stats(all_results)

                    # Display results
                    successful_count = stats.get('successful_videos', 0)
                    total_count = len(uploaded_files)

                    if successful_count == total_count:
                        st.success(f"🎭 Async processing completed successfully!")
                        st.success(f"✅ All {successful_count} videos processed with enhanced AI!")
                    else:
                        st.warning(f"⚠️ Partial success: {successful_count}/{total_count} videos processed")

                    # Display aggregated statistics
                    if stats:
                        st.info("📊 **Processing Summary:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Duration", f"{stats['total_duration']:.1f}s")
                        with col2:
                            st.metric("Total Scenes", stats['total_scenes'])
                        with col3:
                            st.metric("Total Words", stats['total_words'])
                        with col4:
                            st.metric("Unique Speakers", stats['unique_speakers'])

                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Async processing failed: {str(e)}")
                    st.info("💡 Falling back to sequential processing...")

                    # Fallback: Original sequential processing
                    for idx, uploaded_file in enumerate(uploaded_files):
                        # Skip files that are too large
                        if uploaded_file.size > MAX_FILE_SIZE:
                            st.error(f"⏭️ Skipping {uploaded_file.name}: File too large")
                            continue

                        st.header(f"🎬 Processing Video {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                        # Step 1: Video Validation
                        with st.spinner(f"🔍 Step 1: Validating video {uploaded_file.name} with FFmpeg..."):
                            processor = VideoProcessor()
                            # Pass the skip flag to video validation
                            result = processor.validate_video(uploaded_file, skip_duration_check=skip_min_duration_check)
                            st.session_state.validation_result = result

                        if result['valid']:
                            st.success(f"✔ Video validation passed for {uploaded_file.name}!")
                            # Display metadata in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Duration", f"{result['duration']:.1f}s")
                            with col2:
                                st.metric("Resolution", result['resolution'])
                            with col3:
                                st.metric("Has Audio", "✔ Yes" if result['has_audio'] else "❌ No")

                            # Step 2: Scene Detection with Custom Threshold
                            with st.spinner(f"🎬 Step 2: Detecting scenes in {uploaded_file.name} (threshold: {scene_threshold})..."):
                                # Create SceneDetector with custom threshold
                                scene_detector = SceneDetector(threshold=scene_threshold)

                                # Override config values for this session
                                import config
                                original_min_duration = config.MIN_SCENE_DURATION
                                original_merge_threshold = config.MERGE_THRESHOLD_DURATION
                                config.MIN_SCENE_DURATION = min_scene_duration
                                config.MERGE_THRESHOLD_DURATION = merge_threshold_duration

                                scenes = scene_detector.detect_scenes(result['temp_path'])

                                # Restore original config values
                                config.MIN_SCENE_DURATION = original_min_duration
                                config.MERGE_THRESHOLD_DURATION = original_merge_threshold

                                st.session_state.scene_results = scenes

                            if scenes:
                                st.success(f"✔ Detected {len(scenes)} scenes in {uploaded_file.name} with threshold {scene_threshold}!")
                            else:
                                # Create single scene for entire video
                                scenes = [{'scene_id': 1, 'start_time': 0.0, 'end_time': result['duration'], 'duration': result['duration'], 'timestamp_str': f"00:00:00.000 - {int(result['duration']//3600):02d}:{int((result['duration']%3600)//60):02d}:{result['duration']%60:06.3f}"}]
                                st.session_state.scene_results = scenes

                            # Continue with original processing...
                            # (Rest of original sequential processing code)

                        else:
                            st.error(f"❌ **Video validation failed for {uploaded_file.name}!**")
# NEW: Display results for each processed video individually
if st.session_state.processed_videos:
    st.divider()
    # ✅ CLEAN UI: Hide all aggregated stats and processing details
    
    # Individual video analysis sections (only the detailed scene analysis)
    st.subheader("🎬 Individual Video Scene Analysis")
    
    for video_name, video_data in st.session_state.processed_videos.items():
        if video_data.get('status') == 'completed':
            transcription_data = video_data['transcription']
            visual_analysis_results = video_data['visual_analysis']
            
            # ✅ CLEAN UI: Create mapping ONLY for this specific video (no debug info)
            transcription_scenes_map = {}
            if transcription_data.get('scenes_with_text'):
                for scene in transcription_data['scenes_with_text']:
                    scene_id = scene.get('scene_id')
                    source_video = scene.get('source_video')
                    
                    # ✅ ULTRA STRICT: Only accept scenes that explicitly belong to this video
                    if scene_id and source_video and source_video == video_name:
                        transcription_scenes_map[scene_id] = scene
            
            # Complete Scene Analysis for this video
            with st.expander(f"🎬 Complete Scene Analysis with Advanced AI for {video_name}"):
                # Sort scenes by scene_id for consistent display
                all_scenes = sorted(visual_analysis_results, key=lambda x: x.get('scene_id', 0))
                
                for scene in all_scenes:
                    scene_id = scene.get('scene_id')
                    st.write(f"**Scene {scene_id}** ({scene['timestamp_str']}):")
                    
                    # ✅ Clean visual description display
                    visual_desc = scene.get('visual_description', '')
                    if not visual_desc or visual_desc.strip() == '':
                        visual_desc = "⚠️ Visual analysis not available"
                    
                    st.write(f"🎭 **Visual**: {visual_desc}")
                    
                    # ✅ FIX: Get conversation from transcription_data for THIS video, not from visual_analysis_results
                    speaker_enabled = transcription_data.get('speaker_enabled', False)
                    transcription_scene = transcription_scenes_map.get(scene_id)
                    
                    if speaker_enabled and transcription_scene and transcription_scene.get('speaker_transcript'):
                        st.write(f"🎤 **Conversation**:")
                        for line in transcription_scene['speaker_transcript'].split('\n'):
                            if line.strip():
                                st.write(f"  {line}")
                        
                        if transcription_scene.get('unique_speakers'):
                            speakers_text = ", ".join(transcription_scene['unique_speakers'])
                            st.write(f"👥 **Speakers**: {speakers_text}")
                    elif transcription_scene and transcription_scene.get('transcript'):
                        st.write(f"📝 **Audio**: {transcription_scene['transcript']}")
                    else:
                        st.write(f"📝 **Audio**: No speech")
                    
                    st.write(f"🎭 **Type**: {scene.get('scene_type', 'Unknown')}")
                    st.write("---")
            
            # Enhanced transcripts display for this video
            if transcription_data['transcription']['language'] != 'none':
                speaker_enabled = transcription_data.get('speaker_enabled', False)
                
                if not speaker_enabled:
                    with st.expander(f"📝 Scene Transcripts for {video_name}"):
                        speech_scenes = [s for s in transcription_data['scenes_with_text'] if s.get('has_speech', False)][:10]
                        
                        for scene in speech_scenes:
                            st.write(f"**Scene {scene['scene_id']}** ({scene['timestamp_str']}):")
                            st.write(f"_{scene.get('transcript', 'No speech')}_")
                            st.write("---")

# NEW: Step 3 - Script Generation Section (TWO PHASES) - Updated to use ALL videos
if st.session_state.processed_videos and len(st.session_state.processed_videos) > 0:
    st.divider()
    st.header("📝 Step 3: Final Script with Voiceover")
    
    # Show processing steps
    st.markdown("### 📋 Script Generation Pipeline")
    st.markdown("🎭 **Phase 1**: Scene Selection & Initial Script")
    st.markdown("🎙️ **Phase 2**: Final Script with Voiceover")
    
    # PHASE 1: Scene Selection & Initial Script Generation
    st.subheader("🎭 Phase 1: Scene Selection & Initial Script")
    
    with st.form("initial_script_generation_from_all_videos_form"):
        st.markdown("### 🎬 Scene Selection from All Videos")
        
        # User request for script
        script_request = st.text_area(
            "Mô tả kịch bản bạn muốn tạo:",
            placeholder="Ví dụ: Tạo cho tôi một kịch bản highlight những phương pháp học tập từ thủ khoa",
            help="Mô tả rõ ràng nội dung kịch bản bạn muốn tạo từ TẤT CẢ video này",
            height=100
        )
        
        # Show available scenes info from ALL videos
        if st.session_state.processed_videos:
            # Calculate total scenes from all processed videos
            total_scenes_all_videos = 0
            total_scenes_with_speech_all = 0
            all_processed_scenes = []
            
            for video_name, video_data in st.session_state.processed_videos.items():
                if video_data.get('status') == 'completed':
                    visual_scenes = video_data['visual_analysis']
                    total_scenes_all_videos += len(visual_scenes)
                    
                    # Add source video info to each scene
                    for scene in visual_scenes:
                        scene_with_source = scene.copy()
                        scene_with_source['source_video'] = video_name
                        all_processed_scenes.append(scene_with_source)
                        
                        if scene.get('has_speech', False):
                            total_scenes_with_speech_all += 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Available Scenes", total_scenes_all_videos)
            with col2:
                st.metric("🎤 Scenes with Speech", total_scenes_with_speech_all) 
            with col3:
                st.metric("🎭 Ready for Analysis", "✔ Yes")
                
            st.info(f"💡 AI will analyze scenes from ALL {len(st.session_state.processed_videos)} videos and select the most relevant ones for your request.")
        
        # Generate initial script button
        generate_initial_script = st.form_submit_button("🎭 Generate Scene Selection & Initial Script from All Videos", type="primary")

        if generate_initial_script and script_request.strip():
            with st.spinner("🔄 Analyzing scenes from all videos and generating initial script..."):
                try:
                    # Initialize script generator
                    script_generator = DirectScriptGenerator()
                    
                    # Use ALL scenes from ALL processed videos
                    st.info(f"🔍 Analyzing {len(all_processed_scenes)} scenes from {len(st.session_state.processed_videos)} videos for script relevance...")
                    script_result = script_generator.generate_script_from_scenes(
                        script_request, 
                        all_processed_scenes
                    )
                    
                    if script_result.get("success"):
                        # Extract selected scenes from the generated script
                        selected_scenes = script_generator.extract_selected_scenes_from_script(
                            script_result["script"], 
                            all_processed_scenes
                        )
                        
                        # Store initial results
                        st.session_state.script_generator_results = {
                            'phase': 'initial',
                            'initial_script': script_result,
                            'selected_scenes': selected_scenes,
                            'script_request': script_request,
                            'all_scenes': all_processed_scenes
                        }
                        
                        st.success("✔ Initial script and scene selection completed!")
                        st.info(f"📋 Selected {len(selected_scenes)} most relevant scenes from {len(all_processed_scenes)} total scenes across {len(st.session_state.processed_videos)} videos")
                        st.balloons()
                        
                    else:
                        st.error(f"❌ Initial script generation failed: {script_result.get('error')}")
                        
                except Exception as e:
                    st.error(f"❌ Script generation error: {str(e)}")
        elif generate_initial_script:
            st.warning("⚠️ Vui lòng nhập mô tả kịch bản trước khi tạo")

    # Display initial script results if available (Phase 1 completed)
    if st.session_state.script_generator_results and st.session_state.script_generator_results.get('phase') == 'initial':
        script_data = st.session_state.script_generator_results
        
        st.success("✔ Scene Selection & Initial Script Results from All Videos")
        
        # Show initial script stats
        col1, col2, col3 = st.columns(3)
        with col1:
            total_available = len(script_data['all_scenes'])
            st.metric("📊 Total Scenes Analyzed", total_available)
        with col2:
            selected_count = len(script_data['selected_scenes'])
            st.metric("🎭 Selected Scenes", selected_count)
        with col3:
            script_length = len(script_data['initial_script']['script'])
            st.metric("📝 Script Length", f"{script_length:,} chars")
        
        # Show which videos contributed to the selected scenes
        selected_videos = set()
        for scene in script_data['selected_scenes']:
            if scene.get('source_video'):
                selected_videos.add(scene['source_video'])
        
        if selected_videos:
            st.info(f"🎬 **Videos contributing to script:** {', '.join(sorted(selected_videos))}")
        
        # Display the selected scenes in detailed format (similar to Complete Scene Analysis)
        with st.expander("📜 Selected Scenes & Initial Script from All Videos", expanded=True):
            # Show the script title first
            initial_script = script_data['initial_script']['script']
            script_lines = initial_script.split('\n')
            
            # Extract and display title
            title_line = script_lines[0] if script_lines else "Generated Script"
            if title_line.startswith('**') and title_line.endswith('**'):
                st.markdown(f"### {title_line}")
            else:
                st.markdown(f"### {title_line}")
            
            st.markdown("---")
            
            # Display each selected scene in detailed format
            selected_scenes = script_data['selected_scenes']
            for scene in selected_scenes:
                scene_id = scene.get('scene_id', 'Unknown')
                timestamp_str = scene.get('timestamp_str', 'Unknown time')
                source_video = scene.get('source_video', 'Unknown video')
                
                st.write(f"**Scene {scene_id}** ({timestamp_str}) - **From: {source_video}**")
                
                # Visual description
                visual_desc = scene.get('visual_description', 'N/A')
                st.write(f"🎭 **Visual**: {visual_desc}")
                
                # Conversation/Audio
                if scene.get('speaker_transcript'):
                    st.write("🎤 **Conversation**:")
                    conversation_lines = scene['speaker_transcript'].split('\n')
                    for line in conversation_lines:
                        if line.strip():
                            st.write(f"  {line}")
                elif scene.get('transcript'):
                    st.write(f"📝 **Audio**: {scene['transcript']}")
                else:
                    st.write("📝 **Audio**: No speech")
                
                # Speakers
                if scene.get('unique_speakers'):
                    speakers_text = ", ".join(scene['unique_speakers'])
                    st.write(f"👥 **Speakers**: {speakers_text}")
                
                # Scene type
                scene_type = scene.get('scene_type', 'video_analyzed')
                st.write(f"🎭 **Type**: {scene_type}")
                
                st.write("---")
            
            # Option to show raw generated script
            with st.expander("📄 View Raw Generated Script", expanded=False):
                st.text_area("Generated Initial Script:", initial_script, height=300, disabled=True, key="raw_initial_script")
        
        st.info("📋 These scenes were selected from across all processed videos and used in the Generated Script.")
        st.success(f"🎭 {len(script_data['selected_scenes'])} scenes from {len(selected_videos)} videos ready for final script")

        # PHASE 2: Final Script with Voiceover (only show after Phase 1 is complete)
        st.divider()
        st.subheader("🎙️ Phase 2: Final Script with Voiceover")
        
        with st.form("final_script_generation_from_all_videos_form"):
            st.markdown("### 🎙️ Voiceover Configuration")
            
            # Display selected request (read-only)
            st.text_input("Script Request:", value=script_data['script_request'], disabled=True)
            
            # Voiceover tone selection
            voiceover_tone = st.selectbox(
                "Chọn tông điệu Voiceover:",
                [
                    "Chuyên nghiệp, truyền cảm hứng",
                    "Thân thiện, dễ hiểu", 
                    "Trang trọng và mang tính giáo dục",
                    "Năng động, thu hút"
                ],
                help="Tông điệu của người dẫn chương trình sẽ ảnh hưởng đến cách kể chuyện"
            )
            
            # Show selected scenes info
            selected_scenes = script_data['selected_scenes']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Selected Scenes", len(selected_scenes))
            with col2:
                speech_scenes = len([s for s in selected_scenes if s.get('has_speech', False)])
                st.metric("🎤 Scenes with Speech", speech_scenes)
            with col3:
                st.metric("🎭 Ready for Voiceover", "✔ Yes")
            
            # Generate final script button
            generate_final_script = st.form_submit_button("🎙️ Generate Final Script with Voiceover", type="primary")

            if generate_final_script:
                with st.spinner("🔄 Generating final script with voiceover..."):
                    try:
                        script_generator = DirectScriptGenerator()
                        
                        # Generate final script with voiceover
                        final_result = script_generator.generate_final_script_with_voiceover(
                            selected_scenes,
                            script_data['script_request'],
                            voiceover_tone
                        )
                        
                        if final_result.get("success"):
                            # Update results to Phase 2
                            st.session_state.script_generator_results.update({
                                'phase': 'final',
                                'final_script': final_result,
                                'voiceover_tone': voiceover_tone
                            })
                            
                            st.success("✔ Final script with voiceover generated successfully!")
                            st.balloons()
                            
                        else:
                            st.error(f"❌ Final script generation failed: {final_result.get('error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Final script generation error: {str(e)}")

    # Display final script results if available (Phase 2 completed)
    if st.session_state.script_generator_results and st.session_state.script_generator_results.get('phase') == 'final':
        script_data = st.session_state.script_generator_results
        
        st.divider()
        # Enhanced success message with better styling
        st.markdown('<div class="success-header">✔ Final Script with Voiceover - Complete Output</div>', unsafe_allow_html=True)
        st.markdown("")  # Add spacing
        
        # Show final script stats
        col1, col2, col3 = st.columns(3)
        with col1:
            total_acts = script_data['final_script'].get('total_acts', 0)
            st.metric("📊 Total Acts", total_acts)
        with col2:
            scenes_used = script_data['final_script'].get('total_scenes_used', 0)
            st.metric("🎬 Scenes Used", scenes_used)
        with col3:
            script_length = len(script_data['final_script'].get('final_script', ''))
            st.metric("📝 Script Length", f"{script_length:,} chars")
        
        # Display the final script in detailed format (similar to Complete Scene Analysis)
        with st.expander("📜 Final Script with Voiceover - Complete Output", expanded=True):
            final_script = script_data['final_script'].get('final_script', '')
            
            # Parse and display the final script in scene format
            script_lines = final_script.split('\n')
            
            # Extract and display title
            title_line = script_lines[0] if script_lines else "Generated Final Script"
            if title_line.startswith('**') and title_line.endswith('**'):
                st.markdown(f"### {title_line}")
            else:
                st.markdown(f"### {title_line}")
            
            st.markdown("---")
            
            # NEW: Enhanced parsing for scene-level conversation and act-level voiceover
            current_act = ""
            current_scene = ""
            current_content = []
            scene_counter = 1
            in_conversation = False
            
            # Get selected scenes with source info for reference
            selected_scenes_dict = {scene.get('scene_id', 0): scene for scene in script_data['selected_scenes']}
            
            for line in script_lines[1:]:  # Skip title line
                line = line.strip()
                
                if line.startswith('Cảnh ') or line.startswith('**Cảnh '):
                    # New Act detected - display previous act if exists
                    if current_act and current_content:
                        st.write(f"## **{current_act}**")
                        for content_line in current_content:
                            st.write(content_line)
                        st.write("---")
                    
                    # Clean and format act title
                    current_act = line.replace('**', '').strip()
                    if not current_act.startswith('Cảnh'):
                        current_act = f"Cảnh {scene_counter}: {current_act}"
                        scene_counter += 1
                    
                    current_content = []
                    in_conversation = False
                    
                elif line.startswith('Scene '):
                    # Parse scene with enhanced source info display
                    current_scene = line
                    if current_scene:
                        current_content.append(f"**{current_scene}**")
                    in_conversation = False
                        
                elif line.startswith('🎭 Visual:'):
                    current_content.append(line)
                    
                elif line.startswith('🎤 Conversation:'):
                    current_content.append("**🎤 Conversation:**")
                    in_conversation = True
                    
                elif line.startswith('👥 Speakers:'):
                    current_content.append(line)
                    in_conversation = False
                    
                elif line.startswith('🎙️ Voiceover cho Cảnh') or line.startswith('🎙️ Voiceover:'):
                    # Act-level voiceover
                    current_content.append(f"**{line}**")
                    in_conversation = False
                    
                elif line and not line.startswith('**'):
                    # Regular content lines
                    if line.strip():
                        if in_conversation:
                            # Format conversation lines properly
                            current_content.append(f"{line}")
                        else:
                            current_content.append(f"  {line}")
            
            # Display last act with enhanced formatting
            if current_act and current_content:
                st.write(f"## **{current_act}**")
                for content_line in current_content:
                    st.write(content_line)
                st.write("---")
        
        # NEW: Export to JSON functionality
        st.markdown("---")
        st.markdown("### 📥 Export Script to JSON")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📋 Export Script as Separate Act JSON Files", type="primary", use_container_width=True):
                with st.spinner("🔄 Converting script to JSON format and creating files..."):
                    try:
                        # Parse the final script to separate acts
                        final_script_text = script_data['final_script'].get('final_script', '')
                        selected_scenes = script_data['selected_scenes']
                        
                        script_title, acts_data = parse_script_to_acts(final_script_text, selected_scenes)
                        
                        if acts_data:
                            folder_path, created_files = create_acts_folder_and_files(script_title, acts_data)
                            
                            st.success(f"✔ Successfully created script export!")
                            st.info(f"📊 Generated {len(created_files)} act files in folder: `{os.path.basename(folder_path)}`")
                            
                            # Show created files
                            st.markdown("#### � Created Files:")
                            for filename in created_files:
                                st.markdown(f"- 📄 `{filename}`")
                            
                            st.markdown(f"**📍 Export Location:** `{folder_path}`")
                            
                            # Create ZIP file for download
                            import zipfile
                            zip_filename = f"{os.path.basename(folder_path)}.zip"
                            zip_path = os.path.join('data/outputs', zip_filename)
                            
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for filename in created_files:
                                    file_path = os.path.join(folder_path, filename)
                                    zipf.write(file_path, filename)
                            
                            # Download button
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label="💾 Download All Act Files (ZIP)",
                                    data=f.read(),
                                    file_name=zip_filename,
                                    mime="application/zip",
                                    use_container_width=True
                                )
                            
                            # Show sample act structure
                            with st.expander("👀 Sample Act JSON Structure", expanded=False):
                                if acts_data:
                                    first_act = list(acts_data.values())[0]
                                    st.json(first_act)
                            
                        else:
                            st.error("❌ Failed to parse script into acts")
                            
                    except Exception as e:
                        st.error(f"❌ Export error: {str(e)}")
                        st.info("💡 Please ensure the script has been generated successfully before exporting")

# NEW: Cleanup option with unique key
if st.session_state.validation_result:
    if st.button("🗑️ Clean up temporary files", key="cleanup_button_unique"):
        processor = VideoProcessor()
        processor.cleanup_temp_file(st.session_state.validation_result.get('temp_path'))

        # Clear session state
        st.session_state.validation_result = None
        st.session_state.scene_results = None
        st.session_state.transcription_results = None
        st.session_state.visual_analysis_results = None
        st.session_state.script_generator_results = None

        st.success("Temporary files cleaned up!")
        st.rerun()

# Show supported formats
st.markdown("### 📋 Supported Formats")
cols = st.columns(len(SUPPORTED_FORMATS))
for i, format_name in enumerate(SUPPORTED_FORMATS):
    cols[i].info(f"**{format_name.upper()}**")