# -*- coding: utf-8 -*-

# fixed_enhanced_script_generator_ui.py - Streamlit UI for Enhanced Script Generator with Voiceover (FIXED VERSION)

import streamlit as st
import json
from datetime import datetime
from script_generator import DirectScriptGenerator

# Page config
st.set_page_config(
    page_title="Enhanced Script Generator with Voiceover (Fixed)",
    page_icon="🎬",
    layout="wide"
)

def create_sample_scenes_data():
    """Create comprehensive sample scenes data for testing"""
    return [
        {
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": 47.366,
            "visual_description": "Video này trình bày một chuỗi hình ảnh của một nữ sinh viên trẻ Việt Nam trong nhiều bối cảnh khác nhau, chủ yếu là trong khuôn viên trường học và các môi trường học thuật. Cô được thấy đang tạo dáng với các biểu tượng của thành tích học tập như gấu bông tốt nghiệp, bó hoa, và các bằng khen, huy chương.",
            "speaker_transcript": "Speaker A: Một trong 9 thủ khoa khối A00 kỳ thi tốt nghiệp Trung học của Thông năm 2025 với 3.10 tuyệt đối, Nguyễn Lê Hiền Mai với các thành tích học tập đáng nể. Giải nhất kỳ thi học sinh giỏi cấp tỉnh năm học 2021-2022 Mồn Khóa Học được tuyển thẳng vào lớp chuyên hóa.",
            "unique_speakers": ["A"],
            "scene_type": "presentation"
        },
        {
            "scene_id": 6,
            "start_time": 79.766,
            "end_time": 94.0,
            "visual_description": "Video này cho thấy một cuộc phỏng vấn trong một trường quay truyền hình, với một nữ sinh trẻ được phỏng vấn bởi một người đàn ông.",
            "speaker_transcript": "Speaker B: ở tự học toán vật lý hóa học này, lúc này thì chắc chắn với Mai cảm xúc như thế nào về kết quả vừa có đó và được lan toa.",
            "unique_speakers": ["B"],
            "scene_type": "interview"
        },
        {
            "scene_id": 7,
            "start_time": 94.0,
            "end_time": 104.2,
            "visual_description": "Bốn người, bao gồm một phụ nữ và ba người đàn ông, đang ngồi quanh một chiếc bàn lớn trong một trường quay truyền hình sáng sủa. Họ mặc trang phục chuyên nghiệp và đang tham gia vào một cuộc thảo luận.",
            "speaker_transcript": "Speaker B: khán giả truyền hình cũng như là cả ba chúng ta đã cảm thấy rất là ngưỡng mộ với thành tích này của em Mai. Không biết là em có thể chia sẻ một chút là em đã luyện tập như thế nào?",
            "unique_speakers": ["B"],
            "scene_type": "interview"
        },
        {
            "scene_id": 8,
            "start_time": 104.2,
            "end_time": 120.066,
            "visual_description": "Video này hiển thị bốn người, gồm một nữ sinh và ba nam giới, đang ngồi quanh một bàn tròn lớn màu trắng trong một trường quay truyền hình. Nữ sinh mặc áo sơ mi trắng đang phát biểu vào micro.",
            "speaker_transcript": "Speaker C: Kết Quả như vậy? Thưa ba anh, với các môn khoa tự nhiên như các môn khối A00 của em là toán vật lý và hóa học thì với em, em nghĩ đó là ba môn mà yêu cầu tính logic rất là cao, cần tư duy rất là nhiều. Chính vì vậy",
            "unique_speakers": ["C"],
            "scene_type": "qa_session"
        },
        {
            "scene_id": 9,
            "start_time": 120.066,
            "end_time": 125.866,
            "visual_description": "Video quay cảnh năm người, một phụ nữ và bốn người đàn ông, đang ngồi xung quanh một chiếc bàn lớn hình bầu dục trong một trường quay truyền hình hiện đại.",
            "speaker_transcript": "Speaker C: em đã cố gắng để nắm bắt được bài trên lớp trực tiếp từ các thầy cô. Song song với đó",
            "unique_speakers": ["C"],
            "scene_type": "qa_session"
        },
        {
            "scene_id": 10,
            "start_time": 125.866,
            "end_time": 138.266,
            "visual_description": "Đoạn video ghi lại cảnh một nữ sinh đang trò chuyện cùng một nam MC trong chương trình 'Chào Buổi Sáng' của VTV1.",
            "speaker_transcript": "Speaker C: là em có tìm hiểu thêm một số những dạng bài tập trên mạng. Em nghĩ đấy là những yếu tố cần thiết để bản thân mình có thể vận dụng được thứ nhất là tư duy của mình, thứ hai là lý thuyết một cách",
            "unique_speakers": ["C"],
            "scene_type": "results"
        },
        {
            "scene_id": 11,
            "start_time": 138.266,
            "end_time": 153.766,
            "visual_description": "Video cho thấy một cuộc thảo luận hoặc phỏng vấn trong một trường quay truyền hình. Một nữ sinh trẻ đang ngồi cùng ba người đàn ông.",
            "speaker_transcript": "Speaker C: tư duy của mình, thứ hai là lý thuyết một cách thực tiễn và hợp lý nhất có thể để áp dụng được vào bài thi. Vậy thì.\nSpeaker B: Anh đang rất là tự đặt câu hỏi là không hiểu còn một cái bí quyết gì nữa không bên cạnh những cái mà em đã vừa nói đó cũng cơ bản thôi đúng không ạ?",
            "unique_speakers": ["C", "B"],
            "scene_type": "qa_session"
        },
        {
            "scene_id": 12,
            "start_time": 153.766,
            "end_time": 157.866,
            "visual_description": "Video ghi lại một chương trình truyền hình trực tiếp từ trường quay của VTV1, trong đó có bốn người, bao gồm ba nam và một nữ, ngồi quanh một bàn lớn.",
            "speaker_transcript": "Speaker B: gì chăng ở đây không để được 3.10 đúng không ạ? Thực ra thì.\nSpeaker C: Với em thì bí quyết thì cũng không có bí quyết gì cả.",
            "unique_speakers": ["B", "C"],
            "scene_type": "qa_session"
        }
    ]

def display_scenes_preview(scenes_data):
    """Display preview of scenes data with enhanced format"""
    with st.expander("🎬 Complete Scene Analysis Preview", expanded=False):
        # Sort scenes by scene_id for consistent display
        all_scenes = sorted(scenes_data, key=lambda x: x.get('scene_id', 0))

        # Display all scenes
        for scene in all_scenes:
            scene_id = scene.get('scene_id', 'Unknown')
            start_time = scene.get('start_time', 0)
            end_time = scene.get('end_time', 0)

            # Format timestamp string
            start_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}:{int((start_time%1)*1000):03d}"
            end_str = f"{int(end_time//60):02d}:{int(end_time%60):02d}:{int((end_time%1)*1000):03d}"
            timestamp_str = f"{start_str} - {end_str}"

            st.write(f"**Scene {scene_id}** ({timestamp_str}):")
            st.write(f"🎭 **Visual**: {scene.get('visual_description', 'N/A')}")

            # Enhanced speaker transcript display
            if scene.get('speaker_transcript'):
                st.write(f"🎤 **Conversation**:")
                for line in scene['speaker_transcript'].split('\n'):
                    if line.strip():
                        st.write(f"  {line}")

                if scene.get('unique_speakers'):
                    speakers_text = ", ".join(scene['unique_speakers'])
                    st.write(f"👥 **Speakers**: {speakers_text}")
            else:
                st.write(f"📝 **Audio**: No speech")

            st.write(f"🎯 **Type**: {scene.get('scene_type', 'video_analyzed')}")
            st.write("---")

def display_script_with_formatting(script_text):
    """Display script with proper formatting"""
    script_lines = script_text.split('\n')

    for line in script_lines:
        if line.strip():
            # Format different parts of the script
            if line.startswith('**') and line.endswith('**'):
                # Title
                st.markdown(f"## {line}")
            elif line.startswith('Cảnh '):
                # Scene header
                st.markdown(f"### {line}")
            elif line.startswith('Scene '):
                # Scene timestamp
                st.markdown(f"*{line}*")
            elif line.startswith('🎭'):
                # Visual description
                st.markdown(f"**{line}**")
            elif line.startswith('🎤'):
                # Conversation header
                st.markdown(f"**{line}**")
            elif line.startswith('👥'):
                # Speakers
                st.markdown(f"**{line}**")
            elif line.startswith('Voiceover:'):
                # Voiceover
                st.markdown(f"🎙️ **{line}**")
                st.markdown("---")  # Add separator after voiceover
            elif line.strip().startswith('Speaker '):
                # Speaker lines
                st.markdown(f"  {line}")
            else:
                # Regular lines
                st.write(line)
        else:
            # Empty lines for spacing
            st.write("")

def main():
    st.title("🎬 Enhanced Script Generator with Voiceover (FIXED)")
    st.markdown("*Advanced script generation with correct scene extraction for voiceover*")

    # Initialize session state
    if 'selected_scenes' not in st.session_state:
        st.session_state.selected_scenes = []
    if 'generated_script' not in st.session_state:
        st.session_state.generated_script = ""
    if 'all_scenes_data' not in st.session_state:
        st.session_state.all_scenes_data = []
    if 'final_script_result' not in st.session_state:
        st.session_state.final_script_result = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Test mode selection
        test_mode = st.selectbox(
            "Select Data Source",
            ["Sample Data (Lê Hiền Mai)", "Upload JSON", "Custom Input"]
        )

        st.divider()

        # Generator settings preview
        st.subheader("🤖 AI Settings")
        st.write("**Gemini AI (Scene Selection):**")
        st.write("- Temperature: 0.7")
        st.write("- Max Tokens: none")
        st.write("")
        st.write("**Claude 3.5 Sonnet (Voiceover):**")
        st.write("- Temperature: 0.7")
        st.write("- Max Tokens: 1000000")

    # Create tabs for different stages
    tab1, tab2, tab3 = st.tabs(["📋 Scene Selection", "📝 Script Generation", "🎤 Final Script with Voiceover"])

    with tab1:
        st.header("📋 Step 1: Scene Selection & Initial Script")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Input Configuration")

            # User request input
            user_request = st.text_area(
                "Your Script Request",
                value="tạo cho tôi một kịch bản highlight những phương pháp học của Lê Hiền Mai",
                height=100,
                help="Nhập yêu cầu của bạn cho kịch bản",
                key="user_request_tab1"
            )

            # Scenes data configuration
            scenes_data = []
            if test_mode == "Sample Data (Lê Hiền Mai)":
                st.info("🎭 Using sample scenes data from Lê Hiền Mai interview")
                scenes_data = create_sample_scenes_data()

            elif test_mode == "Upload JSON":
                uploaded_file = st.file_uploader(
                    "Upload scenes JSON file",
                    type=['json'],
                    help="Upload a JSON file containing scenes data"
                )

                if uploaded_file:
                    try:
                        scenes_data = json.load(uploaded_file)
                        st.success(f"✅ Loaded {len(scenes_data)} scenes from file")
                    except Exception as e:
                        st.error(f"❌ Error loading JSON: {e}")

            # Store all scenes data in session state
            if scenes_data:
                st.session_state.all_scenes_data = scenes_data

            # Display scenes preview
            if scenes_data:
                st.subheader("📋 Available Scenes Data")
                st.info(f"📊 **Total Scenes:** {len(scenes_data)} scenes loaded successfully")
                display_scenes_preview(scenes_data)

        with col2:
            st.subheader("🎬 Generate Initial Script")

            # Generate initial script button
            if st.button("🚀 Generate Script from All Scenes", type="primary", disabled=not scenes_data or not user_request):
                if not scenes_data:
                    st.error("❌ No scenes data provided!")
                elif not user_request:
                    st.error("❌ Please enter a user request!")
                else:
                    # Initialize generator
                    with st.spinner("Initializing Script Generator..."):
                        generator = DirectScriptGenerator()

                    if generator.model:
                        # Generate script
                        start_time = datetime.now()
                        result = generator.generate_script_from_scenes(user_request, scenes_data)
                        end_time = datetime.now()
                        generation_time = (end_time - start_time).total_seconds()

                        # Display results
                        if result.get("success"):
                            st.success(f"✅ Script generated successfully in {generation_time:.2f} seconds!")

                            # Store results in session state
                            st.session_state.generated_script = result["script"]

                            # 🔧 FIX: Extract only the scenes actually used in the generated script
                            st.session_state.selected_scenes = generator.extract_selected_scenes_from_script(
                                result["script"], 
                                scenes_data
                            )

                            # Display selected scenes info
                            st.info(f"🎯 **Scenes Selected for Script:** {len(st.session_state.selected_scenes)} out of {len(scenes_data)} total scenes")

                            # Display generated script
                            with st.expander("🎬 Generated Script Preview", expanded=True):
                                display_script_with_formatting(result["script"])

                            # Statistics
                            stats = generator.get_generation_stats(result)
                            if stats.get("success"):
                                col3, col4, col5 = st.columns(3)
                                with col3:
                                    st.metric("📋 Total Scenes", stats["total_scenes_available"])
                                with col4:
                                    st.metric("✅ Scenes Used", len(st.session_state.selected_scenes))
                                with col5:
                                    st.metric("⏱️ Gen Time", f"{generation_time:.2f}s")
                        else:
                            st.error(f"❌ Script generation failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Failed to initialize Script Generator. Check your API configuration.")

    with tab2:
        st.header("📝 Step 2: Review Generated Script")

        if st.session_state.generated_script:
            st.success("✅ Script generated successfully! Review below:")

            # Show selected scenes info
            if st.session_state.selected_scenes:
                st.info(f"🎯 **Selected Scenes for Final Script:** {len(st.session_state.selected_scenes)} scenes")

                with st.expander("📋 Selected Scenes Details", expanded=False):
                    for scene in st.session_state.selected_scenes:
                        scene_id = scene.get('scene_id', 'Unknown')
                        timestamp = f"{scene.get('start_time', 0):.1f}s - {scene.get('end_time', 0):.1f}s"
                        st.write(f"**Scene {scene_id}** ({timestamp})")
                        st.write(f"🎭 {scene.get('visual_description', 'N/A')[:100]}...")
                        st.write("---")

            with st.expander("🎬 Generated Script - Complete Output", expanded=True):
                display_script_with_formatting(st.session_state.generated_script)

            # Download option for initial script
            script_content = f"""# Generated Video Script (Initial)

## 📋 Request: {user_request}
## 🕐 Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{st.session_state.generated_script}

---
Generated by Enhanced Script Generator with Gemini AI
"""

            st.download_button(
                label="📄 Download Initial Script",
                data=script_content,
                file_name=f"initial_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                type="secondary"
            )

        else:
            st.info("🔄 Please generate a script in Step 1 first.")

    with tab3:
        st.header("🎤 Step 3: Final Script with Voiceover")

        if st.session_state.selected_scenes:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🎙️ Voiceover Configuration")

                # Script request (can be different from initial)
                final_script_request = st.text_area(
                    "Final Script Request",
                    value=user_request if 'user_request' in locals() else "tạo cho tôi một kịch bản highlight những phương pháp học của Lê Hiền Mai",
                    height=80,
                    help="Có thể điều chỉnh lại yêu cầu cho kịch bản cuối cùng",
                    key="final_script_request"
                )

                # NEW FEATURE: Voiceover Tone Input
                voiceover_tone = st.text_area(
                    "Voiceover Tone",
                    value="Chuyên nghiệp, truyền cảm hứng, trang trọng và mang tính giáo dục",
                    height=80,
                    help="Mô tả tông điệu cho voiceover của từng cảnh",
                    key="voiceover_tone"
                )

                # 🔧 FIXED: Show correct number of selected scenes
                st.success(f"📊 **Selected Scenes:** {len(st.session_state.selected_scenes)} scenes ready for final script")
                st.info("✅ These are the scenes actually used in the Generated Script, not all scenes from the JSON file.")

                # Generate final script button
                if st.button("🎬 Generate Final Script with Voiceover", type="primary"):
                    with st.spinner("🎭 Generating final script with voiceover... This may take a moment."):
                        generator = DirectScriptGenerator()

                        result = generator.generate_final_script_with_voiceover(
                            selected_scenes=st.session_state.selected_scenes,
                            script_request=final_script_request,
                            voiceover_tone=voiceover_tone
                        )

                        if result.get("success"):
                            st.session_state.final_script_result = result
                            st.success("✅ Final script with voiceover generated successfully!")
                        else:
                            st.error(f"❌ Final script generation failed: {result.get('error', 'Unknown error')}")

            with col2:
                st.subheader("🎬 Final Script Output")

                if st.session_state.final_script_result:
                    result = st.session_state.final_script_result

                    # Statistics
                    stats = generator.get_final_script_stats(result)
                    if stats.get("success"):
                        col3, col4, col5 = st.columns(3)
                        with col3:
                            st.metric("🎭 Total Acts", stats["total_acts"])
                        with col4:
                            st.metric("📋 Scenes Used", stats["total_scenes_used"])
                        with col5:
                            st.metric("📄 Script Length", f"{stats['final_script_length']:,} chars")

                    # Display final script
                    with st.expander("🎬 Final Script with Voiceover - Complete Output", expanded=True):
                        display_script_with_formatting(result["final_script"])

                    # Enhanced download options
                    st.subheader("💾 Download Options")

                    final_script_content = f"""# Final Video Script with Voiceover

## 📋 Request: {final_script_request}
## 🎙️ Voiceover Tone: {voiceover_tone}
## 🕐 Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{result["final_script"]}

---

## 📊 Generation Statistics:
- **Total Acts:** {stats.get("total_acts", "N/A")}
- **Total Scenes Used:** {stats.get("total_scenes_used", "N/A")}
- **Script Length:** {stats.get("final_script_length", "N/A"):,} characters
- **Voiceover Tone:** {stats.get("voiceover_tone", "N/A")}

Generated by Enhanced Script Generator with Gemini AI + Claude 3.5 Sonnet
"""

                    col_md, col_txt = st.columns(2)

                    with col_md:
                        st.download_button(
                            label="📄 Download as Markdown",
                            data=final_script_content,
                            file_name=f"final_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            type="primary"
                        )

                    with col_txt:
                        st.download_button(
                            label="📝 Download as Text",
                            data=final_script_content,
                            file_name=f"final_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            type="secondary"
                        )

                else:
                    st.info("🎙️ Generate final script with voiceover to see results here.")

        else:
            st.info("🔄 Please generate a script in Step 1 first to proceed with final script generation.")

    # Help section
    with st.expander("ℹ️ How to Use This Enhanced Generator (FIXED VERSION)", expanded=False):
        st.markdown("""
        ### 🚀 Complete Workflow:

        **Step 1: Scene Selection & Initial Script**
        1. Choose your data source (Sample Data, Upload JSON, or Custom Input)
        2. Enter your script request
        3. Generate initial script - Gemini AI will analyze ALL scenes and select the best ones
        4. ✅ **FIXED:** System now correctly extracts only scenes used in Generated Script

        **Step 2: Review Generated Script**
        1. Review the initial script generated by Gemini AI
        2. See exactly which scenes were selected (not all scenes from JSON)
        3. Download if satisfied, or proceed to Step 3 for enhanced version

        **Step 3: Final Script with Voiceover**
        1. Adjust the final script request if needed
        2. **NEW**: Enter desired voiceover tone (professional, inspiring, educational, etc.)
        3. Generate final script - Claude 3.5 Sonnet will:
           - Group **only selected scenes** into acts
           - Create seamless voiceover for each act
           - Format according to your specified style

        ### 🔧 What Was Fixed:
        - **Scene Selection Logic:** Now correctly uses only scenes from Generated Script
        - **Accurate Counting:** Shows correct number of selected scenes
        - **Better Extraction:** Parses Generated Script to find actual scene IDs used
        - **Debug Information:** Shows which scenes were extracted for transparency

        ### 📋 Expected Behavior:
        - **Input:** 12 scenes from JSON file
        - **Gemini Selection:** 5-8 most relevant scenes for Generated Script
        - **Final Script:** Uses only the 5-8 selected scenes, not all 12
        - **Voiceover:** Creates acts based on selected scenes only
        """)

if __name__ == "__main__":
    main()
