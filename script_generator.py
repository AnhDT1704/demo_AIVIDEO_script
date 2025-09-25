import streamlit as st
import google.generativeai as genai
import requests
import logging  # Add missing import
from typing import Dict, List, Any, Optional
import json
import re
from config import GOOGLE_AI_SCRIPT_API_KEY, GEMINI_MODEL, CLAUDE_API_KEY

class DirectScriptGenerator:
    """Enhanced script generation with missing scene issue resolved"""

    def __init__(self):
        """Initialize AI client using script-specific API key"""
        try:
            genai.configure(api_key=GOOGLE_AI_SCRIPT_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            # Fix: Store model as client for consistency
            self.client = self.model
            # API key for Advanced AI (OpenRouter)
            self.openrouter_api_key = CLAUDE_API_KEY
            self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
            self.claude_model = "anthropic/claude-3.5-sonnet"

            # st.success("✔ Script Generator initialized with Advanced AI")  # ✅ HIDDEN
        except Exception as e:
            st.error(f"❌ Advanced AI initialization failed: {e}")
            self.model = None
            self.client = None

    def generate_script_from_scenes(self, user_request: str, all_scenes: List[Dict]) -> Dict[str, Any]:
        """Direct script generation - send ALL scenes to AI"""
        try:
            if not self.model:
                return {"success": False, "error": "Advanced AI not available"}

            if not all_scenes:
                return {"success": False, "error": "No scenes data provided"}

            # Build comprehensive prompt with ALL scene data
            prompt = self._build_comprehensive_prompt(user_request, all_scenes)

            # Generate with AI using long context
            with st.spinner("🤖 Advanced AI đang phân tích tất cả scenes và tạo kịch bản..."):
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=8000,
                        top_p=0.8,
                        top_k=40
                    )
                )

            if response and response.text:
                return {
                    "success": True,
                    "script": response.text,
                    "scenes_processed": len(all_scenes),
                    "user_request": user_request,
                    "all_scenes": all_scenes
                }
            else:
                return {"success": False, "error": "No response from Advanced AI"}

        except Exception as e:
            return {"success": False, "error": f"Script generation failed: {str(e)}"}

    def extract_selected_scenes_from_script(self, generated_script: str, all_scenes: List[Dict]) -> List[Dict]:
        """Trích xuất scenes từ generated script với improved pattern matching"""
        selected_scenes = []

        if not generated_script or not all_scenes:
            st.warning("⚠️ No script or scenes data provided for extraction")
            return selected_scenes

        try:
            script_lines = generated_script.split('\n')
            used_scene_ids = set()
            
            # 🔍 DEBUG: Log input data
            available_scene_ids = [scene.get('scene_id', 0) for scene in all_scenes]
            # st.info(f"🔍 EXTRACT DEBUG: Available scene IDs from Complete Analysis: {sorted(available_scene_ids)}")  # ✅ HIDDEN
            
            # Improved pattern matching for scene references
            patterns = [
                r'Scene (\d+)[\s:)]',  # "Scene 1:", "Scene 1 ", "Scene 1)"
                r'scene (\d+)[\s:)]',  # lowercase version
                r'Cảnh (\d+)[\s:)]',   # Vietnamese "Cảnh"
                r'(\d+)[\s]*:[\s]*\(',  # "1: (timestamp)"
            ]
            
            for line in script_lines:
                line = line.strip()
                
                # Try multiple patterns
                for pattern in patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        try:
                            scene_id = int(match)
                            if scene_id in available_scene_ids:  # Only add if scene exists
                                used_scene_ids.add(scene_id)
                        except ValueError:
                            continue

            # 🔍 DEBUG: Log extraction results
            # st.info(f"🔍 EXTRACT DEBUG: Found scene IDs in script: {sorted(used_scene_ids)}")  # ✅ HIDDEN
            
            # Match scenes from available data
            for scene in all_scenes:
                scene_id = scene.get('scene_id', 0)
                if scene_id in used_scene_ids:
                    selected_scenes.append(scene)

            selected_scenes.sort(key=lambda x: x.get('scene_id', 0))
            
            # 🔍 FINAL DEBUG
            # st.info(f"🎭 EXTRACT RESULT: Selected {len(selected_scenes)}/{len(all_scenes)} scenes")  # ✅ HIDDEN
            selected_ids = [s.get('scene_id', 0) for s in selected_scenes]
            # st.info(f"🎭 Final selected IDs: {sorted(selected_ids)}")  # ✅ HIDDEN

            # If no scenes extracted, return a reasonable subset
            if not selected_scenes and all_scenes:
                st.warning("⚠️ No scenes matched pattern, using intelligent fallback selection")
                # Take first 6-8 scenes as fallback
                fallback_count = min(8, len(all_scenes))
                selected_scenes = all_scenes[:fallback_count]
                # st.info(f"🔄 Fallback: Selected first {len(selected_scenes)} scenes")  # ✅ HIDDEN

            return selected_scenes

        except Exception as e:
            st.error(f"❌ Error extracting scenes from script: {e}")
            # Return first half of scenes as emergency fallback
            fallback_count = min(6, len(all_scenes))
            return all_scenes[:fallback_count]

    
    def select_best_scenes_from_all_videos(self, script_request: str, multi_video_results: Dict) -> List[Dict]:
        """
        🎯 NEW: Cross-video scene selection
        Select best scenes from ALL videos based on script requirements
        """
        # Prepare all scenes with source tracking
        all_scenes_with_source = self.prepare_scenes_for_cross_video_analysis(multi_video_results)

        if not all_scenes_with_source:
            st.error("❌ No scenes available from any videos")
            return []

        # st.info(f"🔍 Analyzing {len(all_scenes_with_source)} scenes from {len(multi_video_results)} videos...")  # ✅ HIDDEN

        # Create comprehensive scene data for Claude
        scenes_summary = []
        for scene in all_scenes_with_source:
            summary = {
                'identifier': scene['scene_identifier'],
                'source': scene['source_video'],
                'timeline': scene.get('timestamp_str', 'Unknown'),
                'visual_preview': scene.get('visual_description', '')[:200],
                'conversation_preview': scene.get('speaker_transcript', '')[:300],
                'speakers': scene.get('unique_speakers', [])
            }
            scenes_summary.append(summary)

        # Build cross-video selection prompt
        scenes_text = "\n\n".join([
            f"SCENE: {s['identifier']}\n"
            f"SOURCE VIDEO: {s['source']}\n"
            f"TIMELINE: {s['timeline']}\n"
            f"VISUAL: {s['visual_preview']}...\n"
            f"CONVERSATION: {s['conversation_preview']}...\n"
            f"SPEAKERS: {', '.join(s['speakers']) if s['speakers'] else 'None'}"
            for s in scenes_summary
        ])

        # UPDATED: Generic prompt for all video types
        prompt = f"""
Bạn là chuyên gia chọn lọc scenes từ nhiều video để tạo kịch bản tối ưu.

**YÊU CẦU KỊCH BẢN:** {script_request}

**SCENES AVAILABLE FROM ALL VIDEOS:**
{scenes_text}

**SELECTION CRITERIA:**
1. **Relevance:** Chọn scenes có nội dung trực tiếp liên quan đến yêu cầu kịch bản
2. **Quality:** Ưu tiên scenes có nội dung rõ ràng và phong phú
3. **Diversity:** Chọn scenes từ nhiều videos khác nhau để đa dạng hóa
4. **Flow:** Đảm bảo các scenes được chọn có thể kết nối logic với nhau
5. **Completeness:** Đủ scenes để tạo nội dung hoàn chỉnh theo yêu cầu

**HƯỚNG DẪN CHỌN:**
- Tối thiểu 3 scenes, tối đa 15 scenes
- Ưu tiên chất lượng hơn số lượng
- Có thể chọn scenes từ cùng video nếu nội dung phù hợp
- Sắp xếp theo logic phù hợp với yêu cầu kịch bản

**OUTPUT FORMAT JSON:**
{{
  "selected_scenes": [
    {{
      "scene_identifier": "video1.mp4_scene_2",
      "relevance_score": 95,
      "reasoning": "Scene này phù hợp với yêu cầu kịch bản",
      "suggested_order": 1
    }},
    {{
      "scene_identifier": "video2.mp4_scene_1", 
      "relevance_score": 88,
      "reasoning": "Cung cấp thông tin bổ sung cho nội dung",
      "suggested_order": 2
    }}
  ],
  "selection_summary": "Đã chọn X scenes từ Y videos để tạo kịch bản theo yêu cầu"
}}

CHỈ TRẢ VỀ JSON HỢP LỆ, KHÔNG GIẢI THÍCH THÊM.
"""

        try:
            response = self._call_claude_api(prompt, max_tokens=2000)
            result = json.loads(response)

            selected_identifiers = [s['scene_identifier'] for s in result.get('selected_scenes', [])]
            # st.info(f"🎯 Advanced AI selected {len(selected_identifiers)} scenes: {selected_identifiers}")  # ✅ HIDDEN

            # Map back to actual scene data with source info
            selected_scenes = []
            for identifier in selected_identifiers:
                for scene in all_scenes_with_source:
                    if scene['scene_identifier'] == identifier:
                        selected_scenes.append(scene)
                        break

            if selected_scenes:
                # st.success(f"✅ Successfully selected {len(selected_scenes)} scenes from {len(set(s['source_video'] for s in selected_scenes))} different videos")  # ✅ HIDDEN

                # Show selection summary
                with st.expander("🎯 Selected Scenes Summary", expanded=True):
                    for i, scene in enumerate(selected_scenes, 1):
                        st.write(f"**{i}. Scene from {scene['source_video']}**")
                        st.write(f"   📹 {scene.get('visual_description', 'No description')[:100]}...")
                        st.write(f"   🗣️ {scene.get('speaker_transcript', 'No transcript')[:100]}...")
                        st.write("---")

            return selected_scenes

        except json.JSONDecodeError as e:
            st.error(f"🚨 Scene selection JSON parsing failed: {e}")
            # Fallback: return top scenes from each video
            return self._fallback_cross_video_selection(all_scenes_with_source, max_scenes=10)

        except Exception as e:
            st.error(f"🚨 Scene selection failed: {e}")
            return self._fallback_cross_video_selection(all_scenes_with_source, max_scenes=8)

    def prepare_scenes_for_cross_video_analysis(self, multi_video_results: Dict) -> List[Dict]:
        """
        🔄 Prepare scenes from multiple videos for cross-video analysis
        """
        all_scenes = []

        for filename, results in multi_video_results.items():
            if results.get('visual_analysis') and results['status'] == 'completed':
                for i, scene in enumerate(results['visual_analysis']):
                    scene_with_source = scene.copy()
                    scene_with_source['source_video'] = filename
                    scene_with_source['source_scene_index'] = i
                    scene_with_source['scene_identifier'] = f"{filename}_scene_{i+1}"

                    # Ensure required fields exist
                    scene_with_source.setdefault('visual_description', 'No visual description available')
                    scene_with_source.setdefault('speaker_transcript', 'No transcript available')
                    scene_with_source.setdefault('unique_speakers', [])
                    scene_with_source.setdefault('timestamp_str', 'Unknown timeline')

                    all_scenes.append(scene_with_source)

        # st.info(f"📊 Prepared {len(all_scenes)} scenes from {len(multi_video_results)} videos")  # ✅ HIDDEN
        return all_scenes

    def _fallback_cross_video_selection(self, all_scenes_with_source: List[Dict], max_scenes: int = 8) -> List[Dict]:
        """
        🛡️ Fallback cross-video scene selection when AI fails
        """
        st.warning("🔄 Using fallback cross-video scene selection")

        if not all_scenes_with_source:
            return []

        # Strategy: Pick best scenes from each video proportionally
        videos = {}
        for scene in all_scenes_with_source:
            video = scene['source_video']
            if video not in videos:
                videos[video] = []
            videos[video].append(scene)

        # Select scenes proportionally from each video
        scenes_per_video = max(1, max_scenes // len(videos))
        selected_scenes = []

        for video, scenes in videos.items():
            # Take first N scenes from each video (could be improved with scoring)
            selected_from_video = scenes[:scenes_per_video]
            selected_scenes.extend(selected_from_video)

            if len(selected_scenes) >= max_scenes:
                break

        # Limit to max_scenes
        selected_scenes = selected_scenes[:max_scenes]

        # st.info(f"🎯 Fallback selected {len(selected_scenes)} scenes from {len(videos)} videos")  # ✅ HIDDEN
        return selected_scenes

    def generate_final_script_with_voiceover(self, selected_scenes, script_request, voiceover_tone):
        """
        Generate final structured script with voiceover from selected scenes
        """
        try:
            # Fix: Check if client is available
            if not self.client:
                return {"success": False, "error": "Advanced AI client not available"}
            
            # Enhanced prompt for better script structure with scene-level conversation display
            prompt = f"""
Bạn là một chuyên gia viết kịch bản video chuyên nghiệp. 

NHIỆM VỤ: Tạo kịch bản final có cấu trúc rõ ràng với voiceover cấp độ CẢNH từ các scene đã chọn.

YÊU CẦU SCRIPT REQUEST: {script_request}
TÔNG ĐIỆU VOICEOVER: {voiceover_tone}

SCENES ĐƯỢC CHỌN:
{self._format_scenes_for_prompt(selected_scenes)}

ĐỊNH DẠNG XUẤT RA CHÍNH XÁC:
**Tiêu đề kịch bản**

Cảnh 1: [Tên cảnh phù hợp với nội dung]

Scene [ID]: ([timestamp]) - From: [video.mp4]
🎭 Visual: [mô tả visual chi tiết]
🎤 Conversation:
[Speaker]: [nội dung conversation đầy đủ]
👥 Speakers: [danh sách speakers]

Scene [ID]: ([timestamp]) - From: [video.mp4]
🎭 Visual: [mô tả visual chi tiết]  
🎤 Conversation:
[Speaker]: [nội dung conversation đầy đủ]
👥 Speakers: [danh sách speakers]

🎙️ Voiceover cho Cảnh 1: [Nội dung voiceover tổng hợp cho toàn bộ cảnh, phù hợp với tông điệu {voiceover_tone}]

Cảnh 2: [Tên cảnh tiếp theo]
[Các scenes trong cảnh 2...]

🎙️ Voiceover cho Cảnh 2: [Nội dung voiceover tổng hợp cho toàn bộ cảnh]

HƯỚNG DẪN QUAN TRỌNG:
1. SẮP XẾP các scene theo logic kịch bản ({script_request})
2. NHÓM scenes có nội dung tương đồng vào cùng 1 cảnh
3. TRONG MỖI CẢNH: Sắp xếp scenes theo thứ tự conversation logic (conversation tiếp diễn)
4. HIỂN THỊ ĐẦY ĐỦ conversation cho từng scene (không được bỏ sót)
5. VOICEOVER chỉ tạo Ở CUỐI MỖI CẢNH (không tạo cho từng scene)
6. Voiceover phải tổng hợp nội dung của TẤT CẢ scenes trong cảnh đó
7. GIỮ NGUYÊN thông tin Scene ID, timestamp, và From: video.mp4
8. ĐẶT TÊN cảnh rõ ràng, có ý nghĩa dựa trên nội dung scenes
9. Tên cảnh phải phù hợp với yêu cầu kịch bản người dùng

CÁCH SẮP XẾP SCENES TRONG CẢNH:
- Scenes có nội dung liên quan được nhóm vào cùng cảnh
- Trong cùng cảnh, sắp xếp theo thứ tự conversation logic
- Đảm bảo conversation có tính liên tục và mạch lạc

Tạo kịch bản hoàn chỉnh ngay với cấu trúc trên:
"""

            # Generate script with enhanced AI
            response = self.client.generate_content(prompt)
            
            if response and response.text:
                # Count acts and scenes in the generated script
                script_text = response.text.strip()
                total_acts = len([line for line in script_text.split('\n') if line.strip().startswith('Cảnh ')])
                total_scenes_used = len([line for line in script_text.split('\n') if line.strip().startswith('Scene ')])
                
                return {
                    "success": True,
                    "final_script": script_text,
                    "total_acts": total_acts,
                    "total_scenes_used": total_scenes_used,
                    "voiceover_tone": voiceover_tone,
                    "script_request": script_request
                }
            else:
                return {"success": False, "error": "No response from Advanced AI"}
                
        except Exception as e:
            logging.error(f"Final script generation error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _format_scenes_for_prompt(self, scenes: List[Dict]) -> str:
        """Format scenes data for prompt"""
        if not scenes:
            return "No scenes available"
        
        formatted_scenes = []
        for scene in scenes:
            scene_id = scene.get('scene_id', 'Unknown')
            timestamp = scene.get('timestamp_str', 'Unknown time')
            source_video = scene.get('source_video', 'Unknown video')
            visual = scene.get('visual_description', 'N/A')
            conversation = scene.get('speaker_transcript', 'No speech')
            speakers = ', '.join(scene.get('unique_speakers', []))
            
            formatted_scene = f"""Scene {scene_id}: ({timestamp}) - From: {source_video}
🎭 Visual: {visual}
🎤 Conversation: {conversation}
👥 Speakers: {speakers}
---"""
            formatted_scenes.append(formatted_scene)
        
        return '\n'.join(formatted_scenes)

    def _call_claude_api(self, prompt: str, max_tokens: int = 4000) -> str:
        """Gọi Advanced AI API thông qua OpenRouter cho voiceover generation"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Script Generator"
        }

        data = {
            "model": self.claude_model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Advanced AI API: {str(e)}"

    def _build_comprehensive_prompt(self, user_request: str, all_scenes: List[Dict]) -> str:
        """Build comprehensive prompt for script generation - UPDATED: Generic for all video types"""
        # Format scenes data
        scenes_text = self._format_scenes_for_prompt(all_scenes)
        
        # UPDATED: Generic prompt without specific video type assumptions
        prompt = f"""
Bạn là chuyên gia viết kịch bản video chuyên nghiệp.

**YÊU CẦU KỊCH BẢN:** {user_request}

**TẤT CẢ SCENES AVAILABLE:**
{scenes_text}

**NHIỆM VỤ:**
1. Phân tích tất cả scenes và chọn những scenes phù hợp nhất
2. Tạo kịch bản có cấu trúc logic theo yêu cầu
3. Sắp xếp scenes theo thứ tự hợp lý
4. Đảm bảo nội dung kịch bản đáp ứng yêu cầu người dùng

**ĐỊNH DẠNG XUẤT RA:**
**Tiêu đề kịch bản phù hợp với yêu cầu**

**Phần 1: [Tên phần phù hợp]**
Scene [ID]: ([timestamp])
🎭 Visual: [mô tả hình ảnh]
🎤 Conversation: [nội dung đối thoại]
👥 Speakers: [người nói]

**Phần 2: [Tên phần tiếp theo]**
Scene [ID]: ([timestamp])
...

**HƯỚNG DẪN:**
1. Chọn 3-15 scenes phù hợp nhất với yêu cầu
2. Sắp xếp theo logic phù hợp với nội dung
3. Tạo cấu trúc rõ ràng với tiêu đề phần hợp lý
4. Giữ nguyên Scene ID và timestamp
5. Đảm bảo kịch bản có tính liên kết và hoàn chỉnh

Tạo kịch bản hoàn chỉnh ngay:
"""
        return prompt