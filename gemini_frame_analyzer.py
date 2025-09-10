import os
import cv2
import json
import numpy as np
import streamlit as st
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from PIL import Image
import tempfile
import ffmpeg
from config import GOOGLE_AI_API_KEY, GEMINI_MODEL, TEMP_DIR

class GeminiFrameAnalyzer:
    """Handle video analysis using Google Gemini 2.5 Flash API"""
    
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=GOOGLE_AI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.temp_dir = TEMP_DIR

    def extract_scene_clips(self, video_path: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract complete video clip for each scene - WITH DEBUG"""
        try:
            scenes_with_clips = []
            
            # DEBUG: Print initial info
            st.write(f"🔍 **DEBUG**: Starting clip extraction from {len(scenes)} scenes")
            st.write(f"🔍 **DEBUG**: Video path: {video_path}")
            
            st.info(f"🎬 Extracting video clips from {len(scenes)} scenes...")
            progress_bar = st.progress(0)
            
            for i, scene in enumerate(scenes):
                start_time = scene['start_time']
                end_time = scene['end_time']
                duration = end_time - start_time
                
                # DEBUG: Print scene info
                st.write(f"🔍 **DEBUG**: Scene {scene['scene_id']} - Duration: {duration:.1f}s ({start_time:.1f}s - {end_time:.1f}s)")
                
                # Create clip filename
                clip_filename = f"scene_{scene['scene_id']:03d}_clip.mp4"
                clip_path = os.path.join(self.temp_dir, clip_filename)
                
                # Clean up existing clip
                if os.path.exists(clip_path):
                    try:
                        os.remove(clip_path)
                    except:
                        pass
                
                # Extract scene clip using FFmpeg
                success = False
                error_msg = None
                
                try:
                    (
                        ffmpeg
                        .input(video_path, ss=start_time, t=duration)
                        .output(clip_path, 
                               vcodec='libx264', 
                               acodec='aac',
                               preset='medium',
                               crf=28)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True, quiet=True)
                    )
                    
                    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                        file_size = os.path.getsize(clip_path) / (1024*1024)
                        st.write(f"✅ **DEBUG**: Clip extracted - Scene {scene['scene_id']}, Size: {file_size:.1f}MB")
                        success = True
                    else:
                        error_msg = "Clip file not created or empty"
                        st.write(f"❌ **DEBUG**: Clip extraction failed - Scene {scene['scene_id']}: {error_msg}")
                        
                except Exception as e:
                    error_msg = f"FFmpeg extraction failed: {str(e)}"
                    st.write(f"❌ **DEBUG**: FFmpeg error - Scene {scene['scene_id']}: {error_msg}")
                
                # Create scene data
                scene_with_clip = scene.copy()
                if success:
                    scene_with_clip.update({
                        'clip_path': clip_path,
                        'clip_duration': duration,
                        'extraction_method': 'video_clip'
                    })
                else:
                    scene_with_clip.update({
                        'clip_path': None,
                        'clip_duration': duration,
                        'extraction_error': error_msg or 'Unknown error'
                    })
                
                scenes_with_clips.append(scene_with_clip)
                progress_bar.progress((i + 1) / len(scenes))
            
            # DEBUG: Final extraction stats
            successful_clips = len([s for s in scenes_with_clips if s.get('clip_path')])
            failed_clips = len(scenes_with_clips) - successful_clips
            st.write(f"🔍 **DEBUG**: Extraction complete - Success: {successful_clips}, Failed: {failed_clips}")
            
            return scenes_with_clips
            
        except Exception as e:
            st.write(f"❌ **DEBUG**: Fatal error in clip extraction: {str(e)}")
            st.error(f"Scene clip extraction failed: {str(e)}")
            return []

    def analyze_scene_clip_with_gemini(self, clip_path: str, scene_id: int) -> Dict[str, Any]:
        """Analyze complete scene clip using Gemini 2.5 Flash - WITH FILE STATE POLLING"""
        try:
            if not os.path.exists(clip_path):
                return {
                    'scene_id': scene_id,
                    'visual_description': '',
                    'error': 'Clip file not found'
                }
            
            # Check file size (limit to prevent API issues)
            file_size = os.path.getsize(clip_path) / (1024 * 1024)  # MB
            if file_size > 200:  # 200MB limit
                return {
                    'scene_id': scene_id,
                    'visual_description': 'File too large for analysis',
                    'error': f'File size {file_size:.1f}MB exceeds 200MB limit'
                }
            
            # Upload video clip to Gemini
            video_file = genai.upload_file(path=clip_path)
            
            # CRITICAL FIX: Wait for file to become ACTIVE
            import time
            max_wait_time = 60  # 60 seconds timeout
            wait_start = time.time()
            
            while video_file.state.name == "PROCESSING":
                if time.time() - wait_start > max_wait_time:
                    return {
                        'scene_id': scene_id,
                        'visual_description': '',
                        'error': f'File processing timeout after {max_wait_time}s'
                    }
                
                st.write(f"🔄 **DEBUG**: Waiting for Scene {scene_id} file to become ACTIVE... ({video_file.state.name})")
                time.sleep(3)  # Wait 3 seconds
                video_file = genai.get_file(video_file.name)  # Refresh file state
            
            if video_file.state.name == "FAILED":
                return {
                    'scene_id': scene_id,
                    'visual_description': '',
                    'error': 'File processing failed on Gemini servers'
                }
            
            # File is now ACTIVE, proceed with analysis
            st.write(f"✅ **DEBUG**: Scene {scene_id} file is ACTIVE, proceeding with analysis")
            
            # Simple prompt for natural description
            prompt = """
            Mô tả đoạn video này một cách ngắn gọn và súc tích.

            Cung cấp một đoạn văn liền mạch (3-4 câu) nêu:
            - Các đối tượng chính và bối cảnh
            - Hành động nổi bật diễn ra
            - Tông màu và cảm giác tổng thể

            Tránh mô tả quá chi tiết. Tránh sử dụng bullet points hay danh sách.
            """
            
            # Send request to Gemini
            response = self.model.generate_content([video_file, prompt])
            
            # Get description - SIMPLIFIED, no parsing needed
            description = response.text.strip()
            
            result = {
                'scene_id': scene_id,
                'visual_description': description,
                'scene_type': 'video_analyzed'
            }
            
            # Cleanup uploaded file
            try:
                genai.delete_file(video_file.name)
            except:
                pass
            
            return result
            
        except Exception as e:
            return {
                'scene_id': scene_id,
                'visual_description': '',
                'error': str(e)
            }

    def analyze_scenes_batch(self, scenes_with_clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze all scene clips with Gemini API - WITH DEBUG"""
        enriched_scenes = []
        
        # DEBUG: Print initial info
        st.write(f"🔍 **DEBUG**: Starting batch analysis on {len(scenes_with_clips)} scenes")
        
        valid_scenes = [s for s in scenes_with_clips if s.get('clip_path') and os.path.exists(s['clip_path'])]
        
        # DEBUG: Print valid clips count
        st.write(f"🔍 **DEBUG**: {len(valid_scenes)} scenes have valid clips")
        st.write(f"🔍 **DEBUG**: {len(scenes_with_clips) - len(valid_scenes)} scenes have missing/invalid clips")
        
        st.info(f"🧠 Analyzing {len(valid_scenes)} video clips with Gemini 2.5 Flash...")
        progress_bar = st.progress(0)
        
        for i, scene in enumerate(scenes_with_clips):
            # DEBUG: Print each scene processing
            st.write(f"🔍 **DEBUG**: Processing Scene {scene['scene_id']} ({i+1}/{len(scenes_with_clips)})")
            
            if scene.get('clip_path') and os.path.exists(scene['clip_path']):
                # DEBUG: Print clip info
                file_size = os.path.getsize(scene['clip_path']) / (1024*1024)
                st.write(f"🔍 **DEBUG**: Clip exists, size: {file_size:.1f}MB, path: {scene['clip_path']}")
                
                try:
                    # Analyze clip with Gemini
                    visual_analysis = self.analyze_scene_clip_with_gemini(scene['clip_path'], scene['scene_id'])
                    
                    # DEBUG: Print analysis result
                    if visual_analysis.get('error'):
                        st.write(f"❌ **DEBUG**: Analysis failed for Scene {scene['scene_id']}: {visual_analysis['error']}")
                    else:
                        st.write(f"✅ **DEBUG**: Analysis success for Scene {scene['scene_id']}")
                    
                    # SIMPLIFIED: Only add visual description
                    enriched_scene = scene.copy()
                    enriched_scene.update({
                        'visual_description': visual_analysis.get('visual_description', ''),
                        'scene_type': visual_analysis.get('scene_type', 'video_analyzed'),
                        'visual_analysis_error': visual_analysis.get('error', None)
                    })
                    
                except Exception as e:
                    st.write(f"❌ **DEBUG**: Exception during Scene {scene['scene_id']} analysis: {str(e)}")
                    enriched_scene = scene.copy()
                    enriched_scene.update({
                        'visual_description': '',
                        'scene_type': 'unknown',
                        'visual_analysis_error': f'Exception: {str(e)}'
                    })
            else:
                # DEBUG: Print missing clip info
                st.write(f"❌ **DEBUG**: Scene {scene['scene_id']} has no valid clip")
                if scene.get('clip_path'):
                    st.write(f"🔍 **DEBUG**: Clip path exists but file missing: {scene['clip_path']}")
                else:
                    st.write(f"🔍 **DEBUG**: No clip path for Scene {scene['scene_id']}")
                
                # No clip available
                enriched_scene = scene.copy()
                enriched_scene.update({
                    'visual_description': '',
                    'scene_type': 'unknown',
                    'visual_analysis_error': scene.get('extraction_error', 'No clip extracted')
                })
            
            enriched_scenes.append(enriched_scene)
            progress_bar.progress((i + 1) / len(scenes_with_clips))
            
            # Rate limiting
            import time
            time.sleep(2)
        
        # DEBUG: Print final stats
        successful_analyses = len([s for s in enriched_scenes if not s.get('visual_analysis_error')])
        failed_analyses = len(enriched_scenes) - successful_analyses
        st.write(f"🔍 **DEBUG**: Analysis complete - Success: {successful_analyses}, Failed: {failed_analyses}")
        
        return enriched_scenes

    def get_visual_analysis_stats(self, enriched_scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic statistics - SIMPLIFIED"""
        analyzed_scenes = len([s for s in enriched_scenes if not s.get('visual_analysis_error')])
        
        return {
            'analyzed_scenes': analyzed_scenes,
            'failed_scenes': len(enriched_scenes) - analyzed_scenes,
            'scene_types': {'video_analyzed': analyzed_scenes},  # Simplified
            'scene_moods': {'analyzed': analyzed_scenes},  # Simplified
            'avg_objects_per_scene': 0,  # Removed
            'avg_actions_per_scene': 0   # Removed
        }

    def cleanup_clip_files(self, scenes_with_clips: List[Dict[str, Any]]):
        """Clean up extracted video clip files"""
        for scene in scenes_with_clips:
            clip_path = scene.get('clip_path')
            if clip_path and os.path.exists(clip_path):
                try:
                    os.remove(clip_path)
                except Exception as e:
                    st.warning(f"Failed to cleanup clip {clip_path}: {str(e)}")

    # Backward compatibility
    def extract_frames_from_scenes(self, video_path: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Backward compatibility - now extracts clips instead of frames"""
        return self.extract_scene_clips(video_path, scenes)
    
    def cleanup_frame_files(self, scenes_with_frames: List[Dict[str, Any]]):
        """Backward compatibility - now cleans clips instead of frames"""
        return self.cleanup_clip_files(scenes_with_frames)
