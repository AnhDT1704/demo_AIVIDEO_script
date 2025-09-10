import os
import streamlit as st
from scenedetect import ContentDetector, detect
from scenedetect.video_splitter import split_video_ffmpeg
from typing import List, Dict, Any
from config import SCENE_THRESHOLD, TEMP_DIR, ENABLE_SCENE_LIMIT, MAX_SCENES_LIMIT

class SceneDetector:
    """Handle video scene detection using PySceneDetect"""

    def __init__(self, threshold: float = SCENE_THRESHOLD):
        self.threshold = threshold
        self.temp_dir = TEMP_DIR

    def _seconds_to_timecode(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format"""
        from datetime import timedelta
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    def detect_scenes(self, video_path: str, apply_optimization: bool = True) -> List[Dict[str, Any]]:
        """Detect scene changes with optimization to reduce number of scenes"""
        try:
            # Increase threshold to reduce micro cuts
            higher_threshold = min(self.threshold + 15.0, 95.0)  # from 70 to 85
            scene_list = detect(video_path, ContentDetector(threshold=higher_threshold))

            # Convert to custom format with proper timestamp from the start
            scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                duration = end_sec - start_sec
                
                scene_data = {
                    'scene_id': i + 1,
                    'start_time': start_sec,
                    'end_time': end_sec,
                    'duration': duration,
                    'start_frame': start_time.get_frames(),
                    'end_frame': end_time.get_frames(),
                    'timestamp_str': f"{self._seconds_to_timecode(start_sec)} - {self._seconds_to_timecode(end_sec)}"
                }
                scenes.append(scene_data)

            if apply_optimization and scenes:
                original_count = len(scenes)

                scenes = self.filter_short_scenes(scenes, min_duration=3.0)
                after_filter = len(scenes)

                scenes = self.merge_adjacent_short_scenes(scenes, min_duration=5.0)
                after_merge = len(scenes)

                # Apply scene limit only if enabled in config
                if ENABLE_SCENE_LIMIT:
                    scenes = self.limit_max_scenes(scenes, max_scenes=MAX_SCENES_LIMIT)
                    final_count = len(scenes)
                    st.info(f"🎯 Scene optimization: {original_count}→{after_filter}→{after_merge}→{final_count} scenes")
                else:
                    final_count = len(scenes)
                    st.info(f"🎯 Scene optimization: {original_count}→{after_filter}→{final_count} scenes (no limit)")

            return scenes

        except Exception as e:
            st.error(f"Scene detection failed: {str(e)}")
            return []

    def get_scene_stats(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about detected scenes"""
        if not scenes:
            return {}

        durations = [scene['duration'] for scene in scenes]
        return {
            'total_scenes': len(scenes),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_video_duration': sum(durations)
        }

    def filter_short_scenes(self, scenes: List[Dict[str, Any]], min_duration: float = 3.0) -> List[Dict[str, Any]]:
        """Merge short scenes instead of deleting them"""
        if not scenes:
            return scenes
        
        processed_scenes = []
        current_scene = scenes[0].copy()
        
        for next_scene in scenes[1:]:
            if current_scene['duration'] < min_duration:
                # GỘP scene ngắn với scene kế tiếp
                current_scene['end_time'] = next_scene['end_time'] 
                current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
                current_scene['end_frame'] = next_scene['end_frame']
                current_scene['timestamp_str'] = f"{self._seconds_to_timecode(current_scene['start_time'])} - {self._seconds_to_timecode(current_scene['end_time'])}"
                
                st.info(f"🔗 Merged short Scene {current_scene['scene_id']} with Scene {next_scene['scene_id']}")
            else:
                processed_scenes.append(current_scene)
                current_scene = next_scene.copy()
        
        processed_scenes.append(current_scene)
        
        # Cập nhật scene_id
        for i, scene in enumerate(processed_scenes):
            scene['scene_id'] = i + 1
        
        return processed_scenes

    def merge_adjacent_short_scenes(self, scenes: List[Dict[str, Any]], min_duration: float = 5.0) -> List[Dict[str, Any]]:
        """Merge adjacent short scenes WITHOUT creating gaps"""
        if not scenes:
            return scenes

        merged_scenes = []
        current_scene = scenes[0].copy()

        for next_scene in scenes[1:]:
            # Check if we need to merge (short scene OR has gap)
            has_gap = next_scene['start_time'] > current_scene['end_time']
            current_too_short = current_scene['duration'] < min_duration
            
            if current_too_short or has_gap:
                # CRITICAL: Merge without gaps - connect scenes properly
                current_scene['end_time'] = max(current_scene['end_time'], next_scene['end_time'])
                current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
                current_scene['end_frame'] = max(current_scene.get('end_frame', 0), next_scene.get('end_frame', 0))
                # Update timestamp with proper format
                current_scene['timestamp_str'] = f"{self._seconds_to_timecode(current_scene['start_time'])} - {self._seconds_to_timecode(current_scene['end_time'])}"
            else:
                merged_scenes.append(current_scene)
                current_scene = next_scene.copy()

        merged_scenes.append(current_scene)

        # Update scene_id after merging
        for i, scene in enumerate(merged_scenes):
            scene['scene_id'] = i + 1

        return merged_scenes

    def limit_max_scenes(self, scenes: List[Dict[str, Any]], max_scenes: int = 15) -> List[Dict[str, Any]]:
        """Limit max scenes by selecting the most important ones and update timestamp format"""
        if len(scenes) <= max_scenes:
            return scenes

        sorted_scenes = sorted(scenes, key=lambda x: x['duration'], reverse=True)
        important_scenes = sorted_scenes[:max_scenes]
        final_scenes = sorted(important_scenes, key=lambda x: x['start_time'])

        for i, scene in enumerate(final_scenes):
            scene['scene_id'] = i + 1
            # Ensure proper timestamp format
            scene['timestamp_str'] = f"{self._seconds_to_timecode(scene['start_time'])} - {self._seconds_to_timecode(scene['end_time'])}"

        return final_scenes

    def export_scenes_to_files(self, video_path: str, scenes: List[Dict[str, Any]], output_dir: str = None) -> bool:
        """Export each scene as separate video file"""
        try:
            if output_dir is None:
                output_dir = os.path.join(self.temp_dir, "scenes")
            os.makedirs(output_dir, exist_ok=True)

            scene_times = [(scene['start_time'], scene['end_time']) for scene in scenes]

            split_video_ffmpeg(
                video_path,
                scene_times,
                output_file_template=os.path.join(output_dir, 'scene_$SCENE_NUMBER.mp4'),
                video_name='scenes'
            )

            return True
        except Exception as e:
            st.error(f"Scene export failed: {str(e)}")
            return False
