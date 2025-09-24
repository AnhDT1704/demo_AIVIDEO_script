# scene_detector_enhanced.py - Enhanced with Dynamic Threshold Support
import os
import streamlit as st
from scenedetect import ContentDetector, detect
from scenedetect.video_splitter import split_video_ffmpeg
from typing import List, Dict, Any
from config import SCENE_THRESHOLD, TEMP_DIR, ENABLE_SCENE_LIMIT, MAX_SCENES_LIMIT

class SceneDetector:
    """Handle video scene detection using PySceneDetect with configurable threshold"""

    def __init__(self, threshold: float = None):
        # Use provided threshold or fall back to config default
        self.threshold = threshold if threshold is not None else SCENE_THRESHOLD
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

    def detect_scenes(self, video_path: str, apply_optimization: bool = True, custom_threshold: float = None) -> List[Dict[str, Any]]:
        """Detect scene changes with configurable threshold and optimization"""
        try:
            # Use custom threshold if provided, otherwise use instance threshold
            working_threshold = custom_threshold if custom_threshold is not None else self.threshold
            
            pass  # Hidden scene detector message
            
            # Use the working threshold directly without adding boost
            scene_list = detect(video_path, ContentDetector(threshold=working_threshold))

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
                
                # Get dynamic values from config if available
                try:
                    from config import MIN_SCENE_DURATION, MERGE_THRESHOLD_DURATION
                    min_scene_dur = MIN_SCENE_DURATION
                    merge_thresh_dur = MERGE_THRESHOLD_DURATION
                except ImportError:
                    min_scene_dur = 3.0
                    merge_thresh_dur = 5.0
                
                scenes = self.filter_short_scenes(scenes, min_duration=min_scene_dur)
                after_filter = len(scenes)
                scenes = self.merge_adjacent_short_scenes(scenes, min_duration=merge_thresh_dur)
                after_merge = len(scenes)

                # Apply scene limit only if enabled in config
                if ENABLE_SCENE_LIMIT:
                    scenes = self.limit_max_scenes(scenes, max_scenes=MAX_SCENES_LIMIT)
                    final_count = len(scenes)
                    pass  # Hidden scene detector message
                else:
                    final_count = len(scenes)
                    pass  # Hidden scene detector message

            # Provide threshold feedback
            self._provide_threshold_feedback(scenes, working_threshold)
            
            return scenes

        except Exception as e:
            st.error(f"Scene detection failed: {str(e)}")
            return []

    def _provide_threshold_feedback(self, scenes: List[Dict[str, Any]], threshold: float):
        """Provide feedback on threshold effectiveness"""
        scene_count = len(scenes)
        
        if scene_count == 0:
            # st.error(f"âŒ No scenes detected with threshold {threshold}")
            pass  # Hidden scene detector message
        elif scene_count == 1:
            pass  # Hidden scene detector message
            pass  # Hidden scene detector message
        elif scene_count < 3:
            pass  # Hidden scene detector message
            pass  # Hidden scene detector message
        elif scene_count > 50:
            pass  # Hidden scene detector message
            pass  # Hidden scene detector message
        elif scene_count > 30:
            pass  # Hidden scene detector message
        else:
            pass  # Hidden scene detector message

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
                # Merge short scene with next scene
                current_scene['end_time'] = next_scene['end_time']
                current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
                current_scene['end_frame'] = next_scene['end_frame']
                current_scene['timestamp_str'] = f"{self._seconds_to_timecode(current_scene['start_time'])} - {self._seconds_to_timecode(current_scene['end_time'])}"
                pass  # Hidden scene detector message
            else:
                processed_scenes.append(current_scene)
                current_scene = next_scene.copy()

        processed_scenes.append(current_scene)

        # Update scene_id
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
                # Merge without gaps - connect scenes properly
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

    def get_threshold_recommendations(self, video_duration: float, video_type: str = "unknown") -> Dict[str, float]:
        """Get threshold recommendations based on video characteristics"""
        recommendations = {
            "very_sensitive": 10.0,
            "sensitive": 15.0,
            "balanced": 27.0,
            "conservative": 35.0,
            "very_conservative": 50.0
        }
        
        # Adjust based on video duration
        if video_duration < 120:  # < 2 minutes
            # Short videos, be more sensitive
            for key in recommendations:
                recommendations[key] = max(5.0, recommendations[key] - 5)
        elif video_duration > 3600:  # > 1 hour
            # Long videos, be more conservative
            for key in recommendations:
                recommendations[key] = min(95.0, recommendations[key] + 10)
        
        # Adjust based on video type
        if video_type in ["talk", "interview", "lecture"]:
            # Talking head videos, more conservative
            for key in recommendations:
                recommendations[key] = min(95.0, recommendations[key] + 5)
        elif video_type in ["action", "music", "sports"]:
            # Fast-cut videos, more sensitive
            for key in recommendations:
                recommendations[key] = max(5.0, recommendations[key] - 5)
        
        return recommendations