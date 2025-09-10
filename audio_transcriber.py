import os
import requests
import streamlit as st
import ffmpeg
from pydub import AudioSegment
from typing import Dict, List, Any, Optional
from config import BLAZE_API_URL, BLAZE_API_KEY, TEMP_DIR

class AudioTranscriber:
    """Handle audio transcription using Blaze.vn STT API"""
    
    def __init__(self):
        self.api_url = BLAZE_API_URL
        self.api_key = BLAZE_API_KEY
        self.temp_dir = TEMP_DIR
    
    def extract_audio(self, video_path: str) -> str:
        """Extract complete audio from video file"""
        try:
            audio_filename = f"audio_{os.path.basename(video_path).split('.')[0]}.wav"
            audio_path = os.path.join(self.temp_dir, audio_filename)
            
            stream = ffmpeg.input(video_path)
            audio = stream.audio
            out = ffmpeg.output(
                audio,
                audio_path,
                acodec='pcm_s16le',
                ac=1,
                ar='16000'
            )
            ffmpeg.run(out, overwrite_output=True, quiet=True)
            return audio_path
        except Exception as e:
            st.error(f"Audio extraction failed: {str(e)}")
            return None
    
    def extract_scene_audio_segments(self, audio_path: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract separate audio file for each scene"""
        try:
            # Create scenes audio directory
            scenes_audio_dir = os.path.join(self.temp_dir, "scenes_audio")
            os.makedirs(scenes_audio_dir, exist_ok=True)
            
            # Load complete audio
            audio = AudioSegment.from_file(audio_path)
            scene_audio_files = []
            
            progress_bar = st.progress(0)
            
            for i, scene in enumerate(scenes):
                # Extract audio segment for this scene
                start_ms = int(scene['start_time'] * 1000)
                end_ms = int(scene['end_time'] * 1000)
                
                # Extract exact duration
                scene_audio = audio[start_ms:end_ms]
                
                # Skip very short scenes (< 1 second)
                if scene_audio.duration_seconds < 1.0:
                    scene_audio_files.append({
                        'scene_id': scene['scene_id'],
                        'audio_path': None,
                        'duration': scene_audio.duration_seconds,
                        'skip_reason': 'Too short'
                    })
                    continue
                
                # Export scene audio
                scene_audio_path = os.path.join(scenes_audio_dir, f"scene_{scene['scene_id']:03d}.wav")
                scene_audio.export(scene_audio_path, format='wav')
                
                scene_audio_files.append({
                    'scene_id': scene['scene_id'],
                    'audio_path': scene_audio_path,
                    'duration': scene_audio.duration_seconds,
                    'skip_reason': None
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(scenes))
            
            return scene_audio_files
        except Exception as e:
            st.error(f"Scene audio extraction failed: {str(e)}")
            return []
    
    def transcribe_scene_audio(self, scene_audio_path: str, scene_id: int) -> Dict[str, Any]:
        """Transcribe single scene audio using Blaze.vn STT API"""
        try:
            if not os.path.exists(scene_audio_path):
                raise FileNotFoundError(f"Audio file not found: {scene_audio_path}")
                
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }
            
            with open(scene_audio_path, 'rb') as audio_file:
                files = {
                    'audio_file': (
                        os.path.basename(scene_audio_path),
                        audio_file,
                        'audio/wav'
                    )
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    timeout=120  # 2 minutes per scene
                )
            
            if response.status_code == 200:
                result = response.json()
                parsed = self._parse_blaze_response(result, scene_id)
                return parsed
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return {
                    'scene_id': scene_id,
                    'text': '',
                    'language': 'vi',
                    'words': [],
                    'error': f"API Error: {response.status_code}"
                }
        except Exception as e:
            return {
                'scene_id': scene_id,
                'text': '',
                'language': 'vi', 
                'words': [],
                'error': str(e)
            }
    
    def transcribe_scenes_batch(self, scenes: List[Dict[str, Any]], scene_audio_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transcribe all scenes with their respective audio segments"""
        scenes_with_text = []
        progress_bar = st.progress(0)
        
        for i, (scene, audio_info) in enumerate(zip(scenes, scene_audio_files)):
            # Skip scenes without audio or too short
            if audio_info['audio_path'] is None:
                scene_with_text = scene.copy()
                scene_with_text.update({
                    'transcript': '',
                    'word_count': 0,
                    'words': [],
                    'has_speech': False,
                    'skip_reason': audio_info['skip_reason']
                })
                scenes_with_text.append(scene_with_text)
                continue
            
            # Transcribe this scene's audio
            transcription = self.transcribe_scene_audio(audio_info['audio_path'], scene['scene_id'])
            
            # Combine scene data with transcription
            scene_with_text = scene.copy()
            scene_with_text.update({
                'transcript': transcription['text'],
                'word_count': len(transcription['text'].split()) if transcription['text'] else 0,
                'words': transcription.get('words', []),
                'has_speech': bool(transcription['text'].strip()),
                'language': transcription.get('language', 'vi'),
                'transcription_error': transcription.get('error', None)
            })
            
            scenes_with_text.append(scene_with_text)
            
            # Update progress
            progress_bar.progress((i + 1) / len(scene_audio_files))
            
            # Small delay to avoid API rate limiting
            import time
            time.sleep(0.5)
        
        return scenes_with_text
    
    def _parse_blaze_response(self, blaze_result: Dict, scene_id: int) -> Dict[str, Any]:
        """Parse Blaze API response for single scene"""
        try:
            segments = blaze_result.get('segments', [])
            # Combine text from all segments
            full_text = ' '.join(segment.get('text', '') for segment in segments)
            language = 'vi'  # Blaze API always returns Vietnamese
            
            # Extract words with relative timestamps (within scene)
            words = []
            for segment in segments:
                if 'words' in segment:
                    for word_data in segment['words']:
                        words.append({
                            'word': word_data.get('word', ''),
                            'start': word_data.get('start', 0),
                            'end': word_data.get('end', 0),
                            'confidence': word_data.get('confidence', 1.0)
                        })
            
            return {
                'scene_id': scene_id,
                'text': full_text,
                'language': language,
                'words': words
            }
        except Exception as e:
            return {
                'scene_id': scene_id,
                'text': '',
                'language': 'vi',
                'words': [],
                'error': f"Parse error: {str(e)}"
            }
    
    def get_transcription_stats(self, scenes_with_text: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate transcription statistics"""
        total_words = sum(scene['word_count'] for scene in scenes_with_text)
        scenes_with_speech = sum(1 for scene in scenes_with_text if scene['has_speech'])
        
        return {
            'total_words': total_words,
            'scenes_with_speech': scenes_with_speech,
            'scenes_without_speech': len(scenes_with_text) - scenes_with_speech,
            'speech_coverage': scenes_with_speech / len(scenes_with_text) * 100 if scenes_with_text else 0
        }
    
    def cleanup_scene_audio_files(self, scene_audio_files: List[Dict[str, Any]]):
        """Clean up all scene audio files"""
        for audio_info in scene_audio_files:
            if audio_info['audio_path'] and os.path.exists(audio_info['audio_path']):
                try:
                    os.remove(audio_info['audio_path'])
                except Exception as e:
                    st.warning(f"Failed to cleanup scene audio {audio_info['audio_path']}: {str(e)}")
        
        # Remove scenes audio directory
        scenes_audio_dir = os.path.join(self.temp_dir, "scenes_audio")
        if os.path.exists(scenes_audio_dir):
            try:
                os.rmdir(scenes_audio_dir)
            except:
                pass
    
    def cleanup_audio_file(self, audio_path: str):
        """Clean up extracted audio file"""
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                st.warning(f"Failed to cleanup audio file: {str(e)}")
