from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

# Import existing modules
from video_ingestion import VideoProcessor
from scene_detector import SceneDetector
from audio_transcriber import AudioTranscriber
from gemini_frame_analyzer import GeminiFrameAnalyzer
from config import SUPPORTED_FORMATS, MAX_FILE_SIZE, TEMP_DIR

app = FastAPI(
    title="Hybrid Video Scene Extraction API",
    description="API for video scene detection, transcription and analysis",
    version="1.0.0"
)

# In-memory storage for job status
job_status: Dict[str, Dict] = {}

class VideoAnalysisPipeline:
    """Complete video analysis pipeline for API use"""
    
    def __init__(self):
        self.processor = VideoProcessor()
        self.scene_detector = SceneDetector()
        self.transcriber = AudioTranscriber()
        self.frame_analyzer = GeminiFrameAnalyzer()

    async def run_complete_analysis(self, video_id: str, video_path: str) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        try:
            # Update status: Starting
            job_status[video_id]['status'] = 'processing'
            job_status[video_id]['stage'] = 'video_validation'
            
            # Step 1: Video Validation
            validation_result = self.processor.validate_video_by_path(video_path)
            if not validation_result['valid']:
                raise Exception(f"Video validation failed: {validation_result.get('error', 'Unknown error')}")
            
            # Step 2: Scene Detection
            job_status[video_id]['stage'] = 'scene_detection'
            scenes = self.scene_detector.detect_scenes(video_path)
            
            # Step 3: Audio Transcription
            job_status[video_id]['stage'] = 'audio_transcription'
            if validation_result['has_audio']:
                audio_path = self.transcriber.extract_audio(video_path)
                scene_audio_files = self.transcriber.extract_scene_audio_segments(audio_path, scenes)
                scenes_with_text = self.transcriber.transcribe_scenes_batch(scenes, scene_audio_files)
                # Cleanup audio files
                self.transcriber.cleanup_scene_audio_files(scene_audio_files)
                self.transcriber.cleanup_audio_file(audio_path)
            else:
                scenes_with_text = scenes
            
            # Step 4: Visual Analysis
            job_status[video_id]['stage'] = 'visual_analysis'
            scenes_with_frames = self.frame_analyzer.extract_frames_from_scenes(video_path, scenes_with_text)
            enriched_scenes = self.frame_analyzer.analyze_scenes_batch(scenes_with_frames)
            self.frame_analyzer.cleanup_frame_files(scenes_with_frames)
            
            # Prepare final results
            scene_stats = self.scene_detector.get_scene_stats(enriched_scenes)
            transcription_stats = self.transcriber.get_transcription_stats(enriched_scenes)
            visual_stats = self.frame_analyzer.get_visual_analysis_stats(enriched_scenes)
            
            final_results = {
                'video_metadata': {
                    'duration': validation_result['duration'],
                    'resolution': validation_result['resolution'],
                    'has_audio': validation_result['has_audio']
                },
                'scene_stats': scene_stats,
                'transcription_stats': transcription_stats,
                'visual_stats': visual_stats,
                'scenes': enriched_scenes
            }
            
            # Update status: Complete
            job_status[video_id]['status'] = 'completed'
            job_status[video_id]['results'] = final_results
            job_status[video_id]['completed_at'] = datetime.now().isoformat()
            
            # Cleanup temp video file
            self.processor.cleanup_temp_file(video_path)
            
            return final_results
            
        except Exception as e:
            job_status[video_id]['status'] = 'failed'
            job_status[video_id]['error'] = str(e)
            job_status[video_id]['failed_at'] = datetime.now().isoformat()
            raise

# Initialize pipeline
pipeline = VideoAnalysisPipeline()

@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and validate video file"""
    
    # Validate file extension
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {file_ext}")
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {len(file_content)} bytes")
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save file to temp directory
    temp_filename = f"video_{video_id}.{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_content)
    
    # Initialize job status
    job_status[video_id] = {
        'video_id': video_id,
        'filename': file.filename,
        'status': 'uploaded',
        'stage': 'ready',
        'uploaded_at': datetime.now().isoformat(),
        'temp_path': temp_path
    }
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "size": len(file_content),
        "status": "uploaded",
        "message": "Video uploaded successfully. Use /start-analysis to begin processing."
    }

@app.post("/start-analysis/{video_id}")
async def start_analysis(video_id: str, background_tasks: BackgroundTasks):
    """Start video analysis pipeline"""
    
    if video_id not in job_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if job_status[video_id]['status'] != 'uploaded':
        raise HTTPException(status_code=400, detail=f"Video is in {job_status[video_id]['status']} state")
    
    # Start analysis in background
    temp_path = job_status[video_id]['temp_path']
    background_tasks.add_task(pipeline.run_complete_analysis, video_id, temp_path)
    
    job_status[video_id]['status'] = 'queued'
    job_status[video_id]['started_at'] = datetime.now().isoformat()
    
    return {
        "video_id": video_id,
        "status": "queued",
        "message": "Analysis started. Check /analysis-status for progress."
    }

@app.get("/analysis-status/{video_id}")
async def get_analysis_status(video_id: str):
    """Get analysis status and progress"""
    
    if video_id not in job_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = job_status[video_id].copy()
    # Remove temp_path and large results from status response
    status.pop('temp_path', None)
    status.pop('results', None)
    
    return status

@app.get("/results/{video_id}")
async def get_results(video_id: str):
    """Get complete analysis results"""
    
    if video_id not in job_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    job = job_status[video_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Analysis not complete. Status: {job['status']}")
    
    return {
        "video_id": video_id,
        "status": job['status'],
        "completed_at": job['completed_at'],
        "results": job['results']
    }

@app.delete("/cleanup/{video_id}")
async def cleanup_video(video_id: str):
    """Clean up video and results"""
    
    if video_id not in job_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Remove from memory
    del job_status[video_id]
    
    return {"video_id": video_id, "message": "Video cleaned up successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # ✅ SỬA ĐỔI: Render cần PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)
