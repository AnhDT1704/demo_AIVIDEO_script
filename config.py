import os

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Video constraints
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MIN_DURATION = 30  # 30 seconds
MAX_DURATION = 7200  # 2 hours
SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'mkv']

# Processing settings
SCENE_THRESHOLD = 70.0  # PySceneDetect threshold
FRAME_SAMPLE_RATE = 1  # 1 frame per second

# Scene optimization settings
SCENE_OPTIMIZATION = True
MIN_SCENE_DURATION = 3.0
MERGE_THRESHOLD_DURATION = 5.0
ENABLE_SCENE_LIMIT = False  # DISABLED - No scene limit
MAX_SCENES_LIMIT = 15  # Kept for backup/future use
ADAPTIVE_THRESHOLD_BOOST = 15.0

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Blaze.vn STT API settings
BLAZE_API_URL = "https://api.blaze.vn/v1/stt/transcribe?enable_segments=true"
BLAZE_API_KEY = os.getenv('BLAZE_API_KEY')

# Google AI Studio settings
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
GEMINI_MODEL = "gemini-2.5-flash"

# Create directories if not exist
for directory in [UPLOAD_DIR, TEMP_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
