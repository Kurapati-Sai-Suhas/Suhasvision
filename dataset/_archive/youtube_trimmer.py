import os
import time
import json
import yt_dlp
from moviepy import VideoFileClip
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from google import genai

# ==========================================
# 1. CLOUD & AI CONFIGURATION
# ==========================================
load_dotenv(override=True)

AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "raw-videos")

if not AZURE_CONNECTION_STRING:
    raise ValueError("🚨 Missing Azure Connection String in .env file!")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ==========================================
# 2. YOUR CONFIGURATION (THE PRE-TRIM)
# ==========================================
YOUTUBE_URL = "https://youtu.be/dHsUM0mYXys?si=V8aoZY64_7Gi6AcV"
BATSMAN_NAME = "youtube_dataset_1"
ANGLE = "frontview"

# ✂️ Define the exact window to feed the AI (in seconds)
# Example: If he bats from 2:00 to 3:00 in the video, use 120 and 180.
MACRO_START_SECOND = 1
MACRO_END_SECOND = 39 

# ==========================================
# 3. AUTOMATION LOGIC
# ==========================================
RAW_VIDEOS_DIR = "raw_videos"
FULL_YT_VIDEO = "temp_full_youtube.mp4"
AI_CHUNK_VIDEO = "temp_ai_chunk.mp4"

def get_ai_timestamps(video_path, retries=3):
    """Asks Gemini to watch the PRE-TRIMMED chunk and find the exact shots."""
    global client
    print(f"\n🧠 [AI] Uploading the {MACRO_END_SECOND - MACRO_START_SECOND}-second chunk to Gemini...")
    
    for attempt in range(retries):
        try:
            video_file = client.files.upload(file=video_path)
            
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)
                
            prompt = """
            You are an expert cricket analyst. Watch this video chunk.
            Identify EVERY SINGLE batting shot played.
            For each shot, find the exact start time (when the batsman begins their setup/trigger movement) and end time (when the follow-through finishes).
            Return ONLY a strictly formatted JSON array of objects.
            Example format:
            [
                {"start_time": 1.2, "end_time": 4.5},
                {"start_time": 15.0, "end_time": 18.2}
            ]
            """
            
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[video_file, prompt]
            )
            
            client.files.delete(name=video_file.name) 
            
            text = response.text.strip().replace('```json', '').replace('```', '')
            timestamps_list = json.loads(text)
            
            print(f"  [🎯] AI found {len(timestamps_list)} perfect shots in this chunk!")
            return timestamps_list
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "400" in error_msg:
                print(f"  ⏳ AI Token Limit Hit! Cooling down for 60 seconds... (Attempt {attempt + 1} of {retries})")
                time.sleep(60)
                load_dotenv(override=True)
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            else:
                print(f"  ⚠ AI Failed to detect shots: {e}")
                return []
    return []

def upload_to_azure_and_delete(local_filepath, file_name):
    print(f"   ☁️ Uploading {file_name} to Azure Blob Storage...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME).get_blob_client(file_name)
        
        with open(local_filepath, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        os.remove(local_filepath)
        print(f"   🗑️ Upload complete! Deleted local clip.")
    except Exception as e:
        print(f"   ❌ Azure Upload Failed! Error: {e}")

def process_pipeline():
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    
    # 🧹 Clean up any old broken files
    for f in [FULL_YT_VIDEO, AI_CHUNK_VIDEO]:
        if os.path.exists(f): os.remove(f)

    # 📥 STEP 1: Download full YouTube video
    print(f"\n📥 Downloading full video from YouTube...")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': FULL_YT_VIDEO, 'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([YOUTUBE_URL])
    print("✅ Download complete!")

    # ✂️ STEP 2: Pre-trim the master video into a small chunk for the AI
    print(f"\n✂️ Slicing video from {MACRO_START_SECOND}s to {MACRO_END_SECOND}s to save AI tokens...")
    full_video = VideoFileClip(FULL_YT_VIDEO)
    
    # Safety check in case the user types a time past the end of the video
    actual_end = min(MACRO_END_SECOND, full_video.duration)
    
    ai_chunk = full_video.subclipped(MACRO_START_SECOND, actual_end)
    ai_chunk.write_videofile(AI_CHUNK_VIDEO, codec="libx264", audio=False, logger=None)
    full_video.close()
    
    # Delete the massive full video to save hard drive space immediately
    os.remove(FULL_YT_VIDEO)

    # 🧠 STEP 3: Send the small chunk to Gemini
    ai_shots = get_ai_timestamps(AI_CHUNK_VIDEO)
    
    if not ai_shots:
        print("🚨 No shots detected or AI failed. Exiting.")
        if os.path.exists(AI_CHUNK_VIDEO): os.remove(AI_CHUNK_VIDEO)
        return

    # 🔪 STEP 4: Cut the individual shots out of the AI chunk
    print(f"\n🎥 Generating final raw clips...")
    chunk_video = VideoFileClip(AI_CHUNK_VIDEO)
    
    for index, shot in enumerate(ai_shots):
        start_time = float(shot['start_time'])
        end_time = float(shot['end_time'])
        
        if end_time > chunk_video.duration:
            continue 

        output_name = f"{BATSMAN_NAME}_{ANGLE}_{index+1:02d}.mp4"
        output_path = os.path.join(RAW_VIDEOS_DIR, output_name)
        
        print(f"\n-> Isolating Shot {index+1}: {output_name} (from {start_time}s to {end_time}s)...")
        final_clip = chunk_video.subclipped(start_time, end_time)
        final_clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        
        # ☁️ STEP 5: Beam to Azure
        upload_to_azure_and_delete(output_path, output_name)
        
    chunk_video.close()
    
    # Clean up the master chunk
    if os.path.exists(AI_CHUNK_VIDEO):
        os.remove(AI_CHUNK_VIDEO)
        
    print(f"\n🎉 Success! All shots isolated perfectly and beamed to Azure!")

if __name__ == "__main__":
    process_pipeline()