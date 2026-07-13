import os
import time
import json
import csv
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

# Define global client, will be reset on rate limit
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

RAW_VIDEOS_DIR = "raw_videos"
FULL_YT_VIDEO = "temp_full_youtube.mp4"
AI_CHUNK_VIDEO = "temp_ai_chunk.mp4"

def get_ai_timestamps(video_path, macro_length, retries=3):
    """Asks Gemini to watch the PRE-TRIMMED chunk and find the exact shots."""
    global client
    print(f"\n🧠 [AI] Uploading the {macro_length}-second chunk to Gemini...")
    
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

def process_single_video(url, batsman_name, angle, macro_start, macro_end):
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    
    # Clean up broken files
    for f in [FULL_YT_VIDEO, AI_CHUNK_VIDEO]:
        if os.path.exists(f): os.remove(f)

    # STEP 1: Download full YouTube video
    print(f"\n📥 Downloading {url}...")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': FULL_YT_VIDEO, 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return
        
    print("✅ Download complete!")

    # STEP 2: Pre-trim
    print(f"\n✂️ Slicing video from {macro_start}s to {macro_end}s to save AI tokens...")
    try:
        full_video = VideoFileClip(FULL_YT_VIDEO)
        actual_end = min(macro_end, full_video.duration)
        ai_chunk = full_video.subclipped(macro_start, actual_end)
        ai_chunk.write_videofile(AI_CHUNK_VIDEO, codec="libx264", audio=False, logger=None)
        full_video.close()
    except Exception as e:
        print(f"❌ Failed to cut video: {e}")
        if os.path.exists(FULL_YT_VIDEO): os.remove(FULL_YT_VIDEO)
        return
        
    os.remove(FULL_YT_VIDEO)

    # STEP 3: AI Trim
    macro_length = actual_end - macro_start
    ai_shots = get_ai_timestamps(AI_CHUNK_VIDEO, macro_length)
    
    if not ai_shots:
        print("🚨 No shots detected or AI failed. Moving to next video.")
        if os.path.exists(AI_CHUNK_VIDEO): os.remove(AI_CHUNK_VIDEO)
        return

    # STEP 4: Cut individual shots
    print(f"\n🎥 Generating final raw clips...")
    try:
        chunk_video = VideoFileClip(AI_CHUNK_VIDEO)
        
        for index, shot in enumerate(ai_shots):
            start_time = float(shot.get('start_time', 0))
            end_time = float(shot.get('end_time', 0))
            
            if end_time > chunk_video.duration or end_time <= start_time:
                continue 

            output_name = f"{batsman_name}_{angle}_{int(macro_start)}s_{index+1:02d}.mp4"
            output_path = os.path.join(RAW_VIDEOS_DIR, output_name)
            
            print(f"\n-> Isolating Shot {index+1}: {output_name} (from {start_time}s to {end_time}s)...")
            final_clip = chunk_video.subclipped(start_time, end_time)
            final_clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
            
            # STEP 5: Azure Upload
            upload_to_azure_and_delete(output_path, output_name)
            
        chunk_video.close()
    except Exception as e:
        print(f"❌ Error during final cut/upload: {e}")
    finally:
        if os.path.exists(AI_CHUNK_VIDEO):
            os.remove(AI_CHUNK_VIDEO)

def run_batch_pipeline():
    csv_file = "batch_urls.csv"
    if not os.path.exists(csv_file):
        print(f"🚨 Missing {csv_file} file. Please create it first!")
        return

    print("🚀 Starting Batch YouTube Downloader Pipeline...")
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            url = row['url'].strip()
            if not url:
                continue
                
            batsman_name = row['batsman_name'].strip()
            angle = row['angle'].strip()
            
            try:
                macro_start = float(row['macro_start_sec'])
                macro_end = float(row['macro_end_sec'])
            except ValueError:
                print(f"⚠ Skipping row due to invalid timestamps: {row}")
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing Request:")
            print(f"URL: {url}")
            print(f"Metadata: {batsman_name} ({angle}) | Window: {macro_start}s to {macro_end}s")
            print(f"{'='*50}")
            
            process_single_video(url, batsman_name, angle, macro_start, macro_end)
            
            # Sleep briefly to avoid aggressive rate limiting
            print("💤 Sleeping for 5 seconds before next video...")
            time.sleep(5)
            
    print("\n🏁 BATCH PIPELINE COMPLETE! All videos processed.")

if __name__ == "__main__":
    run_batch_pipeline()
