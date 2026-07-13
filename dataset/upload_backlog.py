import os
import time
from azure.storage.blob import BlobServiceClient

# ==========================================
# 1. AZURE CONFIGURATION
# ==========================================
AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")

# Connect to your dsucricketvault99 account
if AZURE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
else:
    blob_service_client = None

def upload_directory_to_azure(local_folder, azure_container):
    print(f"\n🚀 Scanning local '{local_folder}' folder...")
    
    if not os.path.exists(local_folder):
        print(f"⚠️ Folder '{local_folder}' not found. Skipping.")
        return

    for root, dirs, files in os.walk(local_folder):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            blob_name = os.path.relpath(local_file_path, local_folder).replace('\\', '/')
            
            blob_client = blob_service_client.get_blob_client(container=azure_container, blob=blob_name)
            
            # --- UPGRADE 1: Skip if already uploaded ---
            if blob_client.exists():
                print(f"   ⏩ Skipping (Already in cloud): {blob_name}")
                continue
            
            print(f"   ☁️ Uploading: {blob_name} -> {azure_container} container")
            
            # --- UPGRADE 2: Auto-Retry on Bad Internet ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Increased connection timeout just to be safe
                    with open(local_file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True, connection_timeout=14400)
                    break  # Success! Break out of the retry loop
                except Exception as e:
                    print(f"   ⚠️ Internet blip! Retrying {blob_name} in 5 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(5)
                    if attempt == max_retries - 1:
                        print(f"   ❌ Failed completely. Skipping {blob_name}.")
                
    print(f"✅ Finished processing '{local_folder}'!")

if __name__ == "__main__":
    print("Starting Bulletproof Cloud Migration...")
    upload_directory_to_azure("raw_videos", "raw-videos")
    upload_directory_to_azure("frames", "frames")
    print("\n🎉 Migration Complete!")