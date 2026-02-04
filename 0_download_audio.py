import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Input and output paths
input_file = "./suno70k/suno70k_metadata.jsonl"
output_dir = "./suno70k/audio"
failed_ids_file = "./suno70k/failed_download_ids.txt"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(failed_ids_file), exist_ok=True)

# Function to download a single audio file
def download_audio(entry):
    try:
        audio_url = entry.get("audio_url")
        audio_id = entry.get("id")
        
        if not audio_url or not audio_id:
            return f"Skipping entry missing audio_url or id: {entry.get('id', 'unknown')}", audio_id, False
        
        # Use audio ID as the filename
        parsed_url = urlparse(audio_url)
        ext = os.path.splitext(parsed_url.path)[1] or ".mp3"  # Default extension is .mp3
        output_path = os.path.join(output_dir, f"{audio_id}{ext}")

        # Check if file already exists
        if os.path.exists(output_path):
            return f"File already exists, skipping: {output_path}", audio_id, True

        # Download the file
        response = requests.get(audio_url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return f"Downloaded: {output_path}", audio_id, True
        else:
            return f"Download failed {audio_url}: Status Code {response.status_code}", audio_id, False
    except Exception as e:
        return f"Error downloading {audio_url}: {str(e)}", audio_id, False

# Read JSONL file and collect entries
entries = []
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():  # Skip empty lines
            try:
                json_obj = json.loads(line)
                if json_obj.get("audio_url") and json_obj.get("id"):
                    entries.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {line.strip()}. Error: {e}")
                continue

# Download audio files using multi-threading and track failed IDs
failed_ids = []
max_workers = 10  # Adjust based on system resources and server limits
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks and create a mapping of futures to entries
    future_to_entry = {executor.submit(download_audio, entry): entry for entry in entries}
    for future in as_completed(future_to_entry):
        result, audio_id, success = future.result()
        print(result)
        if not success:
            failed_ids.append(audio_id)

# Save failed IDs to a TXT file
if failed_ids:
    with open(failed_ids_file, "w", encoding="utf-8") as f:
        for audio_id in failed_ids:
            f.write(f"{audio_id}\n")
    print(f"\nSaved {len(failed_ids)} failed download IDs to: {failed_ids_file}")
else:
    print("\nAll audio files downloaded successfully. No failed IDs found.")