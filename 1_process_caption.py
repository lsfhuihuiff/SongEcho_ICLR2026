import json
import os

# Define paths
jsonl_file = "suno70k/suno70k_metadata.jsonl"
output_dir = "./suno70k/audio_metadata"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read JSONL file and process each entry
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # breakpoint()
            # Parse JSON line
            entry = json.loads(line.strip())
            
            # Extract id and metadata
            entry_id = entry.get('id')
            metadata = entry.get('metadata', {})
            tags = metadata.get('tags', [])
            prompt = metadata.get('prompt', '')
            
            if not entry_id:
                print(f"Warning: Skipping entry with missing or invalid ID: {line.strip()}")
                continue
                
            # Write tags directly to {id}_prompt.txt
            prompt_file = os.path.join(output_dir, f"{entry_id}_prompt.txt")
            with open(prompt_file, 'w', encoding='utf-8') as pf:
                pf.write(str(tags))
                print(f"Created {prompt_file}")
                
            # Write prompt to {id}_lyrics.txt with \n as actual newlines
            lyrics_file = os.path.join(output_dir, f"{entry_id}_lyrics.txt")
            with open(lyrics_file, 'w', encoding='utf-8') as lf:
                # Replace \n with actual newlines
                formatted_prompt = prompt.replace('\\n', '\n')
                lf.write(formatted_prompt)
                print(f"Created {lyrics_file}")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in line: {line.strip()} - {e}")
        except Exception as e:
            print(f"Error processing entry: {line.strip()} - {e}")

print("Processing complete.")

import json
import os

# Input JSON file path
input_json = "./suno70k/qwen2_caption_v2_postprocess.json"

# Check if the input JSON file exists
if not os.path.isfile(input_json):
    raise FileNotFoundError(f"Input JSON not found: {input_json}")

# Load the JSON data
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each entry in the JSON data
for sample in data:
    path = sample.get("path")
    caption = sample.get("Qwen_caption", "")
    
    if not path:
        print(f"Skipping sample with missing path: {sample}")
        continue
    
    # Map the audio path to the metadata path: replace .mp3 with _prompt.txt 
    # and change the directory from /audio/ to /audio_metadata/
    prompt_path = path.replace(".mp3", "_prompt.txt").replace("/audio/", "/audio_metadata/")
    
    # Check if the target _prompt.txt file exists
    if not os.path.isfile(prompt_path):
        print(f"Prompt file not found: {prompt_path}")
        continue
    
    try:
        # Read the existing content from the prompt file
        with open(prompt_path, "r", encoding="utf-8") as f:
            existing_content = f.read().strip()
        
        # Append the new Qwen_caption to the existing content
        new_content = f"{existing_content}, {caption}" if existing_content else caption
        
        # Split content into tags, strip whitespace, and remove duplicates while preserving order
        tags = [tag.strip() for tag in new_content.split(",")]
        unique_tags = list(dict.fromkeys(tags))
        
        # Limit the number of tags to a maximum of 20
        if len(unique_tags) > 20:
            unique_tags = unique_tags[:20]
            print(f"Path: {path}\nTag count truncated to: {len(unique_tags)}\n")

        # Join the unique tags back into a comma-separated string
        unique_caption = ", ".join(unique_tags)
        
        # Determine the final output path (logic kept consistent with the mapped prompt_path)
        new_prompt_path = path.replace(".mp3", "_prompt.txt")
        
        # Write the updated content back to the _prompt.txt file
        with open(new_prompt_path, "w", encoding="utf-8") as f:
            f.write(unique_caption)
        
        print(f"Successfully updated {prompt_path} with Qwen_caption")
    
    except Exception as e:
        print(f"Error processing {prompt_path}: {str(e)}")

print("Processing completed.")

