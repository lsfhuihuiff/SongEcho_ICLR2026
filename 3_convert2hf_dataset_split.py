
from datasets import Dataset
from pathlib import Path
import os
import random
import logging
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

def create_dataset(data_dir="./data", repeat_count=2000, output_name="zh_lora_dataset", val_size=1000):
    data_path = Path(data_dir)
    all_examples = []

    # 遍历 MP3 文件
    for song_path in data_path.glob("*.mp3"):
        prompt_path = str(song_path).replace(".mp3", "_prompt.txt")
        lyric_path = str(song_path).replace(".mp3", "_lyrics.txt")
        try:
            if not os.path.exists(prompt_path):
                logging.warning(f"Prompt file {prompt_path} does not exist.")
                continue
            if not os.path.exists(lyric_path):
                logging.warning(f"Lyrics file {lyric_path} does not exist.")
                continue
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()
            
            keys = song_path.stem
            example = {
                "keys": keys,
                "filename": str(song_path),
                "tags": prompt.split(", "),
                "speaker_emb_path": "",
                "norm_lyrics": lyrics,
                "recaption": {}
            }
            all_examples.append(example)
        except Exception as e:
            logging.error(f"Error processing {song_path}: {str(e)}")
            continue

    if not all_examples:
        logging.error("No valid examples found.")
        raise ValueError("No valid examples found.")

    # 随机抽取验证集
    if len(all_examples) < val_size:
        logging.warning(f"Only {len(all_examples)} examples available, using {len(all_examples)//2} for validation.")
        val_size = len(all_examples) // 2
    
    val_examples = random.sample(all_examples, val_size)
    train_examples = [ex for ex in all_examples if ex not in val_examples]
    
    # 重复数据集
    train_ds = Dataset.from_list(train_examples * repeat_count)
    val_ds = Dataset.from_list(val_examples * repeat_count)

    # 保存训练集和验证集
    train_output = f"{output_name}_train"
    val_output = f"{output_name}_val"
    train_ds.save_to_disk(train_output)
    val_ds.save_to_disk(val_output)
    
    logging.info(f"Created training dataset with {len(train_examples)} unique samples, repeated {repeat_count} times, saved to {train_output}")
    logging.info(f"Created validation dataset with {len(val_examples)} unique samples, repeated {repeat_count} times, saved to {val_output}")

def main():
    parser = argparse.ArgumentParser(description="Create a dataset from audio files with train-validation split.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the audio files.")
    parser.add_argument("--repeat_count", type=int, default=1, help="Number of times to repeat the dataset.")
    parser.add_argument("--output_name", type=str, default="zh_lora_dataset", help="Name of the output dataset.")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of samples for validation set.")
    args = parser.parse_args()

    create_dataset(
        data_dir=args.data_dir,
        repeat_count=args.repeat_count,
        output_name=args.output_name,
        val_size=args.val_size
    )

if __name__ == "__main__":
    main()
