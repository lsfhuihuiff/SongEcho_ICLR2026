<div align="center">

# 🎵 SongEcho
## Towards Cover Song Generation via Instance-Adaptive Element-wise Linear Modulation


[![Demo Page](https://img.shields.io/badge/Project-Demo_Page-blue?logo=github&logoColor=white)](https://vvanonymousvv.github.io/SongEcho_updated/)
[![ICLR Paper](https://img.shields.io/badge/Paper-ICLR_2026-b31b1b?logo=arxiv&logoColor=white)](https://openreview.net/forum?id=TEKOayiQg2)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](./LICENSE)

</div>

---

## 📝 Abstract

SongEcho achieves our cover song generation by simultaneously synthesizing new vocals and accompaniment conditioned on the original vocal melody and text prompts. Leveraging Instance-Adaptive Element-wise Linear Modulation (IA-EiLM), SongEcho achieves state-of-the-art fidelity with <30% trainable parameters compared to baselines.

### ✨ Key Innovations

IA-EiLM Framework: Integrates EiLM for precise temporal alignment (superior to FiLM) and IACR for instance-adaptive condition refinement.

Suno70k Dataset: A newly constructed open-source dataset containing 70k high-quality songs with comprehensive annotations.

---

## 📦 Installation

We recommend using `conda` to manage the environment.

### 1. Set up Environment
```bash
# Create and activate a new conda environment
conda create -n songecho python=3.10 -y
conda activate songecho

```

### 2. Install Dependencies

```bash
# Install PyTorch (Adjust CUDA version as needed)
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Clone the repository and install requirements
git clone [https://github.com/vvanonymousvv/SongEcho.git](https://github.com/vvanonymousvv/SongEcho.git)
cd SongEcho
pip install -e .

```

---
## 📥 Resource Preparation

Before running data processing or inference, please download the necessary resources.

### 1. Model Checkpoints
Download the model files and place them in the `./checkpoints` directory.

- **Link:** [Download Models via Google Drive](https://drive.google.com/drive/folders/1aJH9EG2_zI53oIs4wGQXB6fF8AKxqbgm?usp=sharing)
- **Target Path:** `./checkpoints/`

### 2. Dataset Metadata
Download the dataset metadata and place the files in the `suno70k` directory.

- **Link:** [Download Metadata via Google Drive](https://drive.google.com/drive/folders/1BgaEVggT6JipF04W71yewoTmRyYuV0kl?usp=sharing)
- **Target Path:** `./suno70k/`

## 📊 Data Processing

To train SongEcho, we utilize the **Suno70k** dataset, which is constructed based on the [**nyuuzyou/suno**](https://huggingface.co/datasets/nyuuzyou/suno) dataset. Please follow the steps below to prepare your data.

> ⚠️ **Note:** Ensure your data is organized according to the structure in the `examples/` directory.

| Step | Script | Description |
| --- | --- | --- |
| **0** | `0_download_audio.py` | Downloads audio files from the source URLs in the metadata. |
| **1** | `1_process_caption.py` | Processes metadata and Qwen2-Audio captions to generate `lyrics` and `prompt` files. |
| **2** | `2_extract_f0.py` | Extracts the vocal melody (F0) from the audio for conditioning. |
| **3** | `3_convert2hf_dataset_split.py` | Splits the data into train/val sets and converts to HF format. <br>⚠️ **Note:** Due to the randomness of the split, we provide our official validation set `suno70k_prompt_val` for reproducibility. |

**Quick Start:**

```bash
# Run the pipeline sequentially
python 0_download_audio.py
python 1_process_caption.py
python 2_extract_f0.py
python 3_convert2hf_dataset_split.py --data_dir ./suno70k/audio --repeat_count 1 --output_name suno70k

```

Here is the updated **Inference** and **Training** section with the modified paths and scripts.

```markdown
---

## 🚀 Inference

Ensure your model checkpoints are located in the `checkpoints/` directory. Run the following command to start generation:

```bash
python inference.py \
    --output_dir ./val_results \
    --pt_path ./checkpoints/melody.pt \
    --dataset_path /path/to/val_dataset 

```

### ⚙️ Arguments Explanation

| Argument | Description |
| --- | --- |
| `--output_dir` | The directory where generated audio files will be saved. |
| `--pt_path` | Path to the specific model checkpoint file (e.g., `.melody.pt` file). |
| `--dataset_path` | Path to the validation dataset folder. |

---

## 🔨 Training

### Start Training

1. **Configure:** Open `train.sh` and adjust the paths (data, logs) and hyperparameters (batch size, learning rate) according to your environment.
2. **Launch:** Run the script to start the training process.

```bash
bash train.sh
```

## 📖 Citation

If you find our work or the **Suno70k** dataset useful, please consider citing:

```bibtex
@misc{gong2025acestep,
    title={ACE-Step: A Step Towards Music Generation Foundation Model},
    author={Junmin Gong, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo}, 
    howpublished={\url{[https://github.com/ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)}},
    year={2025},
    note={GitHub repository}
}

@article{zhang2024gtsinger,
  title={Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks},
  author={Zhang, Yu and Pan, Changhao and Guo, Wenxiang and Li, Ruiqi and Zhu, Zhiyuan and Wang, Jialei and Xu, Wenhao and Lu, Jingyu and Hong, Zhiqing and Wang, Chuxin and others},
  journal={arXiv preprint arXiv:2409.13832},
  year={2024}
}

```

---
