# GMTRouter: Personalized LLM Router Over Multi-turn User Interactions

<p align="center">
    <a href="https://github.com/ulab-uiuc/GMTRouter/tree/main">
        <img alt="Build" src="https://img.shields.io/badge/Project-Page-blue">
    </a>
    <a href="https://github.com/ulab-uiuc/GMTRouter/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <br>
    <a href="https://github.com/ulab-uiuc/GMTRouter">
        <img alt="Build" src="https://img.shields.io/github/stars/ulab-uiuc/GMTRouter">
    </a>
    <a href="https://github.com/ulab-uiuc/GMTRouter">
        <img alt="Build" src="https://img.shields.io/github/forks/ulab-uiuc/GMTRouter">
    </a>
    <a href="https://github.com/ulab-uiuc/GMTRouter">
        <img alt="Build" src="https://img.shields.io/github/issues/ulab-uiuc/GMTRouter">
    </a>
</p>


<p align="center">
    <a href="https://github.com/ulab-uiuc/GMTRouter/tree/main">🌐 Project Page</a> |
    <a href=>📜 arXiv</a> 
<p>


<div align="center">
  <img src="./figures/pipeline.png" width="700" alt="GoR">
</div>



## News
**[2025.10.28]** 🌟 The **GMTRouter** repository is now open-sourced, and the paper will be publicly available on arXiv shortly.



## 📌Preliminary

### Environment Setup

We recommend creating a separate conda environment for GMTRouter to avoid dependency conflicts.

```bash
# 1️⃣ Create a new conda environment with Python 3.11
conda create -n GMTRouter python=3.11.13
conda activate GMTRouter

# 2️⃣ Install PyTorch for your GPU version.
# Here we provide a GPU setup for CUDA 12.4.
# If you use a different CUDA version or CPU, please modify accordingly.
pip install torch==2.6.* torchvision==0.21.* torchaudio==2.6.* --index-url https://download.pytorch.org/whl/cu124

# 3️⃣ Install other required Python packages
pip install -r requirements.txt

# 4️⃣ Install PyTorch Geometric (PyG) and related libraries
# This ensures compatibility with PyTorch 2.6 and CUDA 12.4
pip install torch-geometric==2.6.1
````

**Notes / Tips:**

* Make sure your CUDA driver version matches the PyTorch CUDA version you install. Mismatched versions can cause runtime errors.
* If you only want a CPU setup, you can remove the `--index-url` or choose the CPU-specific wheels from PyTorch’s website.
* `requirements.txt` contains standard dependencies; `requirements-gpu-cu124.txt` is tailored for GPU with CUDA 12.4.


### Dataset Preparation

The datasets used in our experiments have been processed into a unified format and split into **training**, **validation**, and **test** sets.

You can download the data manually from Google Drive via the following link:
[Download from Google Drive](https://drive.google.com/uc?export=download&id=1l-sAXV7i08S9sjJw1_UyvD0mP56QSIKt)

After downloading, decompress the archive **`GMTRouter_dataset.tar.gz`** and move the extracted **`data`** folder to the root directory of the repository:

```bash
tar -xzvf GMTRouter_dataset.tar.gz
mv GMTRouter_dataset/data <destination_path>  # Replace <destination_path> with the root directory of this project
```

## ⭐Experiments

We provide a sample configuration file at **`./configs/sample.yaml`**,  
where you can specify which dataset to run experiments on.  
Supported datasets include:

- `chatbot_arena`
- `gsm8k`
- `mmlu`
- `mt_bench`

To launch an experiment, run the following command, **replacing `<YAML>` with your own configuration file**:

```bash
python -u ./src/train.py --config ./configs/<YAML>
````

For example:

```bash
python -u ./src/train.py --config ./configs/sample.yaml
```

Each configuration file defines dataset selection, training hyperparameters, and model settings.
You can modify it to fit your own experimental setup or dataset.



<!-- ## Citation

```bibtex
@inproceedings{Xie2025GMTRouter,
  title={GMTRouter: Personalized LLM Router Over Multi-turn User Interactions},
  author={Xie, Encheng and Sun, Yihang and Feng, Tao and You, Jiaxuan},
  year={2025}
}
``` -->

