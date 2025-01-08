---

<div align="center">

# UltraSound Foundation Model (USFM)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

---

### ✨✨✨ 
---



## 📌 Configuring the runtime environment

### 1. Configuring the project

```bash
# clone project
git clone https://github.com/openmedlab/USFM.git
cd USFM

# [OPTIONAL] create conda environment
conda create -n USFM python=3.9
conda activate USFM

# install pytorch according to instructions
# https://pytorch.org/get-started/
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt

# install mmcv
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html


# install mmsegmentation [important: from modified mmseg]
mkdir -p useful_modules
cd useful_modules
git clone git@github.com:George-Jiao/mmsegmentation.git
cd mmsegmentation
git checkout gj_mmcv2_2_0
pip install -v -e .
cd ../..

```

### 2. Installing usdsgen (US DownStream Generalizer)

usdsgen is a USFM-based ultrasound downstream task generalization package that can be used for downstream tasks on ultrasound images.

```bash
pip install -v -e .
```

## 📦️ Data preparation

### 1. Datasets Folder

You can save datasets in either folder, the default is the folder \[datasets\].

The folder format is generally:

```bash
datasets/
    ├── Seg/
        ├── dataset_names/
            ├── trainning_set/
                ├── image/ img1.png..
                ├── mask/ img1.png..
            ├── val_set/
                ├── image/
                ├── mask/
            ├── test_set/
                ├── image/
                ├── mask/
    |── Cls/
        ├── dataset_names/
            ├── trainning_set/
                |── class1/
                |── class2/
            ├── val_set/
                |── class1/
                |── class2/
            ├── test_set/
                |── class1/
                |── class2/
```

\*\*\*\* Advanced: data configuration in folder \[configs/data/\]


## 🚀 Finetuning USFM on the downstream dataset

### 1. Download the USFM weights

Download the USFM weight from Google Drive [USFM_latest.pth](https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view) and save it in \[./assets/FMweight/USFM_latest.path\].

### 2. Finetuning USFM for the downstream task

```bash
# setting the environment variable
export batch_size=16
export num_workers=4
export CUDA_VISIBLE_DEVICES=0,1,2
export devices=3 # number of GPUs
export dataset=toy_dataset
export epochs=400
export pretrained_path=./assets/FMweight/USFM_latest.pth
export task=Seg   # Cls for classification, Seg for segmentation
export model=Seg/SegVit # SegVit or Upernet for segmentation, vit for classification

# Segmentation task
python main.py experiment=task/$task data=Seg/$dataset data="{batch_size:$batch_size,num_workers:$num_workers}" \
    model=$model model.model_cfg.backbone.pretrained=$pretrained_path \
    train="{epochs:$epochs, accumulation_steps:1}" L="{devices:$devices}" tag=USFM


# Classification task
export task=Cls
export model=Cls/vit
python main.py experiment=task/$task data=Cls/$dataset data="{batch_size:$batch_size,num_workers:$num_workers}" \
    model=$model model.model_cfg.backbone.pretrained=$pretrained_path \
    train="{epochs:$epochs, accumulation_steps:1}" L="{devices:$devices}" tag=USFM
```

## 📈 Results Folder

The results of the experiment are saved in the logs/fineturne folder.








```
