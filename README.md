## ViT for traffic sign classification
### Overview
This repo contains training and inference script for training ViT for traffic sign classification. It's trained on GTSRB dataset and scores top-5 result.
### Setup
    pip install -r requirement.txt
### Training
    python train.py --model_name_or_path "pretrained model name or path" --model_save_dir "checkpoint saving dir path"
### Inference
    python inference.py --model_name_or_path "finetuned model name or path" --image_path "path to an image to test"
Unfortunately labels in the dataset are named "Label_1", "LABEL_2" etc. This can be fixed later with label mapping.
### Result
The solution was compared with SOTA and other solutions and hits top-5 by accuracy
| Model | Accuracy, % |
| - | - |
| CNN with 3 Spatial Transformers | 99.71 |
| Sill-Net | 99.68 |
| MicronNet | 98.9% |
| ViT-Base patch 16 | 98.81 |
| SEER | 90.71% |
