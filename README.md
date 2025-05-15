# SigLIP2 Fine-Tuning for Verb Classification

This repository contains a PyTorch implementation for fine-tuning the **SigLIP2 ViT-SO400M-16** model on the CholecT45 dataset for multi-label verb classification in surgical video frames.

## 💡 Highlights

* Uses **SigLIP2** from OpenCLIP with a custom classification head.
* Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) as preprocessing.
* Supports **full model unfreezing** for fine-tuning.
* Tracks metrics with **Weights & Biases (wandb)**.
* Evaluates using **macro mAP** (mean average precision).

## 📁 Folder Structure

```
.
├── siglip2_finetune.py  # Main training + evaluation script
├── weights/             # Pretrained SigLIP2 weights
├── weights_finetune/    # Fine-tuned model checkpoints
├── *.json               # Input dataset files
```

## 🔧 Requirements

* Python >= 3.8
* PyTorch >= 1.13
* OpenCLIP
* torchvision
* scikit-learn
* wandb
* opencv-python

## 🚀 Usage

Run training and evaluation:

```bash
python siglip2_finetune.py
```

## 🧪 Performance

* **Maximum Validation Accuracy (macro mAP):** `62%`
* **Test Accuracy (macro mAP):** `62%`

## 📌 Notes

* Only **CLAHE** preprocessing is used (DCP was removed).
* Use your own Weights & Biases API key if needed.

## 📬 Contact

For questions or collaboration, feel free to reach out.

---

Made with ❤️ for surgical action understanding.
