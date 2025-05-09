# Visual Defect Classifier for Lab Components (ResNet50 Transfer Learning + Streamlit)

## Project Overview

This repository contains a prototype system for automatic visual inspection of lab equipment components. The solution uses transfer learning on a ResNet50 CNN to classify parts as **good** or **defective** and provides an interactive Streamlit web app for inference. 

**Key Highlights:**

* **Binary Classification** of component images using a ResNet50 backbone
* **Transfer Learning** implemented directly in PyTorch
* **Streamlit UI** for uploading custom images, adjusting sensitivity, and visualizing confidence
* **Efficient Inference**: Model weights saved as `.pth` files, compatible with CPU or GPU

## Technical Specifications

### 1. Model Architecture & Training

| Component        | Specification                                                |
| ---------------- | ------------------------------------------------------------ |
| Base Model       | ResNet-50 (pre-trained on ImageNet)                          |
| Custom Head      | Flatten → Dropout(0.3) → Linear(2048→1) → Sigmoid activation |
| Input Resolution | 224×224 RGB                                                  |
| Framework        | PyTorch                                                      |
| Loss Function    | Binary Cross-Entropy with Logits                             |
| Optimizer        | Adam (lr=1e-4 initial; lr=1e-5 fine-tune)                    |
| Scheduler        | ReduceLROnPlateau (factor=0.5; patience=2–3)                 |

### 2. Data Loading & Augmentation

* **Directory Structure**: `data/` → `good/` & `defect/` subdirectories
* **Train/Validation Split**: 80/20 random split via `torch.utils.data.random_split`
* **Transforms (Train)**:

  * `RandomResizedCrop(224)`
  * `RandomHorizontalFlip()`
  * `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
  * `Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])`
* **Transforms (Validation)**:

  * `Resize(256)` → `CenterCrop(224)`
  * `Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])`
* **Batch Size**: 16
* **Training Schedule**: 10 epochs (head) + 5 epochs (fine-tune)

### 3. Performance (Example)

| Metric         | Example Value                |
| -------------- | ---------------------------- |
| Validation Acc | \~90–95% (dataset-dependent) |
| Inference Time | \~20–50 ms/image (CPU)       |
| Model Size     | \~100 MB (`.pth`)            |

**Note**: Inference will automatically use GPU if available.

## Key Features

* **Interactive UI**: Upload any component image or select preloaded samples
* **Adjustable Threshold**: Tune defect probability cutoff in real time
* **Confidence Bar Chart**: Visualize model predictions for both classes

## Installation & Usage

### Requirements

* Python 3.8+
* PyTorch 1.12+
* torchvision
* Streamlit
* PIL

```bash
# 1. Clone the repository
git clone https://github.com/harishan0/BioTechne-Prototype.git
cd BioTechne-Prototype

# 2. Install dependencies
pip install -r requirements.txt
```

### Model Training (Optional)

1. Open `finetune.py` in Google Colab
2. Mount Google Drive and set `data_dir` to your `data/` folder
3. Run the script, which outputs:

   * `defect_resnet50_head.pth` (initial head training)
   * `defect_resnet50_finetuned.pth` (fine-tuned model)

### Running the Streamlit App

```bash
# Ensure the .pth file is in the project root or update MODEL_PATH in main.py
streamlit run main.py
```

1. Use the sidebar to upload a custom image or select sample components
2. Adjust the defect threshold to calibrate sensitivity
3. View prediction probabilities and pass/fail status

---

*Built by Harishan Ganeshanathan for Bio-Techne SWE Internship Evaluation.*
