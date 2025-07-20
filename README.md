# Plant Leaf Image Classification using Transfer Learning

A Deep Learning Assignment for CSE366

---

## Project Overview

This project implements a robust solution for plant leaf classification using the power of **transfer learning**. We leverage state-of-the-art pre-trained Convolutional Neural Network (CNN) architectures, fine-tuning them to accurately identify 22 different types of plant leaves. The entire workflow—from data preprocessing and model configuration to training, evaluation, and performance comparison—is meticulously documented in the accompanying Google Colab notebook.

## Group Members

| Name                      | ID              |
| ------------------------- | --------------- |
| MD. Nasibul Islam Sazid   | 2022-3-60-014   |
| Sababa Fairoze Prionty    | 2022-3-60-229   |

---

## Repository Contents

This repository is structured to provide a complete and reproducible overview of our work.

| File/Folder                | Description                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| `CustomCNN_CSE366.ipynb`   | **Main Notebook:** Contains the end-to-end Python code for data loading, model training, and evaluation. |
| `custom_cnn_best.pt`       | **Best Model Weights:** The saved checkpoint of our highest-performing model based on validation F1-score.  |
| `ModelDesign_Rationale.pdf`| **Design Document:** A one-page brief with our model's architecture, design justifications, and key hyperparameters. |
| `README.md`                | **You are here:** Instructions and project summary.                                                     |

---

## Model Design & Technical Approach

Our approach is centered on transfer learning to leverage features learned from large-scale datasets, ensuring high performance and efficient training.

### 1. Conceptual Architecture

The model follows a standard and effective pipeline for image classification:

1.  **Input Layer:** Accepts raw plant leaf images.
2.  **Preprocessing & Augmentation Layer:**
    * **Resize:** All images are standardized to `(160, 160)` pixels.
    * **Data Augmentation:** `TrivialAugmentWide()` is applied during training to create variations of images, which helps the model generalize better and reduces overfitting.
    * **Tensor Conversion:** Images are converted to PyTorch tensors.
    * **Normalization:** Pixel values are normalized using standard ImageNet mean and standard deviation for model stability.
3.  **Pre-trained CNN Feature Extractor:** The powerful convolutional base of a pre-trained model (like `EfficientNet-B0`) is used to extract meaningful, high-level features from the images.
4.  **Custom Classification Head:** A new set of fully connected layers is attached to the feature extractor. This head is trained to map the extracted features to the specific plant leaf classes in our dataset.
5.  **Output Layer:** Produces a probability distribution over the **22 plant leaf categories**.

### 2. Key Hyperparameters

Reproducibility and optimal performance are governed by a set of carefully chosen hyperparameters.

| Hyperparameter  | Value               | Justification                                                                            |
| --------------- | ------------------- | ---------------------------------------------------------------------------------------- |
| `SEED`          | `42`                | Ensures that all random operations are deterministic for reproducible experiments.       |
| `DATA_FRACTION` | `0.1` (for dev)     | Allows for rapid prototyping and debugging on a smaller subset of the data.              |
| `BATCH_SIZE`    | `32`                | Provides a good balance between computational speed and stable gradient estimation.       |
| `IMG_SIZE`      | `(160, 160)`        | Standardizes input dimensions for model compatibility and consistent feature extraction. |
| `LEARNING_RATE` | `0.001`             | A proven and effective starting learning rate for the Adam optimizer.                    |
| `EPOCHS`        | `15`                | A sufficient number of training iterations to achieve convergence without overfitting.   |
| `DEVICE`        | `'cuda' or 'cpu'`   | Automatically utilizes GPU for accelerated training if available, ensuring efficiency.     |

---

## Model Performance Comparison

We fine-tuned and evaluated several well-regarded pre-trained models to identify the best-performing architecture for this task. The results on the test set are as follows:

| Model             | Test Accuracy |
| ----------------- | ------------- |
| ResNet50          | 70.38%        |
| MobileNet V2      | 75.45%        |
| **EfficientNet B0** | **81.94%** |

**Conclusion:** `EfficientNet-B0` demonstrated superior performance, making it our model of choice for this classification task.

---

## How to Reproduce

Follow these steps to replicate our results.

### 1. Environment Setup
* **Platform:** Open the project in a **Google Colab** environment.
* **Hardware Accelerator:** Enable a GPU for training (`Runtime` -> `Change runtime type` -> `T4 GPU`).

### 2. Dataset
* **Source:** The project uses the **Plant Leaves for Image Classification** dataset.
* **Link:** [https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification)
* **Setup in Colab:** To run the notebook, you will need to download the dataset from Kaggle. The easiest way is to upload your `kaggle.json` API token to Colab and use the Kaggle API to download and unzip the data directly.

### 3. Execution
1.  Clone this GitHub repository or upload the `CustomCNN_CSE366.ipynb` notebook to your Colab environment.
2.  Follow the instructions within the notebook to download and set up the dataset.
3.  **Run All Cells:** Execute the notebook from top to bottom. It is self-contained and will automatically handle:
    * Data loading and preprocessing.
    * Model building and configuration.
    * Training and validation loops.
    * Final evaluation and generation of performance metrics.
"""
