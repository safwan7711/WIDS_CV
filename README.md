# WIDS_CV
### Project Overview
This repository contains the practical implementations and exercises for the Foundations of Computer Vision intensive course as beginner. 

### Week 1: Image Fundamentals & Processing
Understanding pixel representation and color spaces (RGB, HSV, Grayscale).
Applying Gaussian&Median blurring for noise reduction and analyzing image gradients.
Implementing Canny operators to identify structural boundaries.
Applying gradient and laplacian filters also laplacian of gaussian to visualize the edges 

**Implementation:**
`blur_filters.py` & `edge_detection.py`: simple code implementation that loads raw imagery, converts color spaces, applies kernel-based filtering, and extracts edge maps using the Canny algorithm.

For understanding the math behind those filters I referred opencv tutorials, fpcv.cs.columbia.edu and other wikipedia stuffs.And for implemention in python and basic intro to cv youtube videos were helpful.

###  Week 2: Feature Extraction & Image Stitching Pipeline
**Keypoint Detection:** Studied the mathematical part of identifying distinct image features (corners and blobs) using intensity gradients ($I_x, I_y$) and the Second Moment Matrix.
**Feature Description:** Analyzed how algorithms encode local image appearance into vectors. Specifically focused on **ORB** (Oriented FAST and Rotated BRIEF), which uses binary strings for efficient description.
**Feature Matching:** Implemented descriptor matching using **Hamming Distance** for binary vectors, as opposed to Euclidean distance used in SIFT.
**Robust Estimation:** Utilized **RANSAC** (Random Sample Consensus) to statistically filter outlier matches and compute a reliable Homography matrix with 8 degrees of freedom.
**Geometric Transformation:** Applied the calculated Homography matrix to warp images into a common coordinate system for panoramic stitching.

**Implementation Details (`classical_pipeline.py`):**
* Developed a full image stitching pipeline that automatically benchmarks on standard datasets (OpenCV Boat).
* Detects top 2000 keypoints using ORB.
*  Computes binary descriptors and performs Brute Force Matching with Cross-Checking enabled.
* Filters matches to the top 15% to ensure quality,removing weak matches.
* Estimates the $3 \times 3$ Homography matrix using RANSAC with a reprojection threshold of 5.0 pixels.
* Warps the secondary image using perspective transformation and blends it with the primary image to generate a seamless panorama `output_panorama.jpg`.

# WIDS FINAL PROJECT
# Comparative Study of Visual Representations for Image Classification

## 1. Project Overview

The goal of this project is to compare **two different visual representation approaches** for image classification:

- **Pipeline A (Classical Computer Vision)**  
  Hand-crafted features using **ORB descriptors** combined with a traditional **k-Nearest Neighbors (kNN)** classifier.

- **Pipeline B (Deep Learning)**  
  Automatically learned features using a **Convolutional Neural Network (CNN)** based on **MobileNetV2**.

The comparison focuses not only on classification accuracy, but also on **error patterns and limitations** of each approach, as required in the project description.

---

## 2. Dataset

- **Dataset:** CIFAR-10  
- **Number of classes:** 10  
- **Train / Validation split:**
  - Training set: 45,000 images
  - Validation set: 5,000 images

The **same dataset split and preprocessing** are used for both pipelines to ensure a fair comparison.
---

## 3. Pipeline A: ORB + Bag of Visual Words + kNN

### 3.1 Visual Representation

- **ORB (Oriented FAST and Rotated BRIEF)** is used to extract local keypoints and binary descriptors.
- Since each image produces a variable number of descriptors, a **Bag of Visual Words (BoVW)** approach is applied:
  - ORB descriptors from training images are clustered using **K-Means**.
  - Each image is represented as a **fixed-length histogram** of visual word occurrences.

### 3.2 Classifier

- A **k-Nearest Neighbors (kNN)** classifier is trained on the BoVW histograms.
- Distance metric: Euclidean distance.

### 3.4 Performance

- Validation accuracy is approximately **14%**, slightly above random guessing (10%).
- A **confusion matrix** is used to analyze class-wise misclassifications.

### 3.5 Observations

- ORB relies on corner and edge features.
- CIFAR-10 images are low-resolution and lack strong corner structures.
- BoVW discards spatial information, limiting discriminative power.

---

## 4. Pipeline B: CNN (MobileNetV2)

### 4.1 Visual Representation

- A **Convolutional Neural Network** learns hierarchical visual features directly from pixel data.
- **MobileNetV2** pretrained on ImageNet is used as the backbone.
- The final classification layer is replaced to match the CIFAR-10 classes.

### 4.2 Training

- Loss function: Cross-entropy loss
- Optimizer: Adam
- Training is performed for a small number of epochs due to computational constraints.
- Validation accuracy is monitored to evaluate performance.

### 4.3 Performance

- The CNN achieves **significantly higher accuracy** than the classical pipeline. **90%**
- The confusion matrix shows fewer misclassifications and stronger diagonal dominance.

### 4.4 Observations

- CNNs learn both low-level (edges, textures) and high-level (object structure) features.
- Spatial relationships are preserved, unlike in BoVW.
- CNNs are computationally more expensive but much more expressive as it took more time.

---

## 5. Comparative Analysis

| Aspect | Pipeline A (ORB + kNN) | Pipeline B (CNN) |
|------|------------------------|------------------|
| Feature type | Hand-crafted | Learned automatically |
| Spatial information | Lost (BoVW) | Preserved |
| Accuracy | Low (~14%) | High |
| Interpretability | High | Lower |
| Computational cost | Low | High |

### Key Insight

The experiment demonstrates that **classical hand-crafted features struggle** on complex, low-resolution image datasets like CIFAR-10, whereas **CNNs perform significantly better** by learning task-specific representations directly from data.

---

## 6. Error Analysis

- Confusion matrices reveal that Pipeline A frequently confuses visually similar classes.
- Pipeline B shows fewer confusions and better class separation.
- This highlights the limitations of classical representations compared to deep learning approaches.
- you can see that the values in the diagonals are big unlike in pipeline A.

---

## 7. Conclusion

This project shows that:
- Classical visual representations such as ORB + BoVW are simple and computationally efficient but limited in performance.
- CNN-based representations are significantly more powerful for image classification.
- Validation-based hyperparameter tuning and confusion matrix analysis provide meaningful insights beyond overall accuracy.

---


## 8. Libraries Used

- NumPy  
- OpenCV  
- scikit-learn  
- PyTorch  
- Torchvision  
- Matplotlib  






