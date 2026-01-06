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


