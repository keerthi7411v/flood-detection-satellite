🛰️ Flood Detection from Satellite SAR Imagery

Deep Learning-based flood detection system using U-Net architecture to identify flood-affected areas from Synthetic Aperture Radar (SAR) satellite images.

 📌 Overview

This project implements an end-to-end pipeline for flood segmentation using deep learning. The model processes SAR satellite imagery and outputs binary flood masks for disaster management.

🚀 Features

- U-Net Architecture: Custom implementation with encoder-decoder and skip connections
- Data Augmentation: Elastic deformation, rotation, scaling, flipping
- Mixed Precision Training: Faster training with AMP
- Sliding Window Inference: Handles images of any size

📊 Results

| Metric | Value |
|--------|-------|
| Validation IoU (Synthetic Data) | **98.55%** |
| Real-world Test | Identified overfitting gap — key learning for production ML |

🛠️ Tech Stack

- PyTorch
- U-Net (CNN)
- Albumentations
- NumPy, Matplotlib
- Google Colab (GPU training)

📁 Files

- `flood_detector.py` — Main training & prediction script
- `model.py` — U-Net architecture
- `data.py` — Dataset loader & preprocessing

🎯 Key Learnings

1. Built complete ML pipeline from data ingestion to prediction
2. Achieved high accuracy on synthetic data (98.55% IoU)
3. Discovered critical overfitting gap when testing on real-world imagery
4. Learned SAR preprocessing and satellite data handling

👤 Author

KEERTHI R




