# ASL Hand Gesture Recognition – Machine Learning Evaluation
This repository contains a comprehensive pipeline for recognizing American Sign Language (ASL) finger-spelling (Letters A-J) using both supervised and unsupervised machine learning techniques. This project was developed as part of the Artificial Intelligence module.

## Features
- Data Extraction: Automated extraction of 21 3D hand landmarks using Google MediaPipe.

- Custom KNN: A K-Nearest Neighbors classifier built entirely from scratch using only standard Python libraries (math, numpy).

- Model Comparison: Comparison of KNN against Decision Trees and Random Forest models.

- Unsupervised Exploration: Dimensionality reduction (PCA, t-SNE) and clustering (K-Means, Hierarchical) to identify natural groupings in hand-pose data.

- Optimization: Implementation of 5-fold cross-validation for hyperparameter tuning.

- ROOT DIR/
├── data/
│   ├── extracted_features/      # Raw and sanitised CSV landmark data
│   └── CW2_dataset_final/       # (Excluded/External) Source images
├── src/
│   ├── data extraction/         # Scripts for landmark extraction and cleaning
│   ├── supervised learning/     # KNN (Scratch), Decision Trees, and Random Forest
│   ├── unsupervised learning/   # K-Means, Hierarchical clustering, PCA, and t-SNE
│   └── generate_dataset.py      # Utility for creating initial data structures
├── A114_POSTER.pdf              # Visual abstract of results
└── A114_PRESENTATION.pdf        # Group presentation slides


##  Installation & Setup
1. Clone the repository:
```
git clone https://github.com/your-username/asl-gesture-recognition.git
cd asl-gesture-recognition
```

2. Create the Environment: It is recommended to use the specific AI-CW2 conda environment:
```
 conda activate AI-CW2
 ```

3. Run Data Sanitisation:
```
python "src/data extraction/sanitiser.py"
```

## Methodology

**Data Preprocessing**
To ensure model robustness, the raw landmark data underwent:

Centering: Translating all landmarks relative to the wrist.

Normalisation: Scaling coordinates based on the distance of the middle finger MCP to account for different hand sizes and distances from the camera.

**Supervised Learning**
The models were evaluated using a 60/20/20 split (Train/Validation/Test).
- KNN (Scratch): Optimized using $k=3$ and Euclidean distance.
- Random Forest: Achieved the highest performance with an accuracy of 96.35%.

**Unsupervised Learning**
Used to explore if ASL signs form natural clusters without labels. While t-SNE provided clear local groupings, the significant overlap between similar signs (like 'M' and 'N' or 'E' and 'S') resulted in a moderate Adjusted Rand Index (ARI) of 0.333.

