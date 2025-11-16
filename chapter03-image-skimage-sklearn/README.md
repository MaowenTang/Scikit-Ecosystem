# 📘 Chapter 3 — Image Processing with scikit-image + scikit-learn

*A practical introduction to classical computer vision using Python.*

---

## 1. Chapter Goals

After completing this chapter, you will be able to:

* Understand the role of **scikit-image** in the scikit ecosystem
* Perform essential image operations:

  * reading images
  * converting to grayscale
  * resizing
  * filtering
  * edge detection
* Extract handcrafted features (HOG, LBP)
* Train ML models (SVM / Logistic Regression) using extracted image features
* Build a **complete traditional CV pipeline**: preprocessing → feature extraction → classification

This chapter is **self-contained** and provides a fully runnable notebook example.

---

## 2. Relationship Map: scikit-image + scikit-learn

```mermaid
flowchart LR
    A[Image Files] --> B[scikit-image<br>Image Loading & Preprocessing]
    B --> C[Feature Extraction<br>(HOG / LBP / Stats)]
    C --> D[scikit-learn<br>ML Model]
    D --> E[Classification / Regression]

    B --> F[Filtering / Denoising]
    B --> G[Edge Detection]
    B --> H[Segmentation]
```

### Interpretation

* **scikit-image** handles the raw image operations
* **scikit-learn** handles the ML part
* Together, they form a complete classical CV pipeline

---

## 3. Why scikit-image?

`scikit-image` is ideal for:

* Lightweight computer vision tasks
* Experiments where deep learning is unnecessary
* Feature extraction for traditional ML methods
* Educational, explainable, and interpretable workflows

It does NOT require GPU, making it suitable for beginner-friendly setups.

---

## 4. Installation

```
pip install scikit-image
pip install scikit-learn
```

Optional (for visualizations):

```
pip install matplotlib
```

---

## 5. Minimal Working Example: Image Classification Using HOG Features

You can save this as:

```
03_image_features_classification.ipynb
```

This example:

* Loads sample images
* Converts them to grayscale
* Extracts HOG features
* Trains an SVM classifier

---

### **Complete Example (Fully Runnable)**

```python
# ============================================================
# Image Classification using scikit-image + scikit-learn
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, feature
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------
# 1. Load Sample Images
# ------------------------------------------------------------
# Replace these with your own image folder if needed
image_paths = [
    "images/cat1.jpg",
    "images/cat2.jpg",
    "images/dog1.jpg",
    "images/dog2.jpg"
]

labels = [0, 0, 1, 1]  # 0 = cat, 1 = dog

images = [io.imread(p) for p in image_paths]

# ------------------------------------------------------------
# 2. Preprocess Images (resize + grayscale)
# ------------------------------------------------------------
processed = []
for img in images:
    img_resized = transform.resize(img, (128, 128), anti_aliasing=True)
    img_gray = color.rgb2gray(img_resized)
    processed.append(img_gray)

processed = np.array(processed)

# ------------------------------------------------------------
# 3. Extract HOG Features
# ------------------------------------------------------------
hog_features = []
for img in processed:
    hog = feature.hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    hog_features.append(hog)

hog_features = np.array(hog_features)

print("HOG feature shape:", hog_features.shape)

# ------------------------------------------------------------
# 4. Train/Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels, test_size=0.5, random_state=42
)

# ------------------------------------------------------------
# 5. Train a Classical ML Model (SVM)
# ------------------------------------------------------------
clf = LinearSVC()
clf.fit(X_train, y_train)

# ------------------------------------------------------------
# 6. Predict & Evaluate
# ------------------------------------------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Test Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 7. Visualize One Example
# ------------------------------------------------------------
plt.imshow(processed[0], cmap="gray")
plt.title("Example Preprocessed Image (Grayscale)")
plt.axis("off")
plt.show()
```

---

## 6. Explanation of the Image Pipeline

### **Step 1 — Image Loading**

`io.imread()` reads JPG/PNG/TIFF files into NumPy arrays.

### **Step 2 — Normalize and Resize**

Images must have a consistent input size.
128×128 grayscale is a common baseline.

### **Step 3 — Feature Extraction**

We use **HOG (Histogram of Oriented Gradients)**:

* captures edge structures
* widely used before deep learning
* excellent for tasks like person detection

Other feature options:

| Method           | From         | Use Case                   |
| ---------------- | ------------ | -------------------------- |
| HOG              | scikit-image | edges & shapes             |
| LBP              | scikit-image | texture patterns           |
| Raw pixels       | Numpy        | simple tasks only          |
| Color histograms | Numpy        | color-based classification |

### **Step 4 — ML Model**

We use SVM (Support Vector Machine):

* fast
* robust
* good for high-dimensional features like HOG

### **Step 5 — Evaluation**

`accuracy_score` + `classification_report`
Even with a tiny dataset, the pipeline works.

---

## 7. Extensions: Other Useful scikit-image Tools

### ✔ Edge Detection

```python
from skimage.filters import sobel
edges = sobel(img_gray)
```

### ✔ Canny Detector

```python
from skimage.feature import canny
edges = canny(img_gray, sigma=2)
```

### ✔ Gaussian Filtering

```python
from skimage.filters import gaussian
blurred = gaussian(img_gray, sigma=1)
```

### ✔ Image Segmentation

```python
from skimage.segmentation import slic
segments = slic(img_resized, n_segments=100)
```

---

## 8. Exercises (Optional)

### **Exercise 1 — Replace HOG with LBP**

Hint:

```python
from skimage.feature import local_binary_pattern
```

### **Exercise 2 — Build a 3-class classifier**

Add images from a third class (e.g., car/bird/etc.).

### **Exercise 3 — Visualize HOG Features**

Use `feature.hog(..., visualize=True)`.

### **Exercise 4 — Test a different ML model**

Try:

```python
from sklearn.ensemble import RandomForestClassifier
```

### **Exercise 5 — Implement a preprocessing function**

Create a reusable function for:

* resizing
* gray-scaling
* HOG extraction

---

## 9. When to Use Traditional CV vs Deep Learning?

| Scenario                          | Traditional CV (HOG/LBP) | Deep Learning (CNN) |
| --------------------------------- | ------------------------ | ------------------- |
| Very small dataset                | ✔                        | ✖ (overfits)        |
| Need fast training                | ✔                        | ✖                   |
| No GPU available                  | ✔                        | ✖                   |
| Complex patterns (faces, objects) | ✖                        | ✔                   |
| Research on handcrafted features  | ✔                        | —                   |

This chapter helps you understand classical pipelines, which are still widely used in lightweight vision tasks.

---

## 10. Next Chapter

If you want to continue:

### 👉 **Chapter 4 — Signal and Time-Series Processing using scikit-signal + scikit-learn**

In Chapter 4, you will learn:

* Basic filtering (low-pass, high-pass)
* Spectral analysis
* Time-series feature extraction
* Anomaly detection with scikit-learn

---
