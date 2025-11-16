# 📘 Chapter 9 — Conformal Prediction for Classification

*Set-valued predictions, uncertainty, and reliable classification under distribution shift.*

---

## 1. Chapter Goals

By the end of this chapter, you will understand:

* The difference between **regression CP** and **classification CP**
* Set-valued outputs (possibly multiple labels at test time)
* **Confidence** and **credibility** measures
* How to apply **split CP** to any classifier
* How to handle **multiclass and imbalanced datasets**
* How to evaluate classification CP models
* How CP behaves under distribution shift

Everything is fully runnable and self-contained.

---

# 2. Why Classification Needs Conformal Prediction

A normal classifier outputs:

```
class = argmax(p_i)
```

But real-world data often has:

* overlapping classes
* ambiguous samples
* low-quality or noisy observations
* out-of-distribution samples
* adversarial cases (e.g., deepfake detection)

Classification CP addresses this by producing:

### ✔ A *set of plausible labels*, not just one

### ✔ Guaranteed error control

### ✔ A formal measure of uncertainty

A CP classifier might output:

```
{cat}
{dog, wolf}
{unknown}   ← empty set = highly uncertain
```

This is **safer** than forcing a single label.

---

# 3. Intuition Behind Classification CP

Given a classifier providing probability estimates:

[
\hat{p}_1(x), \hat{p}_2(x), ..., \hat{p}_K(x)
]

CP builds a threshold ( \tau ) such that:

[
C(x) = {, k : \hat{p}_k(x) \ge \tau ,}
]

where ( C(x) ) is the *prediction set*.

The threshold ( \tau ) is chosen using calibration data to guarantee:

[
\Pr( y \in C(x) ) \ge 1 - \alpha
]

e.g., 95% confidence.

---

# 4. Minimal Working Example — Split Conformal Prediction for Classification

This is a complete runnable example.

---

## 4.1 Prepare the dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# load data
X, y = load_iris(return_X_y=True)

# split: train → calibration → test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

---

## 4.2 Train model and obtain calibration probabilities

```python
clf = RandomForestClassifier().fit(X_train, y_train)

cal_probs = clf.predict_proba(X_cal)
```

---

## 4.3 Compute nonconformity scores

A simple nonconformity score:

[
\text{nc} = 1 - \hat{p}_{y_i}(x_i)
]

```python
nc_scores = 1 - np.max(cal_probs * (np.eye(3)[y_cal]), axis=1)
```

---

## 4.4 Compute quantile threshold

```python
alpha = 0.05
tau = np.quantile(nc_scores, 1 - alpha)
tau
```

---

## 4.5 Build prediction sets

For each test point:

```python
test_probs = clf.predict_proba(X_test)
prediction_sets = []

for probs in test_probs:
    S = np.where(1 - probs <= tau)[0]  # include classes passing threshold
    prediction_sets.append(set(S))
```

---

# 5. Evaluating Coverage

```python
contain = [y_test[i] in prediction_sets[i] for i in range(len(y_test))]
coverage = np.mean(contain)
coverage
```

Expected ≈ 0.95.

---

# 6. Visualizing Prediction Set Sizes

```python
import matplotlib.pyplot as plt

sizes = [len(S) for S in prediction_sets]
plt.hist(sizes, bins=[1,2,3,4], align="left", rwidth=0.6)
plt.xlabel("Set Size")
plt.ylabel("Count")
plt.title("Distribution of Prediction Set Sizes")
plt.show()
```

Interpretation:

* size = 1 → confident
* size = 2 → moderate uncertainty
* size = 3 → high uncertainty

---

# 7. Confidence and Credibility (Important!)

For test sample *x*, sort probabilities:

[
p_{(1)} \ge p_{(2)} \ge ... \ge p_{(K)}
]

### ► **Confidence**

How much probability we must remove from the top class to include the true label.

Simple implementation:

```python
def confidence(probs):
    sorted_probs = np.sort(probs)[::-1]
    return 1 - sorted_probs[0]
```

Lower confidence → more uncertain.

---

### ► **Credibility**

How compatible the prediction is with the model:

```python
def credibility(probs):
    return np.max(probs)
```

Lower credibility → likely OOD or unknown.

---

# 8. Handling Multiclass & Imbalanced Datasets

### ✔ Use stratified calibration split

### ✔ Consider classwise nonconformity

### ✔ Asymmetric thresholds for imbalanced classes

Example: classwise CP

```python
class_nc = {}
for c in np.unique(y_cal):
    mask = y_cal == c
    class_nc[c] = np.quantile(1 - cal_probs[mask][:, c], 1 - alpha)
```

Prediction sets become class-dependent.

---

# 9. Reject Option (optional extension)

Sometimes prediction sets become empty:

```
{}
```

This means the classifier “refuses to guess”.

Useful in:

* safety-critical systems
* deepfake detectors
* face recognition
* autonomous driving
* medical diagnosis

To enable reject:

```python
reject = (len(S) == 0)
```

---

# 10. CP for Deepfake Detection (Your Research Case)

Classification CP is extremely relevant for:

* ambiguous frames
* noisy or low-light fake videos
* diffusion-generated content
* adversarial cases
* motion-inconsistent frames

A CP detector:

* high confidence → set = {real}
* medium confidence → set = {real, fake}
* low confidence → set = {}

This is much safer than forced binary classification.

You can also compute:

* coverage per video
* interval size per frame
* uncertainty distribution across patches
* credibility for OOD frames

Integrate CP at the end of your model:

```python
logits = model(frame)
probs = softmax(logits)
```

Then apply classification CP.

---

# 11. Full Reusable Template

```python
def conformal_classification(clf, X_train, y_train, 
                             X_cal, y_cal, X_test, alpha=0.05):
    
    clf.fit(X_train, y_train)
    
    cal_probs = clf.predict_proba(X_cal)
    nc_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    tau = np.quantile(nc_scores, 1 - alpha)
    
    test_probs = clf.predict_proba(X_test)
    pred_sets = [set(np.where(1 - p <= tau)[0]) for p in test_probs]
    
    return pred_sets, tau
```

Usage:

```python
prediction_sets, tau = conformal_classification(
    clf, X_train, y_train, 
    X_cal, y_cal, X_test, alpha=0.05
)
```

---

# 12. Exercises (Optional)

### Exercise 1

Apply classification CP to CIFAR-10 on a small ML model.

### Exercise 2

Implement classwise CP and compare set sizes.

### Exercise 3

Plot confidence vs credibility across misclassified samples.

### Exercise 4

Train multiple classifiers and compare prediction-set widths.

### Exercise 5

Apply CP to deepfake detection and visualize:

* prediction set size per frame
* uncertainty across video time

---
