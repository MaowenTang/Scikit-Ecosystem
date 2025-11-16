# 📘 Chapter 7 — Advanced scikit-learn: Pipelines, Transformers, Explainability, and Model Deployment

*How to build robust, maintainable, production-ready ML systems with scikit-learn.*

---

## 1. Chapter Goals

After completing this chapter, you will master:

* **Pipelines** — chaining preprocessing + models cleanly
* **ColumnTransformer** for mixed numeric/categorical data
* **Custom transformers** to insert your own preprocessing logic
* **Feature engineering inside Pipelines**
* **Cross-validation strategies**
* **Model explainability techniques** (permutation importance, SHAP-ready structure)
* **Saving, loading, and deploying scikit-learn models**

This chapter is **self-contained**, and each section includes runnable examples and practical guidance.

---

## 2. Why Advanced scikit-learn Matters

Beginners often write ML code like this:

```python
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)
pred = model.predict(X_scaled)
```

But professionals write **modular, reusable pipelines**:

```python
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
```

Advanced scikit-learn turns your ML into:

* clean
* reproducible
* shareable
* deployable

This is how industry-grade ML is built.

---

# 3. Pipelines — The Most Important Tool in scikit-learn

A **Pipeline** chains multiple steps into one object:

```mermaid
flowchart LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model]
    D --> E[Prediction]
```

---

## 3.1 Minimal Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf", SVC(kernel="rbf"))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
```

### Benefits:

* Consistency (preprocessing always applied correctly)
* Cleaner code
* Ready for cross-validation
* Works well with GridSearchCV and BayesSearchCV

---

# 4. ColumnTransformer — Handling Mixed Data

Real datasets often contain:

* numeric columns
* categorical columns
* binary flags
* text features

`ColumnTransformer` lets you preprocess each type differently.

---

## 4.1 Example: Numeric + Categorical

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

numeric_cols = ["age", "income"]
categorical_cols = ["country"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(), categorical_cols)
])

pipeline = Pipeline([
    ("pre", preprocess),
    ("clf", LogisticRegression())
])

pipeline.fit(df[numeric_cols + categorical_cols], df["label"])
```

---

# 5. Custom Transformers — Insert Your Own Logic

Useful when:

* you need domain-specific feature engineering
* scikit-learn doesn’t provide what you need
* you want to keep everything inside a `Pipeline`

---

## 5.1 Minimal Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class PeakToPeak(BaseEstimator, TransformerMixin):
    """Compute peak-to-peak amplitude for each sample."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return (np.max(X, axis=1) - np.min(X, axis=1)).reshape(-1, 1)
```

Add it to Pipeline:

```python
pipeline = Pipeline([
    ("ptp", PeakToPeak()),
    ("clf", RandomForestClassifier())
])
```

---

# 6. FeatureUnion — Combining Multiple Feature Extractors

Useful for:

* combining multiple representations
* image + numeric
* FFT + time-domain
* domain-specific + automatic features

---

## 6.1 Example

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA

combined = FeatureUnion([
    ("pca", PCA(n_components=10)),
    ("scale", StandardScaler())
])
```

---

# 7. Advanced Cross-Validation Strategies

### ✔ Standard k-fold

```
KFold(n_splits=5, shuffle=True)
```

### ✔ Stratified k-fold (classification)

```
StratifiedKFold(n_splits=5)
```

### ✔ TimeSeriesSplit (temporal data)

```
TimeSeriesSplit(n_splits=5)
```

### ✔ GroupKFold (grouped data, e.g., subjects in medical trials)

---

## 7.1 Example: TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    print(train_idx, test_idx)
```

---

# 8. Explainability — Understanding Model Behavior

scikit-learn integrates well with:

* **Permutation Importance**
* **Feature Importances**
* **Partial Dependence Plots (PDP)**
* **SHAP** (external library)

---

## 8.1 Permutation Importance

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test)
print(result.importances_mean)
```

---

## 8.2 Feature Importance (Tree Models)

```python
model.feature_importances_
```

---

## 8.3 Partial Dependence

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X, [0, 1])
```

---

# 9. Saving & Deploying Models

### ✔ Save model

```python
import joblib
joblib.dump(pipeline, "model.pkl")
```

### ✔ Load model

```python
model = joblib.load("model.pkl")
model.predict(X_test)
```

### ✔ Convert to web API

Use **FastAPI**:

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    X = [list(features.values())]
    return {"prediction": int(model.predict(X)[0])}
```

---

# 10. Professional Patterns for Real Systems

These patterns reflect real industry workflows:

---

## ✔ Pattern: Production ML Pipeline

* ColumnTransformer
* Pipeline
* Hyperparameter optimization
* Cross-validation
* Explainability
* Model persistence

---

## ✔ Pattern: Research Pipeline

* Custom transformers
* FeatureUnion
* PCA / dimensionality reduction
* Experiment tracking
* Conformal prediction (optional extension chapter)

---

## ✔ Pattern: Hybrid ML + Deep Learning Pipeline

* scikit-image preprocessing
* Classical feature extraction (HOG/LBP)
* CNN embeddings
* scikit-learn classifier on top

This pattern is extremely common in:

* remote sensing
* soil spectroscopy
* medical imaging
* forensics
* some deepfake detection baselines

---

# 11. Exercises (Optional)

### **Exercise 1 — Build a mixed-type Pipeline**

Use numeric + categorical + custom features.

### **Exercise 2 — Add Bayesian Optimization**

Use BayesSearchCV + Pipeline.

### **Exercise 3 — Build your own Transformer**

Feature engineering from your research domain.

### **Exercise 4 — Interpret a model**

Plot permutation importance and PDP.

### **Exercise 5 — Deploy a model**

Use FastAPI or Flask to serve a scikit-learn Pipeline.

---

# 12. Next Steps (Optional “Chapter 8”)

If you want a continuation, possible directions are:

### **📘 Chapter 8 — Conformal Prediction for Reliable ML**

* Prediction intervals
* Uncertainty quantification
* Integration with scikit-learn models
* Real-world reliability metrics


