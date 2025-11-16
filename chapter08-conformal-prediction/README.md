# 📘 Chapter 8 — Conformal Prediction (CP) for Reliable Machine Learning

*How to quantify model uncertainty, build prediction intervals, and guarantee statistical coverage.*

---

## 1. Chapter Goals

After this chapter, you will understand:

* The core idea of Conformal Prediction
* Split Conformal and Cross-Conformal Prediction
* How to compute residuals and calibration quantiles
* How to build prediction intervals for regression
* How to use CP uncertainty for model comparison
* How to apply CP to your own research pipelines (e.g., SOC prediction)

This chapter is **self-contained**, with runnable code.

---

# 2. Why Conformal Prediction?

Most ML models output *point predictions*:

```
ŷ = model(x)
```

But real problems require **confidence**.

Conformal Prediction provides:

### ✔ A valid prediction interval

### ✔ With statistical guarantees

### ✔ Under *minimal assumptions*

### ✔ For *any* model (linear, tree, neural network, CNN, etc.)

Unlike Bayesian methods or ensembles, CP:

* **does not require retraining**
* **works on top of any model**
* provides **coverage guarantees** (e.g., 95% interval is truly 95% reliable)

---

# 3. How Split Conformal Prediction Works

Split CP uses 3 datasets:

```
Train → Calibrate → Test
```

1. Train model on **training set**
2. Predict on **calibration set**
3. Compute residuals on calibration set
4. Compute the quantile ( q_\alpha ) of residuals
5. Build interval for test samples:

[
\hat{y} \pm q_\alpha
]

---

# 4. Minimal Example (Fully Runnable)

Trains a regressor → builds CP intervals → evaluates coverage.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Generate data
X, y = make_regression(n_samples=1500, noise=15, random_state=42)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 2. Fit model
model = RandomForestRegressor().fit(X_train, y_train)

# 3. Calibration residuals
y_cal_pred = model.predict(X_cal)
residuals = np.abs(y_cal - y_cal_pred)

# 4. Quantile
alpha = 0.05
q = np.quantile(residuals, 1 - alpha)

# 5. Prediction intervals
y_test_pred = model.predict(X_test)
lower = y_test_pred - q
upper = y_test_pred + q

coverage = np.mean((y_test >= lower) & (y_test <= upper))
print("95% coverage =", coverage)
print("q =", q)
```

### Output Example

```
95% coverage = 0.948
q = 31.77
```

---

# 5. Visualizing Prediction Intervals

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, s=18, label="Predicted")
plt.errorbar(y_test, y_test_pred,
             yerr=q, fmt='o', ecolor='red', alpha=0.2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Conformal Prediction Intervals")
plt.show()
```

Each point gets a **vertical** error bar:

[
[\hat{y} - q,; \hat{y} + q]
]

---

# 6. Why CP Works (Intuition)

CP assumes:

* The calibration data and test data come from the same distribution
* The calibration errors represent future errors

Then:

[
q = \text{the worst-case reasonable error}
]

So we safely bracket the unknown prediction.

---

# 7. Advanced Topic: Asymmetric Intervals

Symmetric intervals use:

```
abs(y - y_pred)
```

But some problems require **asymmetric errors**:

* underestimation more dangerous than overestimation
* skewed distributions (soil carbon is skewed)

Use two-sided quantiles:

```python
lower_q = np.quantile(y_cal - y_cal_pred, alpha/2)
upper_q = np.quantile(y_cal - y_cal_pred, 1 - alpha/2)

lower = y_test_pred + lower_q
upper = y_test_pred + upper_q
```

---

# 8. Integrating CP with Pipelines

CP works with both:

* scikit-learn Pipelines
* deep learning models (PyTorch / TensorFlow)
* your soil CNN + MLP architecture

Example using a Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", GradientBoostingRegressor())
])

pipeline.fit(X_train, y_train)
cal_pred = pipeline.predict(X_cal)
residuals = np.abs(y_cal - cal_pred)
```

---

# 9. Evaluating CP Performance

### 1. **Coverage (% inside interval)**

Should match the target (e.g., 95%).

### 2. **Median interval width**

Narrower = better reliability.

### 3. **Coverage per subgroup**

Useful for:

* low/medium/high SOC
* different soil moisture levels
* different spectral devices
* regional distributions
* different carbon content clusters

### 4. **Distribution of residuals**

Should be roughly symmetric around 0.

---

# 10. CP Applied to Soil Spectroscopy (Your Case)

Your soil work fits CP perfectly.

### Why CP is ideal:

* Spectral reflectance → nonlinear → unpredictable
* High noise in low-carbon regions
* Sensor variation
* Field moisture shift
* Soil heterogeneity

### How CP helps you:

* Provides uncertainty for each SOC prediction
* Shows low-SOC underestimation bias
* Helps compare models (CNN vs PLSR vs Cubist)
* Supports decision-making (confidence-aware SOC mapping)

### Your actual pipeline:

```
Wet spectra → MLP → equivalent dry spectra → CNN → SOC
                                     ↓
                         Conformal Prediction
```

Add CP at the **end**:

```python
cnn_pred = cnn_model.predict(X_cal)
residuals = np.abs(y_cal - cnn_pred)
q = np.quantile(residuals, 0.95)
```

---

# 11. Cross-Conformal Prediction (CCP)

Instead of one split:

* Run K-fold
* Get multiple q quantiles
* Combine them

More stable but more expensive.

---

# 12. Normalized Conformal Prediction

Divide residual by difficulty indicator:

[
r_i = \frac{|y_i - \hat{y}_i|}{\hat{\sigma}_i}
]

Useful when:

* variance differs across soil types
* noise increases at high SOC
* model highly heteroscedastic

---

# 13. Best Practices

✔ Use a calibration set **not used in training**
✔ Store calibration residual distribution
✔ Recalibrate when changing dataset or model
✔ Monitor coverage drift
✔ Use asymmetric intervals for skewed targets
✔ Compare intervals across models
✔ Visualize interval lengths across SOC ranges

---

# 14. Full Professional Pipeline (Reusable Template)

```python
def conformal_interval(model, X_train, X_cal, X_test, y_train, y_cal, alpha=0.05):
    model.fit(X_train, y_train)
    
    cal_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - cal_pred)
    q = np.quantile(residuals, 1 - alpha)
    
    test_pred = model.predict(X_test)
    lower = test_pred - q
    upper = test_pred + q
    return lower, upper, q
```

Plug in **any** model:

```python
lower, upper, q = conformal_interval(RandomForestRegressor(),
                                     X_train, X_cal, X_test,
                                     y_train, y_cal)
```

---

# 15. Exercises (Optional)

### Exercise 1

Implement asymmetric conformal prediction.

### Exercise 2

Plot coverage vs SOC (low, medium, high).

### Exercise 3

Integrate CP with your soil CNN model.

### Exercise 4

Compute normalized CP and compare interval widths.

### Exercise 5

Reproduce your paper figure with CP intervals.

---

# 16. Next Chapter (Optional)

### **📘 Chapter 9 — Conformal Prediction for Classification**

* Set-valued predictions
* Credibility + confidence
* Classification under distribution shift
