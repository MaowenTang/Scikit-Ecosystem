# 📘 Chapter 10 — Conformal Prediction under Distribution Shift & Domain Adaptation

*How to maintain statistical guarantees when data changes across domains.*

---

## 1. Chapter Goals

You will learn:

* Why standard CP fails under distribution shift
* How to detect shift and measure calibration drift
* How to correct prediction intervals using:

  * covariate shift correction
  * weighted conformal prediction
  * Mondrian CP
  * adaptive CP
  * domain adaptation + CP combinations
* How to apply CP reliably across:

  * soil spectroscopy domain shift
  * deepfake detection dataset shift
  * energy system drift
* How to design CP pipelines for real production systems

---

# 2. What Is Distribution Shift?

A shift happens when:

### **Training Distribution**

[
(X_{\text{train}}, Y_{\text{train}})
]

is not equal to

### **Test Distribution**

[
(X_{\text{test}}, Y_{\text{test}})
]

Common types:

| Type                    | Meaning                                                  |            |
| ----------------------- | -------------------------------------------------------- | ---------- |
| **Covariate shift**     | Input X changes (e.g., spectra from new soil region)     |            |
| **Label shift**         | Class proportions change (e.g., fake videos more common) |            |
| **Conditional shift**   | Relationship P(Y                                         | X) changes |
| **Subpopulation shift** | Only a subgroup changes (e.g., high SOC)                 |            |
| **Temporal drift**      | Time-varying shift (e.g., HVAC systems, sensors)         |            |

Under shift, **residuals from calibration no longer represent test samples**, breaking CP guarantees.

---

# 3. Why Standard Conformal Prediction Fails Under Shift

CP assumes:

[
(X_{\text{cal}}, Y_{\text{cal}}) \sim (X_{\text{test}}, Y_{\text{test}})
]

Under shift:

* calibration residuals are **too optimistic**
* q-quantile is **too small**
* coverage drops far below 95%
* intervals become **misleadingly narrow**

This is exactly what happens when:

* You test your soil CNN on **Texas soils** after training on **LUCAS**
* You test deepfake detectors on **CelebDF** after training on **FF++**
* You test HVAC anomaly detectors on different seasons

---

# 4. Detecting Distribution Shift Before Using CP

Before computing intervals, compute statistical tests on:

### ✔ Feature distribution

e.g., KS-test, Kolmogorov–Smirnov distance

### ✔ Embedding shift

Useful in deepfake detection:

[
| f(X_{\text{test}}) - f(X_{\text{cal}}) |
]

### ✔ Shift metrics:

* Maximum Mean Discrepancy (MMD)
* Fréchet Feature Distance (FFD)
* Bhattacharyya distance
* KL divergence

### ✔ Drift in residuals

After initial CP application:

```
CP coverage drops from 95% → 70%
```

= strong distribution shift.

---

# 5. Weighted Conformal Prediction (WCP)

Fix covariate shift using **importance weights**:

[
w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}
]

Compute q such that:

[
F_{\text{cal}}(q) = 1 - \alpha
]

but using **weighted quantile**.

---

## 5.1 Minimal Example: Weighted Quantile

```python
def weighted_quantile(values, weights, q):
    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cumw = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cumw, q)]
```

Use a density-ratio model:

```python
from sklearn.linear_model import LogisticRegression

# combine train and test as domain labels
X_domain = np.vstack([X_cal, X_test])
y_domain = np.hstack([np.zeros(len(X_cal)), np.ones(len(X_test))])

w_model = LogisticRegression().fit(X_domain, y_domain)
p_test = w_model.predict_proba(X_cal)[:, 1]
p_train = 1 - p_test
weights = p_test / p_train
```

Then compute:

```python
q = weighted_quantile(residuals, weights, 1 - alpha)
```

---

# 6. Mondrian Conformal Prediction (Subpopulation CP)

When shift affects only subgroups (e.g., high SOC):

Compute **per-category quantiles**.

Example: soil carbon bins:

* Low 0–1.5%
* Medium 1.5–4%
* High >4%

Compute:

```python
q_low = quantile(residuals_low, 0.95)
q_med = quantile(residuals_med, 0.95)
q_high = quantile(residuals_high, 0.95)
```

At test time, detect which bin the sample belongs to → apply correct interval.

This method handles **heteroscedasticity** and **population drift**.

---

# 7. Adaptive Conformal Prediction (ACP)

Creates **locally adaptive** prediction intervals.

Instead of one global q, use a **local nonconformity score**:

[
r_i = \frac{|y_i - \hat{y}_i|}{\sigma(x_i)}
]

where ( \sigma(x_i) ) is local model uncertainty.
You can obtain it using:

* kNN variance
* dropout variance (deep models)
* ensemble variance
* residual regression network

This produces **narrow intervals in easy regions**
and **wider intervals in difficult regions**.

---

# 8. Conformal Prediction + Domain Adaptation

### Powerful combination under real-world shift:

1. **Align domains**

   * CORAL
   * MMD minimization
   * Domain adversarial training (DANN)
   * Feature standardization per domain

2. **Apply CP after alignment**

3. **Use weighted or adaptive CP** to correct post-alignment drift

This is extremely effective in:

### Soil spectroscopy

LUCAS → Texas soils
Dry → wet spectra differences
Different devices → distribution drift

### Deepfake detection

FF++ → CelebDF → DFDC → Diffusion-based fakes
Different camera pipelines
Different compression profiles

### Energy analytics

Winter → summer
Daytime → nighttime
Different building seasons

---

# 9. Practical CP Strategies Under Distribution Shift

Below are **battle-tested strategies** used in high-impact research.

---

## ✔ Strategy 1 — Local CP via kNN Embedding Distance

Compute embedded features (CLIP, DINOv2, CNN latent space):

```
dist = || f(x_test) - kNN(f(X_cal)) ||
```

If distance is high → go to **wider CP interval**.

---

## ✔ Strategy 2 — Recalibrate Using Few Target Samples (Semi-supervised)

Collect small labeled set in target domain:

```
train → original cal → target cal → test
```

Use target calibration residuals to update q.

This cuts error drastically.

---

## ✔ Strategy 3 — Reject Option for Extreme Shift

If prediction set is **empty** → model doesn’t trust the sample.

Used in:

* deepfake detectors for out-of-domain attacks
* soil estimation for rare soil types
* HVAC detection of unseen failure modes

---

# 10. Deepfake Detection Case Study (Your research)

### Problem:

Model trained on FF++ but must detect diffusion-generated deepfakes or CelebDF.

### Fix with CP:

* Use DINOv2/CLIP embeddings to measure distribution shift
* Weight calibration set according to embedding density
* Use Mondrian CP based on motion-level difficulty
* Add reject option for extreme uncertainty

This gives:

* Higher coverage
* Better reliability under novel fake types
* Robustness to video compression and motion drift

---

# 11. Soil Spectroscopy Case Study (Your SOC work)

### Problem:

CNN trained on LUCAS → applied to Texas soils or wet spectra.

### Fix with CP:

* Weighted CP using spectral distance weighting
* Local CP using Savitzky–Golay smoothed spectral norms
* Mondrian CP per carbon bin (low/medium/high)
* Adaptive CP using wavelength-level prediction variance

This stabilizes SOC intervals and fixes underestimation bias in high-carbon regions.

---

# 12. Full Implementation Template — Weighted + Local CP

```python
def domain_aware_cp(model, X_train, y_train, 
                    X_cal, y_cal, X_test, alpha=0.05):
    
    model.fit(X_train, y_train)
    cal_pred = model.predict(X_cal)

    # residuals
    residuals = np.abs(y_cal - cal_pred)

    # local density weights (domain classifier)
    from sklearn.linear_model import LogisticRegression
    X_domain = np.vstack([X_cal, X_test])
    y_domain = np.hstack([np.zeros(len(X_cal)), np.ones(len(X_test))])
    w_model = LogisticRegression().fit(X_domain, y_domain)
    p_test = w_model.predict_proba(X_cal)[:, 1]
    p_train = 1 - p_test
    weights = p_test / p_train

    # weighted quantile
    def weighted_quantile(v, w, q):
        idx = np.argsort(v)
        v, w = v[idx], w[idx]
        cumw = np.cumsum(w) / np.sum(w)
        return v[np.searchsorted(cumw, q)]
    
    q = weighted_quantile(residuals, weights, 1 - alpha)

    test_pred = model.predict(X_test)
    return test_pred - q, test_pred + q, q
```

---

# 13. Evaluation Under Shift

### ✔ 1. Coverage drop

Measure coverage in the target domain.

### ✔ 2. Interval width

How much wider CP becomes under shift.

### ✔ 3. Calibration drift

Plot calibration residuals vs test residuals.

### ✔ 4. Embedding shift

Measure distance in latent space.

### ✔ 5. Set-size increase (classification CP)

If prediction set sizes grow → drift.

---

# 14. Exercises (Optional)

### Exercise 1

Implement Mondrian CP for your SOC bins.

### Exercise 2

Compute embedding shift using DINOv2 for deepfake frames.

### Exercise 3

Apply weighted CP to your soil CNN using Texas soils as target domain.

### Exercise 4

Implement reject option for high embedding-distance samples.

### Exercise 5

Evaluate CP under FF++ → CelebDF shift.

---
