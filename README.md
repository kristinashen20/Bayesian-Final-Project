# 🫀 Heart‑Disease 30‑Day Readmission — Bayesian & ML Study  

Predicting early readmission for cardiac inpatients using an interpretable **Bayesian Network** and baseline machine‑learning models  
*(UCI “Diabetes 130‑US Hospitals 1999‑2008” dataset, filtered to ICD‑9 circulatory diagnoses)*  

---

## 1  Project Abstract
Early 30‑day readmission of heart‑disease patients drives cost and signals gaps in continuity of care.  
We build an interpretable **Bayesian Network** (BN) to capture probabilistic dependencies between demographics, utilisation variables, co‑morbidities and the binary `<30 d` readmission outcome, using the UCI *Diabetes 130‑US Hospitals* dataset (101 766 encounters).  
The workflow:

1. **Clinical filter** → retain only encounters with an ICD‑9 circulatory‑system diagnosis (390‑459, 785) → 59 313 rows.  
2. **Cleaning & EDA** → drop high‑missing columns, impute residual NaNs, visualise numeric and categorical distributions.  
3. **Models**:  
   * Bayesian Network (pgmpy) – hill‑climb + BIC structure, Bayesian parameter estimation.  
   * Logistic Regression – baseline calibration benchmark.  
   * XGBoost – strong non‑linear baseline.  
4. **Evaluation** – AUC, precision‑recall, Brier score & calibration curves.  

---

## 2  Dataset

| Item | Value |
|------|-------|
| Source | <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008> |
| Raw size | 101 766 inpatient encounters, 50 columns |
| Heart‑disease subset | 59 313 rows after ICD‑9 390‑459 / 785 filter |
| Target | `readmitted = <30` (binary) |

---

## 3  Data‑Cleaning Pipeline

| Step | Action |
|------|--------|
| **Drop high‑missing columns** | `weight` (~97 %), `medical_specialty` (~49 %), `payer_code` (~40 %) |
| **Replace “?” → NaN** | normalise missing markers |
| **Impute** | diagnosis NaNs → `UnknownDiag`; other categoricals → mode |
| **Binary target** | `readmitted_binary = 1` if `<30`, else 0 |
| **Cardiac filter** | keep rows where any of `diag_1/2/3` ∈ 390‑459 or 785 |
| **Save** | `diabetic_heart_cleaned.csv` (~9 MB) |

---

## 4  Exploratory Data Analysis (highlights)

### Numeric snapshots  

| Feature | Mean ± SD | 75 %‑tile | Notes |
|---------|-----------|-----------|-------|
| `time_in_hospital` | 4.3 ± 3.0 days | 6 d | short LOS; long‑tail to 14 d |
| `num_lab_procedures` | 42.9 ± 19.5 | 56 | quasi‑normal; heavy utilisation |
| `num_medications` | 16.6 ± 8.5 | 20 | right‑skew; outliers > 60 |

### Categorical distributions  
* **Race**: 75 % Caucasian, 19 % African‑American — heavy imbalance.  
* **Age**: skewed older; `[70‑80)` largest bucket.  
* **Readmission flag**: `<30 d` = 37 % of encounters (positive class).  

### Readmission rate drivers  
| Factor | Higher risk | Lower risk |
|--------|-------------|------------|
| Age | `[20‑30)` 18 % | `[80‑90)` 10 % |
| Medication dosage change | *Yes* 12 % | *No* 11 % |
| On diabetes meds | *Yes* 12 % | *No* 10 % |

### Correlation matrix (numeric)  
No pair |ρ| > 0.5 ⇒ multicollinearity is low; all numeric features retained.

---

## 5  Modelling Road‑map

| Model | Library | Purpose |
|-------|---------|---------|
| **Bayesian Network** | `pgmpy` | Interpret joint dependencies & posterior probabilities |
| Logistic Regression | `scikit‑learn` | Linear baseline & calibration reference |
| XGBoost | `xgboost` | Non‑linear performance ceiling |

---
