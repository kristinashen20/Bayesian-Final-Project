# 🫀 Heart‑Disease 30‑Day Readmission — Bayesian & ML Study  

Predicting early readmission for cardiac inpatients using an interpretable **Bayesian Network** and baseline machine‑learning models  
*(UCI “Diabetes 130‑US Hospitals 1999‑2008” dataset, filtered to ICD‑9 circulatory diagnoses)*  

---

## 1. Project Abstract
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

## 2. Dataset

| Item | Value |
|------|-------|
| Source | <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008> |
| Raw size | 101 766 inpatient encounters, 50 columns |
| Heart‑disease subset | 59 313 rows after ICD‑9 390‑459 / 785 filter |
| Target | `readmitted = <30` (binary) |

---

## 3. Data‑Cleaning Pipeline

| Step | Action |
|------|--------|
| **Drop high‑missing columns** | `weight` (~97 %), `medical_specialty` (~49 %), `payer_code` (~40 %) |
| **Replace “?” → NaN** | normalise missing markers |
| **Impute** | diagnosis NaNs → `UnknownDiag`; other categoricals → mode |
| **Binary target** | `readmitted_binary = 1` if `<30`, else 0 |
| **Cardiac filter** | keep rows where any of `diag_1/2/3` ∈ 390‑459 or 785 |
| **Save** | `diabetic_heart_cleaned.csv` |

---

## 4. Key Exploratory Insights

### Length of Stay (LOS)
Most patients are discharged in under 6 days — right-skewed with a long tail to 14.

<img src="images/eda_time_in_hospital.png" width="480">

### Readmission Outcome Balance
About 37% of patients were readmitted within 30 days. The rest split into “NO” (52%) and “>30” (11%).

<img src="images/eda_readmitted_bar.png" width="480">

---

## Key Drivers of Readmission Risk

### Age
Younger patients (<50) have **higher** readmission rates — opposite of typical risk expectations.
Rates fall steadily with age (possible survivorship bias or different care pathways). Age buckets carry a clear monotonic trend—keep ordered encoding.

<img src="images/eda_readmit_rate_by_age.png" width="480">

### Medication Change
Patients whose medication dosages were adjusted are more likely to return within 30 days.

<img src="images/eda_readmit_rate_by_change.png" width="480">

---


## Correlation Matrix

The strongest correlation is `time_in_hospital` ↔ `num_medications` (ρ ≈ 0.45).  
No pair exceeds ρ = 0.7 → low multicollinearity, so we can retain all numeric features in logistic regression without severe variance inflation.

<img src="images/eda_correlation_table.png" width="720">

---


## 5  Modelling Road‑map

| Model | Library | Purpose |
|-------|---------|---------|
| **Bayesian Network** | `pgmpy` | Interpret joint dependencies & posterior probabilities |
| Logistic Regression | `scikit‑learn` | Linear baseline & calibration reference |
| XGBoost | `xgboost` | Non‑linear performance ceiling |

---

## Bayesian-Network Classifier with **pgmpy**

---

### 1&nbsp;&nbsp;Method overview
* **Dataset** UCI “Diabetes 130-US Hospitals 1999–2008” – 59 k encounters after cleaning  
* **Feature set** 14 high-signal columns (demographics, discharge context, prior utilisation, therapy flags)  
* **Numeric handling** `KBinsDiscretizer(n_bins = 3, strategy = "quantile")` → low / mid / high bins  
* **Structures compared**  
  1. **PC algorithm**   (constraint-based)  
  2. Hill-Climb + BIC   (max 3 parents)  
  3. Chow–Liu Tree      (root = `target`)  
* **Parameter fit** Bayesian Estimator (`prior_type="BDeu"`)  
* **Validation split** 80 / 20 train-test

---

### 2&nbsp;&nbsp;Result sheet 1 – model selection

| Candidate | Validation BIC (↓) |
|-----------|-------------------:|
| **PC**           | **− 72 294.85** |
| Hill-Climb       | − 70 883.71 |
| Tree             | − 70 984.39 |

**PC is chosen** because it has the most-negative (best) BIC, i.e. the best
balance of goodness-of-fit and model simplicity.

---

### 3&nbsp;&nbsp;Performance of the selected PC model

| Metric                    | PC Network | What it tells us                                                                                 |
| ------------------------- | ---------: | ------------------------------------------------------------------------------------------------ |
| **ROC-AUC (↑)**           | **0.595**  | ≈ 60 % chance the model ranks a readmitted patient above a non-readmitted one.                   |
| **Average Precision (↑)** | **0.143**  | Precision–recall summary for an imbalanced target (≈ 14 % baseline).                             |
| **Brier Score (↓)**       | **0.099**  | Calibration error; well below the 0.12–0.13 “coin-flip” region.                                  |

---

### 4&nbsp;&nbsp;Graph comparison

| PC (best) | Hill-Climb | Chow–Liu Tree |
|-----------|------------|---------------|
| ![PC](images/pgmpy_pc_structure.png) | ![HC](images/pgmpy_hc_structure.png) | ![Tree](images/pgmpy_tree_structure.png) |

|                 | Key edges you can point out |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **PC**          | `discharge_disposition_id → target` • `number_inpatient → target` • Therapy hub `diabetesMed → change → insulin`                     |
| **Hill-Climb**  | Denser; feedback chain age → insulin → change, but still funnels into `target` via discharge disposition.                            |
| **Tree**        | Strict Chow–Liu spanning tree; `insulin` drives `diabetesMed` and `change`, then links to `target` through `number_inpatient`.      |

**Recurring themes**

* Patients discharged **anywhere other than routine home** have far higher 30-day return risk.  
* **Frequent prior admissions** (“high utilizer” flag) are the second-strongest independent driver.

---

### 5&nbsp;&nbsp;Best-model CPT excerpts (PC)

<details>
<summary><strong>Click to view the main conditional-probability tables</strong></summary>

```text
CPD of discharge_disposition_id
  (0) Routine home  → 0.618
  (1) SNF / Rehab   → 0.382
────────────────────────────────────────────────────────────
CPD of target | discharge_disposition_id
  disposition 0 → P(readmit)=0.063
  disposition 1 → P(readmit)=0.142
────────────────────────────────────────────────────────────
CPD of number_inpatient | disposition, target
  target=1 & dispo=1 → P(high bin) = 0.497
────────────────────────────────────────────────────────────
CPD of insulin | age, change, diabetesMed, admission_type_id, disposition
  Highest-risk parent combo → 0.590 “Up/Steady” insulin
... (see full notebook for complete tables)

---
**Interpreting the CPTs**
| #     | Node / CPT shown                         | Take-away                                                                                                  |
| ----- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **1** | `discharge_disposition_id`               | 62 % routine home vs 38 % SNF/rehab/other.                                                                 |
| **2** | `insulin`                                | Dose level driven by admission type, age, med change, etc.; 59 % “Up/Steady” in highest-risk parent combo. |
| **3** | `age`                                    | Right-skewed; older bins dominate when meds are changed and disposition is non-routine.                    |
| **4** | `number_inpatient`                       | High-utiliser status ≈ 50 % when eventual readmission = 1.                                                 |
| **5** | `target`                                 | Readmit risk 14 % if disposition = SNF vs 6 % if routine home.                                             |
| **6** | `change`                                 | Med list changed in 58 % of encounters that have diabetes meds; almost never when no meds ordered.         |
| **7** | roots `admission_type_id`, `diabetesMed` | 53 % elective vs 47 % emergency; 77 % of stays involve diabetes meds.                                      |

Operational insights
• Post-acute planning (home supports vs rehab) is the highest-leverage intervention.
• Flag high-utiliser diabetics (≥ 1 prior admission) on day 1 and schedule enhanced follow-up.
• Medication intensification (change, insulin) acts as a mediator—useful for explanation, less for frontline triage thresholds.
