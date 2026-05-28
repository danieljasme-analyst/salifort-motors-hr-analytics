# Salifort Motors — Employee Retention Prediction

> End-to-end HR analytics project using Python, machine learning, and data storytelling.
> Built as the capstone of the [Google Advanced Data Analytics Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics).

---

## 📌 Project Summary

Salifort Motors is a fictional French-based alternative energy vehicle manufacturer with over 100,000 employees worldwide. The company is experiencing a high rate of employee turnover and wants to understand why — and predict who is likely to leave next.

As the data professional on this project, I was tasked with:
- Analyzing HR survey data to surface key drivers of attrition
- Building a predictive model to classify whether an employee will leave
- Translating findings into actionable recommendations for senior leadership

**Core business question:**
> *What factors make an employee likely to leave — and can we predict who will leave before they do?*

---

## 📁 Files in This Repository

| File | Description |
|------|-------------|
| `Salifort_Motors_Capstone.ipynb` | Full Jupyter notebook — EDA, logistic regression, decision tree, random forest, feature engineering, model evaluation |
| `PACE_Strategy_Document_Capstone.md` | PACE strategy document covering all four stages: Plan, Analyze, Construct, Execute |
| `Salifort_Motors_Capstone_Executive_Summary.pdf` | One-page executive summary for stakeholders |
| `README.md` | This file |

> **Dataset:** `HR_capstone_dataset.csv` is not included in this repo. It is publicly available on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction). Download it and place it in the same directory as the notebook before running.

---

## 📊 Dataset at a Glance

| Property | Detail |
|----------|--------|
| Source | Kaggle — HR Analytics and Job Prediction |
| Raw rows | 14,999 |
| Columns | 10 |
| After cleaning | 11,991 rows (3,008 duplicates dropped) |
| Target variable | `left` — binary (0 = stayed, 1 = left) |
| Task type | Binary classification |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `satisfaction_level` | float | Employee-reported job satisfaction [0–1] |
| `last_evaluation` | float | Score of last performance review [0–1] |
| `number_project` | int | Number of projects employee contributes to |
| `average_monthly_hours` | int | Average hours worked per month |
| `tenure` | int | Years at the company |
| `work_accident` | int | Whether employee experienced a work accident (binary) |
| `left` | int | Whether employee left the company (binary — **target**) |
| `promotion_last_5years` | int | Whether employee was promoted in last 5 years (binary) |
| `department` | str | Employee's department |
| `salary` | str | Salary level: low / medium / high |

> Columns were renamed during cleaning — `time_spend_company` → `tenure`, `Work_accident` → `work_accident`, `average_montly_hours` → `average_monthly_hours` (typo fix), `Department` → `department`.

---

## 🔬 Approach

This project follows the **PACE framework** (Plan → Analyze → Construct → Execute).

### Plan
- Defined the business problem and success criteria
- Identified stakeholders: Data Science Lead, Data Science Manager, Data Scientist, PMO, Finance Lead, Operations Lead
- Selected binary classification as the task type (target variable `left` is 0 or 1)

### Analyze — EDA Highlights

Produced 8+ visualizations to understand the data before modeling. Key findings:

- **100% of employees with 7 projects left the company** — the most striking single finding
- Employees working **240–315 hours/month** had near-zero satisfaction levels — clear burnout signal
- The nominal working baseline is ~166.67 hrs/month (50 weeks × 40 hrs ÷ 12). Most employees far exceed this
- **4-year tenured employees** show an unusually high attrition rate with unusually low satisfaction
- Employees with **6+ years at the company** overwhelmingly stayed
- Very few employees who were **promoted in the last 5 years** left
- No single department showed a dramatically different attrition rate

### Construct — Modeling

Two approaches were implemented and compared:

**Approach A — Logistic Regression**
- `salary` encoded as ordinal (low=0, medium=1, high=2)
- `department` dummy-encoded
- Tenure outliers removed (logistic regression is sensitive to outliers)
- 75/25 train/test split — `stratify=y`, `random_state=42`

**Approach B — Tree-based Models**
- Decision Tree and Random Forest, both with `GridSearchCV` (`cv=4`, `refit='roc_auc'`)
- **Round 1:** All features included
- **Round 2 (feature engineering):**
  - Dropped `satisfaction_level` — risk of data leakage (companies may not always have this on record)
  - Created binary `overworked` feature: `1` if `average_monthly_hours > 175`, else `0`
  - Dropped `average_monthly_hours` (replaced by `overworked`)

### Execute — Evaluation & Recommendations

Evaluated all models on the held-out test set. Extracted feature importances. Translated findings into six concrete stakeholder recommendations.

---

## 📈 Results

| Model | Precision | Recall | F1-Score | Accuracy | AUC |
|-------|-----------|--------|----------|----------|-----|
| Logistic Regression | ~80% | ~83% | ~80% | ~83% | — |
| Decision Tree Round 1 | ~87% | ~91% | ~89% | ~96% | ~97% |
| Random Forest Round 1 | ~88% | ~90% | ~89% | ~96% | ~98% |
| Decision Tree Round 2 | ~85% | ~90% | ~87% | ~95% | ~94% |
| **Random Forest Round 2 ★** | **~87%** | **~90%** | **~88%** | **~96%** | **~96%** |

**★ Champion model:** Random Forest Round 2 — feature-engineered, data leakage mitigated, strong and stable performance on held-out test data.

---

## 🔑 Top Features Driving Attrition

Both the Decision Tree and Random Forest models identified the same four features as most important:

| Rank | Feature | Insight |
|------|---------|---------|
| #1 | `last_evaluation` | High performers who are overworked and unrecognized are most at risk |
| #2 | `number_project` | 100% of 7-project employees left; optimal range is 3–4 projects |
| #3 | `tenure` | 4-year employees spike in attrition; 6+ year employees tend to stay |
| #4 | `overworked` | Working >175 hrs/month is a strong departure signal, especially without promotion |

---

## 💡 Recommendations

1. **Cap projects at 5** — Assigning 7 projects to any employee is a near-certain attrition signal
2. **Investigate the 4-year attrition spike** — Conduct targeted interviews; review promotion criteria at this tenure milestone
3. **Reward long hours or reduce them** — Employees should not work 200+ hours/month without recognition or additional compensation
4. **Clarify overtime and time-off policies** — Ensure all employees know their entitlements
5. **Reform evaluation scoring** — High scores should not be exclusive to those who overwork; adopt proportionate, effort-based recognition
6. **Conduct culture reviews** — Department-level and company-wide discussions to surface systemic dissatisfaction drivers

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| pandas | Data manipulation and cleaning |
| numpy | Numerical operations |
| matplotlib / seaborn | Data visualization |
| scikit-learn | Logistic Regression, Decision Tree, Random Forest, GridSearchCV, metrics |
| XGBoost | Available for further extension |
| pickle | Model serialization |
| ReportLab | Executive Summary PDF generation |

---

## 🔗 More of My Work

| | |
|---|---|
| 📓 Blog | [danieljasme.hashnode.dev](https://danieljasme.hashnode.dev) |
| 📦 Google Advanced DA — Courses 1–5 | [google-advanced-data-analytics-portfolio](https://github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio) |
| 📊 Google BI Certificate | [google-bi-portfolio](https://github.com/danieljasme-analyst/google-bi-portfolio) |

---

## 📄 License

This project was completed as part of the [Google Advanced Data Analytics Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics) on Coursera. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction) and has been repurposed for this educational project.
