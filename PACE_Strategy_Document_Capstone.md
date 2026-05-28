# PACE Strategy Document — Google Advanced Data Analytics Capstone
## Salifort Motors: Employee Retention Prediction

**Project:** Capstone — Salifort Motors HR Analysis  
**Prepared by:** Ahmad Daniel  
**Date:** 2024  
**Framework:** PACE (Plan → Analyze → Construct → Execute)

---

## PLAN

### Business Scenario

Salifort Motors, a fictional French-based alternative energy vehicle manufacturer with over 100,000 employees worldwide, is experiencing a high rate of employee turnover. The senior leadership team is concerned about how many employees are leaving and has commissioned an analysis to understand and predict attrition.

The HR department recently surveyed a sample of employees to identify what might be driving turnover. You have been asked to:
1. Analyze the HR survey data
2. Identify the key factors driving employee departure
3. Build a predictive model that estimates whether an employee will leave the company

### Business Problem

**What's likely to make an employee leave the company?**

High turnover is costly — Salifort makes a significant investment in recruiting, training, and upskilling employees. A model that can predict which employees are likely to leave allows the company to take proactive steps to improve retention, job satisfaction, and ultimately save time and money.

### Stakeholders

| Name | Role |
|------|------|
| Willow Jaffey | Data Science Lead |
| Rosie Mae Bradshaw | Data Science Manager |
| Orion Rainier | Data Scientist |
| Mary Joanna Rodgers | Project Management Officer |
| Margery Adebowale | Finance Lead |
| Maika Abadi | Operations Lead |

### Dataset Summary

**File:** `HR_capstone_dataset.csv`  
**Rows:** 14,999 (one row per employee)  
**Columns:** 10

| Column | Type | Description |
|--------|------|-------------|
| satisfaction_level | float | Employee-reported job satisfaction [0–1] |
| last_evaluation | float | Score of last performance review [0–1] |
| number_project | int | Number of projects employee contributes to |
| average_monthly_hours | int | Average hours worked per month |
| time_spend_company (→ tenure) | int | Years at the company |
| work_accident | int | Whether employee experienced a work accident (binary) |
| left | int | Whether employee left the company (binary — target variable) |
| promotion_last_5years | int | Whether employee was promoted in last 5 years (binary) |
| Department | str | Employee's department |
| salary | str | Salary level: low / medium / high |

### Initial Observations

- The target variable `left` is binary (0 = stayed, 1 = left), making this a **binary classification** problem.
- The dataset contains misspellings and inconsistent formatting that must be cleaned (e.g., `average_montly_hours`, `Work_accident`, `time_spend_company`).
- `salary` is ordinal — it should be encoded as numbers (0=low, 1=medium, 2=high), not dummy-encoded.
- `department` is nominal — it should be dummy-encoded.

### Plan: PACE Task Classification

| PACE Stage | Tasks |
|------------|-------|
| **Plan** | Understand business problem, review dataset, identify stakeholders, select modeling approach |
| **Analyze** | Load and clean data, EDA, visualizations, statistical summaries |
| **Construct** | Feature encoding, train/test split, Logistic Regression model, Decision Tree model, Random Forest model, feature engineering |
| **Execute** | Evaluate models, extract feature importances, business recommendations, next steps |

### Modeling Strategy

Two approaches will be used:
- **Approach A:** Logistic Regression (statistical baseline)
- **Approach B:** Tree-based Machine Learning (Decision Tree + Random Forest with GridSearchCV)

The primary evaluation metric is **AUC (Area Under the ROC Curve)** for the tree-based models, and weighted F1/accuracy/recall for logistic regression.

### Ethical Considerations — Plan Stage

- The dataset is self-reported by employees, so there may be bias in how employees describe their satisfaction and working conditions.
- Predictions should be used ethically — to improve working conditions, not to unfairly penalize employees identified as "at risk."
- Sensitive features (department, salary) should be used carefully to avoid reinforcing systemic biases.

---

## ANALYZE

### Data Cleaning Steps

1. **Rename columns** for consistency and correctness:
   - `Work_accident` → `work_accident`
   - `average_montly_hours` → `average_monthly_hours` (fix typo)
   - `time_spend_company` → `tenure`
   - `Department` → `department`

2. **Check for missing values** — None found.

3. **Check for duplicates** — 3,008 duplicate rows found (~20% of data). These are dropped.

4. **Check for outliers** — Outliers exist in `tenure` (employees with very long tenures). These are handled differently depending on the model:
   - For Logistic Regression: outliers in `tenure` are **removed** (logistic regression is sensitive to outliers).
   - For tree-based models: outliers are **kept** (decision trees are robust to outliers).

### Key EDA Observations

**Projects and Monthly Hours:**
- Employees who worked on 7 projects **all left** — this is a strong signal.
- There appear to be two groups of employees who left: those who worked far fewer hours than peers (possibly fired), and those who worked far more (likely burned out and quit).
- The optimal number of projects is 3–4; employees in this range had the lowest attrition rate.
- Most employees work significantly more than the nominal 166.67 hrs/month (~50 weeks × 40 hrs / 12 months).

**Satisfaction and Hours:**
- A cluster of employees worked ~240–315 hours/month with near-zero satisfaction — strong burnout signal.
- Another group with moderate hours (~155–175) also left, with satisfaction around 0.4.

**Satisfaction and Tenure:**
- Employees who left fall into two groups: dissatisfied short-tenured employees and satisfied medium-tenure employees (4–6 years).
- 4-year tenured employees show an unusually low satisfaction level — worth investigating.
- Employees with 6+ years at the company overwhelmingly stayed.

**Salary by Tenure:**
- Long-tenured employees are not disproportionately high-paid, which may contribute to dissatisfaction.

**Monthly Hours and Evaluation Scores:**
- There is a positive correlation between hours worked and evaluation scores.
- Employees who worked the most hours and had the highest evaluations were among those who left — overwork without promotion is a driver.

**Promotions:**
- Very few employees who worked the most hours were promoted in the last 5 years.
- Virtually all employees who were promoted in the last 5 years stayed.

**Department:**
- No single department shows a dramatically different attrition rate from others.

**Correlation Heatmap:**
- `number_project`, `average_monthly_hours`, and `last_evaluation` are positively correlated with each other.
- `satisfaction_level` is negatively correlated with `left`.

### Ethical Considerations — Analyze Stage

- The distributions have an unusual shape for some variables (e.g., satisfaction levels), which may indicate synthetic or manipulated data. Results should be interpreted cautiously.
- Decisions made based on EDA observations should not be used to discriminate against employees.

---

## CONSTRUCT

### Feature Engineering

**Logistic Regression:**
- Encode `salary` as ordinal: low=0, medium=1, high=2
- Dummy-encode `department`
- Remove tenure outliers (IQR method) — logistic regression is outlier-sensitive
- Split: 75% train / 25% test, `stratify=y`, `random_state=42`

**Tree-based Models (Round 1):**
- Same encoding as above (no outlier removal needed)
- Split: 75% train / 25% test, `stratify=y`, `random_state=0`

**Tree-based Models (Round 2 — Feature Engineering):**
- Drop `satisfaction_level` (possible data leakage — company may not always have access to this)
- Create `overworked` binary feature: 1 if `average_monthly_hours > 175`, else 0
- Drop `average_monthly_hours` (replace with `overworked`)

### Model Hyperparameter Search

Decision Tree and Random Forest both use `GridSearchCV`:
- `cv=4`
- `refit='roc_auc'`
- Scoring: `{'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}`

**Decision Tree hyperparameters searched:**
- `max_depth`: [4, 6, 8, None]
- `min_samples_leaf`: [2, 5, 1]
- `min_samples_split`: [2, 4, 6]

**Random Forest hyperparameters searched:**
- `max_depth`: [3, 5, None]
- `max_features`: [1.0]
- `max_samples`: [0.7, 1.0]
- `min_samples_leaf`: [1, 2, 3]
- `min_samples_split`: [2, 3, 4]
- `n_estimators`: [300, 500]

### Ethical Considerations — Construct Stage

- The model should not be used to automatically terminate or disadvantage employees identified as high attrition risk.
- Model outputs should inform conversations and process improvements, not replace human judgment.

---

## EXECUTE

### Summary of Model Results

| Model | Precision | Recall | F1-Score | Accuracy | AUC |
|-------|-----------|--------|----------|----------|-----|
| Logistic Regression | ~80% | ~83% | ~80% | ~83% | — |
| Decision Tree (Round 1) | ~87% | ~91% | ~89% | ~96% | ~97% |
| Random Forest (Round 1) | ~88% | ~90% | ~89% | ~96% | ~98% |
| Decision Tree (Round 2) | ~85% | ~90% | ~87% | ~95% | ~94% |
| Random Forest (Round 2) | ~87% | ~90% | ~88% | ~96% | ~96% |

**Champion model:** Random Forest (Round 1 or Round 2 depending on leakage concern)

### Top Feature Importances

Both the Decision Tree and Random Forest models identified the same top features:
1. `last_evaluation` — highest importance
2. `number_project`
3. `tenure`
4. `overworked` (engineered feature)

### Business Recommendations

1. **Cap the number of projects** employees can work on simultaneously — especially prevent assignment of 7 projects.
2. **Investigate 4-year employee dissatisfaction** — consider structural changes to promotion criteria or workload at this tenure milestone.
3. **Reward or reduce long working hours** — employees should not be expected to work 200+ hours/month without recognition.
4. **Clarify overtime pay and time-off policies** — ensure employees are aware of their rights and entitlements.
5. **Reform evaluation scoring** — high scores should not be reserved exclusively for those who overwork; effort and efficiency should be recognized proportionally.
6. **Conduct culture reviews** by department and company-wide to understand drivers of dissatisfaction.

### Next Steps

- Consider removing `last_evaluation` from the model as well (evaluations may not be frequent), to test whether predictions remain strong without it.
- Consider a K-means clustering analysis to segment employees by behavior patterns — may yield actionable insights.
- Monitor model performance over time and retrain with new survey data as it becomes available.

### Ethical Considerations — Execute Stage

- Any attrition predictions should be used to **improve working conditions**, not to preemptively penalize predicted leavers.
- Employees should not be made aware that they are being scored by an algorithm in a way that could cause harm or bias.
- Ensure that model recommendations are reviewed by HR and management before being acted upon.
