# Comparing Classifiers: Bank Telemarketing Term Deposit Prediction

## Project Overview
This repository contains the analysis and machine learning pipeline for **Practical Application 17.1: Comparing Classifiers**. Using the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing), this project compares four supervised classification algorithms to predict whether a client will subscribe to a long-term bank term deposit:

1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Decision Trees**
4. **Support Vector Machines (SVM)**

🔗 **[View the Complete Jupyter Notebook (`prompt_III.ipynb`)](./prompt_III.ipynb)**

---

## Methodology & Data Preparation
* **Data Cleaning & Leakage Prevention:** I examined implicit missing values and excluded the post-call `duration` feature to ensure the models only train on pre-call attributes known prior to dialing.
* **Preprocessing Pipeline:** I built a scikit-learn `ColumnTransformer` applying `StandardScaler` to numerical features and `OneHotEncoder to categorical features inside a leak-free `Pipeline`.
* **Evaluation Metric Justification:** Because the dataset exhibits severe class imbalance, raw accuracy is not showing the full picture. Models were tuned via Stratified `GridSearchCV` optimizing for **ROC-AUC** and evaluated on **F1-Score and Recall** using `class_weight='balanced'`.

---

## Tuned Model Comparison Summary (Test Set)

| Model | Avg CV Fit Time (s) | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Train ROC-AUC | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Dummy Classifier)** | 0.004 | 0.8873 | 0.8874 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| **Logistic Regression (Tuned)** | **0.258** | 0.8280 | 0.8332 | 0.3645 | **0.6466** | 0.4662 | 0.7949 | **0.8008** |
| **Decision Tree (Tuned)** | 0.409 | 0.8438 | 0.8473 | 0.3879 | 0.6153 | **0.4758** | 0.7877 | 0.7984 |
| **K-Nearest Neighbors (Tuned)** | 7.366 | 0.9023 | 0.9011 | 0.6707 | 0.2392 | 0.3527 | 0.8450 | 0.7832 |
| **Support Vector Machine (Tuned)** | 1.679 | 0.7309 | 0.7401 | 0.2532 | 0.6703 | 0.3675 | 0.7242 | 0.7481 |

* **Recommended Production Model:** **Logistic Regression (`C=0.1, class_weight='balanced'`)** and **Decision Tree (`class_weight='balanced'`)** deliver the highest Test ROC-AUC (**0.8008** and **0.7984**) and fastest training times (`< 0.45s`), capturing **64.7% of all actual subscribers** while eliminating over 60% of unproductive outbound phone calls.

---

## Summary of Key Findings & Actionable Recommendations
1. **Deploy Pre-Call Lead Scoring:** Score prospective clients prior to dialing and only contact leads exceeding the balanced probability threshold, capturing ~65% of term deposit subscriptions while cutting cold-call volume by more than half.
2. **Prioritize Past Subscribers ("Warm Leads"):** Previous campaign success (`poutcome_success`, Log-Odds = `+0.416`, Odds Ratio = `1.52` after controlling for macroeconomic factors, and `65.1%` raw conversion rate) is a major positive predictor of repeat subscription.
3. **Shift Outbound Dialing from Landlines to Mobile Phones:** Calling a landline (`contact_telephone`, Log-Odds = `-0.561`, Odds Ratio = `0.57`) reduces subscription odds by **43%** compared to cellular outreach.
4. **Target Students & Retirees (`job_retired`, Odds Ratio = `1.44`):** Students (`31.4%` conversion) and retirees (`25.2%` conversion) convert at more than double the baseline average (`11.27%`).
5. **Align Campaigns with Macroeconomic Conditions & High-Yield Months:** Lower employment variation rates (`emp.var.rate`, Log-Odds = `-1.656`) and lower Euribor 3-month rates strongly boost deposit demand, and seasonal outreach in March (`month_mar`, Odds Ratio = `3.36`), August (`1.54`), and December (`1.50`) vastly outperforms high-volume May campaigns (`month_may`, Odds Ratio = `0.59`).
