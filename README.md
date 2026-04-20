# Predicting Customer Churn in Italian Motor Insurance

An end-to-end machine learning project focused on predicting customer churn for an Italian motor insurance company, with the goal of enabling targeted retention campaigns and improving long-term customer retention.

---

##  Project Overview

Customer churn is one of the most critical challenges in the insurance industry. In highly competitive markets, policyholders can easily switch providers at renewal — making the annual contract expiry a natural and recurring risk point for every insurer. In this project, we developed a predictive model to identify customers most likely to **not renew their motor insurance policy**, so the company can intervene proactively with personalised retention actions before the renewal window closes.

The observed churn rate in the dataset is approximately **21.5%** — roughly 1 in 5 customers does not renew their policy each year.

The project covers the full machine learning lifecycle:
- Business understanding and problem framing
- Data cleaning and preparation
- Churn target definition
- Exploratory data analysis
- Feature engineering and selection
- Predictive modelling and evaluation
- Business interpretation of model outputs

---

##  Business Problem & Objective

The company's strategic goal is to **reduce customer abandonment** by identifying clients who are unlikely to renew their motor insurance contracts before they actually leave.

A predictive model that estimates the churn probability within one year allows the business to:
- Prioritize high-risk customers for outreach
- Optimize retention campaigns and reduce wasted spend
- Allocate CRM resources more efficiently
- Maximize customer lifetime value

---

##  Data Sources

The project integrates four sources of insurance data, merged at the customer level. All raw files were originally bundled in `5_CHURN_AUTO.zip`:

| Dataset | Key Fields |
|---|---|
| **Client data** (`CHURN_CLIENT_INFO.csv`) | Age, seniority, number of accidents, app usage, risk rating, province of residence |
| **Contract data** (`CHURN_CONTRACT_AUTO.csv`) | Premium amounts, discounts, payment method, number of installments, product/risk type, warranty/coverage flags |
| **Contract dates** (`CHURN_AUTO_CONTRACT_DATES.csv`) | Effective date, closing date, expiration date |
| **Vehicle data** (`CHURN_INFOAUTO.csv`) | Commercial value, vehicle type, engine power, yearly mileage, ABS/airbag/anti-theft flags |

All column names were translated from Italian to English during the cleaning phase.

---

##  Unit of Analysis & Timeframe

The **unit of analysis is customer-year**: one row per customer per year. The project tracks the full customer base active in **2015** and follows complete contract history through **2020**, covering approximately **908,000 observations** across 43 variables.

The final modelling dataset is restricted to contracts expiring between **January 2015 and June 2020**.

---

##  Churn Definition

A customer is classified as **churned** when their policy expires and they **do not open a new contract within 366 days** of the expiry date. This definition reflects the real business meaning of churn and was engineered from raw contract data through the following steps:

- **`WINDOW_END`** — expiry date + 366 days (the renewal observation window)
- **`NEXT_CONTRACT_DATE`** — the start date of the client's next contract, computed chronologically per client
- **`CONTRACT_RENEWED`** — `1` if a new contract started within the window, `0` otherwise
- **`EARLY_TERMINATION`** — `1` if the contract closed before its natural expiry date
- **`CHURN`** — `1` if the contract was not renewed (covers both natural expiry and early termination without renewal)

The project also distinguishes between:
- Early termination without renewal (churn)
- Early termination followed by renewal (retained)
- Normal expiry followed by renewal or non-renewal

An additional **`Previously_Churned`** feature captures whether a client has ever churned in prior years, providing a strong historical behavioural signal.

Contracts with administrative reversal or cooling-off cancellation statuses were excluded, as they do not represent genuine churn behaviour.

---

##  Data Preparation

Key preprocessing steps:
- Parsed date columns (`Effective_Date`, `Expiry_Date`, `Closing_Date`) into proper datetime format
- Translated all column names from Italian to English
- Dropped columns with high null rates (mostly property-related coverage flags not applicable to auto policies)
- Mapped binary flags (`Presence_ABS`, `Presence_Airbag`, `Presence_Theft_Protection`, `Children_Flag`) from `S/N` or `Y/N` to `1/0`
- Aggregated contract-level data into one row per customer per year using appropriate strategies: `sum` for financial amounts, `max` for binary flags, `mean` for installments, `count` for number of contracts
- Removed leakage-prone features: columns directly used to construct the CHURN label (`EARLY_TERMINATION`, `NEXT_CONTRACT_DATE`, `WINDOW_END`) and `Status`, which encodes the contract renewal outcome and caused artificial accuracy inflation to ~92%
- Removed multicollinear features: `Effective__Year` (correlation 0.983 with `Expiry__Year`) and `Presence_Airbag` (correlation 0.871 with `Presence_ABS`)

---

##  Exploratory Data Analysis

The EDA covered nine analytical dimensions and produced three sets of saved visualisations (`EDA_Key_Insights.png`, `EDA_Coverage_vs_Churn.png`, `EDA_Previously_Churned.png`). Key findings:

- **Churn remained stable over time** at around 21%, with no dramatic year-on-year spikes between 2015 and 2020
- **New and mid-tenure customers churn most**: clients in their first year and those with 4–7 years of seniority showed the highest churn rates
- **Optional coverages reduce churn significantly**: customers with zero add-on coverages had materially higher churn rates than those with one or more
- **Previously churned customers churn again**: historical churn is one of the strongest predictors of future churn
- **Discounts alone don't retain everyone**: low-premium customers churned regardless of discount level, suggesting price sensitivity is not the only driver
- **Accident history matters**: the number of reported accidents correlates with renewal behaviour

---

##  Feature Engineering & Selection

Feature selection combined correlation analysis (keeping features with |corr| ≥ 0.05 against the target), multicollinearity checks (dropping one of any pair with inter-feature correlation > 0.80), zero-variance checks, and domain knowledge.

One engineered feature was created: **`Total_Warranties`** — the sum of key optional coverage flags held by the customer, which captures product depth in a single signal.

**Final feature set (14 features):**

| Feature | Type | Description |
|---|---|---|
| `Active_Seniority` | Numerical | Years the client has been active |
| `Contract_Discount_Amount` | Numerical | Total discount applied to the contract |
| `Year_of_Birth` | Numerical | Client birth year (proxy for age) |
| `Contract_Premium_Amount` | Numerical | Total annual premium |
| `Sum_of_Premiums_Paid` | Numerical | Cumulative lifetime premiums paid |
| `Commercial_Value` | Numerical | Estimated commercial value of the insured vehicle |
| `Presence_ABS` | Binary | Whether the vehicle has ABS |
| `Num_Installments` | Numerical | Number of payment installments |
| `Number_of_Accidents` | Numerical | Total reported accidents |
| `Total_Warranties` | Engineered | Count of optional coverages held |
| `Business_Grouping` | Categorical | Business line classification |
| `Payment_Method_Signing` | Categorical | Payment method at contract signing |
| `Installment_Payment_Method` | Categorical | Payment method for installments |
| `Previously_Churned` | Binary | Whether the client has ever churned before |

Among the strongest predictive signals: **seniority**, **discount**, **previous churn history**, and **selected coverage indicators**.

---

##  Modelling

Three model types were trained and compared across two feature configurations — a baseline set with feature engineering (`Total_Warranties`) and an expanded set using raw coverage columns instead:

- **Logistic Regression** — interpretable baseline
- **XGBoost** — gradient boosting with `scale_pos_weight` for class imbalance
- **LightGBM** — gradient boosting with `class_weight='balanced'`

To ensure realistic evaluation, an **out-of-time split** was used: models were trained on past observations and tested on future observations. Class imbalance (~21.5% churn rate) was addressed through class weighting in all models.

A 30% stratified sample of the full dataset was used during model comparison and hyperparameter tuning to reduce compute time while preserving the churn ratio.

### Results

| Model | Accuracy | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 62.01% | 74.40% | 76.60% | 33.33% |
| XGBoost | 71.90% | 92.80% | 88.50% | 43.10% |
| **LightGBM**  | **71.50%** | **93.40%** | **88.40%** | **42.60%** |

**LightGBM was selected as the champion model** based on its highest ROC-AUC (93.4%) and strong recall. Recall was prioritised over precision because in a retention context, **missing a churner is more costly than contacting a false positive**.

In practical terms:
- The model catches approximately **88 out of 100 customers who would have churned**
- Around **43 out of 100 customers contacted** are genuinely at risk
- Even with moderate precision, contacting false positives is still worthwhile given the cost of acquiring a new customer vs. retaining an existing one

### Hyperparameter Tuning

The LightGBM champion was fine-tuned using **`RandomizedSearchCV`** with:
- 50 random iterations
- 5-fold stratified cross-validation
- Scoring metric: `roc_auc`

Parameters searched: `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`

The final model was then retrained on the **full 750K+ dataset** (80/20 stratified split) and serialised to `lgbm_churn_final.pkl`.

---

##  Business Insights

The project identified the main high-risk customer segments:

- **New clients** in their first year with the company
- **Mid-tenure clients** with 4–7 years of seniority
- Customers with **no optional coverages**
- Customers with **low premiums and no discount**
- Customers with a **prior churn history**

### Recommended Retention Strategy

| Segment | Recommended Action |
|---|---|
| High-risk clients (model score) | Contact 60–90 days before renewal |
| Mid-tenure clients | Loyalty rewards and acknowledgement |
| New clients | Onboarding value communication |
| Low-coverage clients | Propose relevant add-on coverages |
| Previously churned | Personalised win-back outreach |

Churn scores should be integrated directly into CRM workflows to enable automated, prioritised outreach at scale.

---

##  Limitations & Next Steps

**Current limitations:**
- Missing behavioural signals: no call centre interaction data, no digital engagement history
- No telematics data (driving behaviour, mileage tracking)
- No competitor pricing information

**Recommended next steps:**
- Integrate model outputs into CRM systems for automated scoring
- Monitor campaign outcomes and track model performance over time
- Retrain the model every 6–12 months as new data becomes available
- Enrich the dataset with call centre, digital, and telematics signals

---

##  Running the Project

1. Clone the repository and place `5_CHURN_AUTO.zip` in the project root
2. Open `DSBA_ML_Churn_prediction.ipynb` in Jupyter or Google Colab
3. Run all cells sequentially — the notebook is structured in 10 sections:

| Section | Description |
|---|---|
| 1 | Import libraries and load raw data |
| 2 | Data cleaning and column renaming |
| 3 | Data exploration and initial merging |
| 4 | Build the final modelling dataset |
| 5 | Churn label construction |
| 6 | Exploratory Data Analysis |
| 7 | Feature selection |
| 8 | Model training and comparison |
| 9 | WandB experiment tracking |
| 10 | Hyperparameter tuning and final model training |


---

##  Experiment Tracking

All model runs are logged to **Weights & Biases (WandB)** under the project `Machine Learning Project _Churn`. Tracked metrics per run: accuracy, precision, recall, F1 score, and ROC-AUC.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | Preprocessing, pipelines, evaluation |
| LightGBM | Champion model |
| XGBoost | Comparison model |
| matplotlib / seaborn | Visualisation |
| WandB | Experiment tracking |
| Jupyter Notebook | Development environment |
| pickle | Model serialisation |

---

##  Credits

Developed as part of the **Data Science & Business Analytics (DSBA)** programme at **Bologna Business School**, in collaboration with **AnalyticsNetwork**.
