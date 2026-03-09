# CRM Churn Prediction Model

## Overview

Customer churn, the loss of clients who choose not to renew or continue a service, is one of the most critical challenges facing insurance companies today. Acquiring a new customer is significantly more expensive than retaining an existing one, making churn prediction a high-priority business problem.

This project addresses that challenge head-on. Built for an **Italian insurance company operating in the automotive and property insurance markets**, this repository contains the work behind an end-to-end **anti-churn machine learning model** designed to predict which customers are most likely to leave within the next 12 months. The goal is not just prediction, it is actionable intelligence: by identifying at-risk clients early, the company can deploy **targeted retention campaigns** before a customer decides to walk away.

---

## Business Problem

In the insurance industry, contracts are typically annual. This means that every year, every single customer faces a natural decision point: renew or leave. Unlike subscription services where inertia keeps customers subscribed, insurance customers actively re-evaluate their options at renewal time, making churn a structural, recurring risk.

The core challenge this project aims to solve is:

> **Which customers, among the entire 2015 customer base, are most likely to not renew their contract within the next year?**

Answering this question accurately allows the business to:
- Prioritize outreach to high-risk customers before their contract expires
- Design personalized retention offers based on customer profiles
- Reduce overall churn rate and improve long-term revenue stability
- Optimize the marketing budget by focusing retention spend where it matters most

---

## Dataset

The model is trained and validated on **detailed contract and client data spanning from 2015 to 2020**, providing a rich longitudinal view of customer behavior over time.

### Client Information
- **Client type** - e.g., individual vs. corporate
- **Residence location** - geographic data that may reflect regional risk profiles or competitive dynamics
- Other registry-level demographics

### Contract Information
- **Contract type** - the category of insurance product held by the client
- **Installment structure** - how the client pays (monthly, quarterly, annually), which may correlate with financial engagement
- **Underwriting conditions** - the specific terms and risk classifications associated with each policy

### Contract Timeline Data
- **Effective date** - when the contract began
- **Closure date** - if and when a contract was terminated early
- **Expiration date** - the natural end date of the contract, which is critical for defining the churn event

### Insured Asset Characteristics
- Details about the goods or assets covered under each policy (e.g., vehicle type, property details), which can influence renewal likelihood

---

## Project Structure & Key Aspects

### 1. Training Setting
A careful training setup is essential in churn modeling to avoid data leakage and ensure the model generalizes to real-world conditions. Since we are predicting churn one year in advance, the model must be trained only on information that would have been available *before* the prediction window - no future data can leak into the features. The 2015 customer base serves as the cohort of interest, and historical contract behavior up to that point is used to construct features.

### 2. Model Specification
The model specification phase involves selecting the right algorithm, features, and evaluation criteria for this specific churn problem. Given the class imbalance typical of churn datasets (most customers do *not* churn), special attention is paid to:
- Handling imbalanced classes (e.g., oversampling, class weighting)
- Choosing metrics beyond accuracy such as precision, recall, AUC-ROC, and lift curves — that better reflect business value
- Exploring both interpretable models (e.g., logistic regression, decision trees) and high-performance ensemble models (e.g., gradient boosting)

### 3. Value-Added Indicators
Beyond raw churn probability scores, this project aims to define and engineer **value-added indicators**: features and outputs that enrich the business understanding of churn. These may include:
- A **churn risk score** per client (probability of not renewing)
- **Customer lifetime value (CLV)** estimates to prioritize retention efforts by business impact
- **Segment-level churn rates** to identify systemic risks within specific product lines or geographies
- **Retention campaign targeting lists** ranked by expected ROI

---

## Expected Outcomes

The primary deliverable of this project is a **trained, validated anti-churn model** that can be operationalized within the company's CRM workflows. Concretely, this means:

- A model that outputs a **churn probability score** for each active customer
- A **ranked list of at-risk customers** for the retention team to act upon
- Insights into the **key drivers of churn**, enabling smarter product and service decisions
- A framework that can be **re-run annually** as new contract cohorts become available

---

## Credits

Developed in collaboration with **AnalyticsNetwork** - a data science and analytics consultancy specializing in applied machine learning for business.
