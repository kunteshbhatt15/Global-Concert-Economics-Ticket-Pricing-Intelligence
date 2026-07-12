# Concert Ticket Pricing Intelligence

An end-to-end Machine Learning application that predicts global concert ticket prices using artist popularity, event characteristics, venue information, social media metrics, seasonality, and market conditions.
Built using Python, Scikit-Learn, XGBoost, Streamlit and deployed as an interactive web application.
---
# Project Overview
This project simulates how event organizers, promoters and venue managers can estimate average concert ticket prices before announcing an event.
Instead of relying solely on historical averages, the application combines:

- Artist popularity
- Event information
- Venue characteristics
- Social media presence
- Festival information
- Market conditions
- Calendar effects
to predict an expected average ticket price.

---

# Business Problem
Concert pricing is influenced by dozens of variables.Promoters often struggle to determine an optimal ticket price that maximizes revenue while maintaining demand.
This project develops an end-to-end machine learning pipeline capable of predicting expected average ticket prices using historical concert and artist information.

---

# Dataset
- 20,000+ Global Concert Events
- Years Covered: 2016–2026
- 69 Features
- Target Variable:
    Average Ticket Price (USD)
---

# Tech Stack
Python
Pandas
NumPy
Scikit-Learn
XGBoost
Plotly
Streamlit
Joblib

---

# Machine Learning Workflow

Data Cleaning

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Encoding

↓

Scaling

↓

Feature Selection (VIF)

↓

Model Training

↓

Hyperparameter Tuning

↓

Model Comparison

↓

Final XGBoost Model

↓

Deployment using Streamlit

---

# Models Compared

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- XGBoost Regressor

---

# Final Model Performance

| Metric | Score |
|---------|-------|
| R² | 0.896 |
| MAE | $15.74 |
| RMSE | $20.73 |

---

# Application Features

## Dashboard

Executive KPIs

Genre-wise Analysis

Price Distribution

Artist Trends

---

## Price Predictor

Predict ticket prices using 60+ business features.

---

## What-if Simulator

Compare Conservative

Base

High Demand

Scenarios

---

## Model Intelligence

Model Metrics

Feature Importance

Performance Summary

---

## Data Explorer

Interactive dataset filtering

Search

Analysis


---

# Repository Structure

app.py

artifacts/

notebooks/

screenshots/

requirements.txt

README.md

---

# Future Improvements

- SHAP Explainability
- Docker Deployment
- REST API using FastAPI
- User Authentication
- Cloud Deployment
- Auto Retraining Pipeline

---

# Author

**Kuntesh Bhatt**

M.Tech Data Science & AI

Data Analyst | Business Analytics | Machine Learning
