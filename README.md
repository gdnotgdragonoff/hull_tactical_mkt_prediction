# Hull Tactical Market Prediction
A full reproducible pipeline for EDA, feature engineering, model training and backtesting.

This repository contains a complete end-to-end workflow for the Kaggle Hull Tactical Market Prediction competition.
The goal is to build a machine-learning pipeline that predicts short-horizon stock returns and evaluates trading strategies on top of model outputs.

## Project Overview

The original dataset and competition description come from Kaggle. https://www.kaggle.com/competitions/hull-tactical-market-prediction
The goal is to predict a target variable (forward returns) using numerical market, macroeconomic and sentiment features.

This repository provides a clean, extendable engineering structure where each step:

·Loads and cleans data

·Performs EDA and quality checks

·Selects features based on statistics + model importance

·Constructs engineered features

·Trains a LightGBM classifier

·Generates predictions for backtesting

·Performs strategy evaluation with several position-mapping rules and volatility targeting

hull_tactical_market_prediction/
├─ README.md
├─ .gitignore
├─ requirements.txt / pyproject.toml
│
├─ data/                 # Not included in repo (gitignored)
│   ├─ raw/              # train.csv / test.csv from Kaggle
│   └─ interim/          # cleaned data, parquet/intermediate files
│
├─ notebooks/
│   ├─ 01_eda_overview.ipynb
│   ├─ 02_feature_selection.ipynb
│   └─ 03_model_backtest.ipynb
│
├─ src/
│   └─ hull_tactical/
│       ├─ __init__.py
│       ├─ config.py
│       ├─ paths.py
│       ├─ data_loading.py
│       ├─ eda.py
│       ├─ cleaning.py
│       ├─ feature_selection.py
│       ├─ feature_engineering.py
│       ├─ model_training.py
│       ├─ backtest.py
│       └─ metrics.py
│
├─ scripts/
│   ├─ run_cleaning.py
│   ├─ run_feature_selection.py
│   ├─ run_feature_engineering.py
│   ├─ run_training.py
│   └─ run_backtest.py
│
├─ configs/
│   ├─ base_config.yml
│   ├─ model_lgbm_classifier.yml
│   └─ backtest_grid.yml
│
├─ results/
│   ├─ figures/
│   ├─ tables/
│   ├─ backtest_reports/
│   └─ logs/
│
└─ artifacts/
    ├─ models/               # saved LightGBM models
    └─ configs_snapshot/     # saved configs for reproducibility
