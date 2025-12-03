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
