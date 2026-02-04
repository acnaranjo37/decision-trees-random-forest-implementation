# Decision Trees & Random Forest Implementation (From Scratch)

This repository contains a custom implementation of **Decision Trees** and **Random Forest** algorithms built entirely in Python using `NumPy`. The goal of this project is to demonstrate a deep understanding of the underlying logic of these machine learning algorithms without relying on high-level libraries like Scikit-Learn for the core logic.

## 🚀 Key Features
* **Custom Node Class:** Recursive structure to build the tree.
* **Training Algorithms:**
    * Implementation of **Information Gain (Entropy)** for split criteria.
    * Handling of continuous variables via dynamic thresholding.
* **Random Forest:** Ensemble method implementation with bootstrapping and feature subsampling.
* **Optimization:** Includes hyperparameters for `max_depth`, `min_samples_split`, and feature selection.

## 🛠️ Technologies
* **Python**
* **NumPy** (for efficient matrix operations)
* **Pandas** (for data manipulation)
* **Scikit-Learn** (Used *only* for data loading and preprocessing, not for the model logic).

## 📂 Project Structure
* `decision_trees.py`: Contains the `Nodo`, `ArbolDecision`, and `RandomForest` classes.
* `data_loader.py`: Utilities to load datasets (Titanic, IMDB, MNIST Digits, etc.).

## 📊 Datasets Tested
The models were tested and hyper-parameter tuned on several datasets:
* Titanic (Classification)
* IMDB (Sentiment Analysis / NLP)
* MNIST Digits (Image Classification)
* Adult Census Income
