# Advanced ML Project — ECG Classification with Lottery Ticket Hypothesis

This project explores the **Lottery Ticket Hypothesis (LTH)** in the context of **ECG (electrocardiogram) classification**.  
LTH states that large randomly-initialized neural networks contain **sparse subnetworks** that can be trained in isolation to achieve performance comparable to the original dense model.  

The goal is to reproduce iterative pruning experiments, compare **weight rewinding** and **fine-tuning**, and evaluate different pruning strategies using ECG data from the PhysioNet CinC Challenge 2017 dataset.

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Dataset Preparation](#dataset-preparation)  
5. [Launch Experiments](#4launch-experiments)

---

## 1. Overview

This repository implements iterative magnitude-based pruning to identify sparse “winning tickets” in a deep neural network trained for ECG classification.  
Inspired by:

- Frankle & Carbin (2019), *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*  
- Sahu et al. (2022), *LTH-ECG: Applying Lottery Ticket Hypothesis to ECG Classification*

The network architecture is a 1D ResNet variant trained to classify ECG rhythms. Pruning schemes include:

- Global magnitude pruning  
- Weight rewinding  
- Fine-tuning without rewinding  

---

## 2. Repository Structure
```text
Advanced-ML-Project/  
    ├── checkpoints/ # Saved model masks and states 
    ├── plots/ # Result figures generated during runs  
    ├── utils/ # Utility modules
    ├── build_datasets.py # Script to build train/dev splits
    ├── iterative_pruning_f1.ipynb # Notebook: iterative pruning & F1 tracking
    ├── lth_ecg_final.ipynb # Main experiment notebook
    ├── resnet1d.py # ResNet1D model definition
    ├── requirements.txt # Python dependencies
    ├── setup.sh # Setup script for environment
    ├── train.json # Training metadata
    ├── dev.json # Development/validation metadata
    └── README.md # This README
```

---

## 3. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/felixdechamps/Advanced-ML-Project.git
cd Advanced-ML-Project
python3 -m venv venv
source venv/bin/activate
./setup.sh
```
The `setup.sh` file will automatically : 
- install the libraries from requirements.txt. 
- create a data/ folder and download data from the PhysioNet Computing in Cardiology Challenge 2017 dataset: https://physionet.org/content/challenge-2017/1.0.0/ inside.
- launch build_datasets.py to create train.json and dev.json files. 

Ensure you have Python 3.8+ and a CUDA-enabled GPU for efficient training.  

---

## 4. Launch experiments 

- `lth_ecg.ipynb` allows to run LTH-ECG an experiment using the framework described in Sahu et al. (2022) for finding Winning Tickets. 
- `iterative_pruning.ipynb` allows to run experiment using the iterative pruning framework described in Frankle and Carbin (2019) with or without (fine-tuning) weights rewinding at each round.  

Results are stored in `checkpoints/` and visualized in `plots/.`  

In case you want to restart an experiment where it stopped use the `resume=True` parameter in the `run_lth_ecg` or `iterative_pruning` functions.

