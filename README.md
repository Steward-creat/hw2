# Multi-Armed Bandit Homework

This repository contains the simulation of a Multi-Armed Bandit problem comparing 6 different exploration-exploitation strategies:
- **A/B Test**
- **Optimistic Initialization**
- **Epsilon-Greedy**
- **Softmax**
- **Upper Confidence Bound (UCB)**
- **Thompson Sampling**

## Problem Setup
- 3 Arms: A (mean=0.8), B (mean=0.7), C (mean=0.5)
- Budget: 10,000 pulls ($10,000)

## Files Included
- `hw2_6_strategies.py`: Main Python script containing the implementation of all 6 strategies and generating the evaluation metrics/charts.
- `hw2_comparison_table.csv`: Auto-generated table comparing the different methods based on allocation, total expected reward, and regret.
- `hw2_cumulative_reward.png`: Plot of the average cumulative reward over 10,000 pulls across 200 simulation runs.
- `hw2_cumulative_regret.png`: Plot of the average cumulative expected regret.
- `hw2_cumulative_regret_log.png`: Plot of the average cumulative expected regret in log scale.

## How to Run
1. Ensure you have the required dependencies installed:
   ```bash
   pip install numpy pandas matplotlib
   ```
2. Run the Python script:
   ```bash
   python hw2_6_strategies.py
   ```
   This will execute the simulations across 200 runs and output the CSV table along with the result plots in the current directory.
