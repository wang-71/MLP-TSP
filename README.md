# MLP-based Deep Reinforcement Learning for the Traveling Salesman Problem (TSP)

This repository implements a **REINFORCE-based deep reinforcement learning (DRL)** approach for the Traveling Salesman Problem (TSP), using a **multilayer perceptron (MLP)** as the policy network to learn a heuristic routing strategy.

The project serves as a lightweight baseline for learning-based combinatorial optimization and provides a clear, end-to-end example of training and evaluating a neural policy for TSP.

---

## Setup

This project uses **Python 3.12.7**. Please install the corresponding Python version following the official Python documentation.

It is recommended to install dependencies in a virtual environment.

### Install required packages
```bash
pip install torch gymnasium tqdm
