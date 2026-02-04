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

## Training

To start training the agent, run:
```bash
python reinforce.py

The training process applies the REINFORCE algorithm to iteratively improve the policy based on episodic rewards.

## Environment

The TSP environment is implemented using the Gymnasium framework.
It defines the state representation, action space, reward function, and episode termination conditions for the TSP.

## Agent 

The agent consists of:

A multilayer perceptron (MLP) policy network

A forward method that maps environment states to action probabilities

A masking vector is applied to prevent revisiting nodes

## Baseline

A Nearest Neighbor (NN) heuristic is implemented as a baseline to evaluate the performance of the learned policy.

## Results

After approximately 24 hours of training, the model was trained for ~6,800 epochs.
Selected policy checkpoints are saved in the policies/ directory.

