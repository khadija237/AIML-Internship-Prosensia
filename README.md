# Deep Learning Baseline - PyTorch MLP
ProSensia AI/ML Bootcamp | Week 3 Day 1

## what i built
first deep learning model using pytorch. converted loan default dataset into tensors and trained a multi layer perceptron from scratch without using sklearn fit.

## files
- deep_learning_baseline.ipynb
- loan_default_dataset.csv
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
jupyter notebook deep_learning_baseline.ipynb

## model
Input(15) -> Linear(64) -> ReLU -> Dropout -> Linear(32) -> ReLU -> Dropout -> Linear(2)

## training
20 epochs, manual loop:
- zero_grad
- forward pass
- loss calculation
- backward (backprop)
- optimizer step

## numpy vs pytorch tensor
numpy: cpu only, no gradients
pytorch: cpu+gpu, autograd tracks gradients for backprop

## relu vs sigmoid
sigmoid has vanishing gradient problem for deep networks
relu = max(0,x), gradient doesnt go to zero, trains much better
