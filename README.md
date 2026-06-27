# PyTorch DataLoader + Adam Optimizer
ProSensia AI/ML Bootcamp | Week 3 Day 2

## what i did today
upgraded day 11 notebook to use TensorDataset and DataLoader for mini-batch training.
also tracked validation loss every epoch to check for overfitting.

## files
- deep_learning_baseline.ipynb
- loan_default_dataset.csv
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
jupyter notebook deep_learning_baseline.ipynb

## changes from day 11
- wrapped tensors in TensorDataset
- created train_loader (batch_size=64, shuffle=True) and val_loader (shuffle=False)
- training loop now iterates over batches not full tensor
- added validation loss tracking per epoch
- 25 epochs instead of 20

## adam vs sgd
sgd uses same learning rate for all parameters
adam adapts lr per parameter using moving averages of gradients (mt) and squared gradients (vt)
adam converges faster and handles sparse features better

## why mini-batches
loading full dataset at once causes RAM issues for large data
batch_size=64 means only 64 samples in memory at a time
more frequent gradient updates = better learning
