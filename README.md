# Regularization - BatchNorm + Dropout
ProSensia AI/ML Bootcamp | Week 3 Day 3

## what i did
added batchnorm1d and dropout to the mlp from day 12.
trained both baseline and regularized model for 30 epochs and compared their loss curves.

## files
- deep_learning_baseline.ipynb
- loan_default_dataset.csv
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
jupyter notebook deep_learning_baseline.ipynb

## regularized architecture
Linear -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Output

## batchnorm
normalizes features across the batch using learnable gamma and beta
reduces internal covariate shift so training is more stable

## dropout
p=0.3 means 30% neurons randomly turned off each forward pass
retained activations scaled by 1/(1-p) = 1/0.7 during training
at eval time dropout is disabled
acts like an ensemble of smaller networks

## model.train() vs model.eval()
model.train() -> dropout active, batchnorm uses batch stats
model.eval()  -> dropout off, batchnorm uses running averages
