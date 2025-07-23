# Hypothesis 01: 
# If features are independent, then we can check individually if the features is causal parent to target. 
# Even in non-linear regime. 

from data import gauss_tl
import numpy as np 
import pandas as pd 
import method


# 01. Linear regime 
# Generate data 
n_tasks = 10

# Test if can find features 1 as causal 
dataset = gauss_tl(
    n_task=n_tasks,      # 3 tasks
    n=50,          # 50 samples per task
    p=10,          # 10 total features
    p_s=5,         # 5 signal features
    p_conf=2,      # 2 confounding features
    eps=0.1,       # noise level
    g=0.5,         # noise feature strength
    lambd=0.3,     # training task similarity
    lambd_test=0.4 # test task similarity
)

# Access the data
x_train = dataset.train['x_train'][:, 0]  # Training features
y_train = dataset.train['y_train']  # Training labels
x_test = dataset.train['x_test'][:, 0]    # Test features
y_test = dataset.train['y_test']    # Test labels

n_ex = dataset.train['n_ex']
n_train_tasks = np.arange(2, n_tasks)

shat = method.SHat().set_params({'alpha': 0.001, 'use_hsic': False})
params = {
        'n_samples_per_task': tuple(n_ex[task] for task in range(n_tasks)),
    }
shat.fit(x_train, y_train, params)
shat.evaluate(x_test, y_test)
