from sklearn.linear_model import LinearRegression
from method import SHat  # assuming you put SHat in methods/shat.py
import numpy as np
from data import gauss_tl

# ==== 1. Generate data ====
dataset = gauss_tl(
    n_task=5, n=200, p=15, p_s=5, p_conf=2,
    eps=0.1, g=0.5, lambd=0.3, lambd_test=0.9,
    nonlinear='linear'
)

X_train_all = []
y_train_all = []

# ==== 2. Combine data from all domains ====
for k in range(dataset.n_task):
    domain_data = dataset.train['domains'][f'domain_{k}']
    X_train_all.append(domain_data['X'])
    y_train_all.append(domain_data['y'])

X_train_all = np.vstack(X_train_all)
y_train_all = np.vstack(y_train_all).ravel()[:, np.newaxis]

print(X_train_all.shape)
print(y_train_all.shape)

# ==== 3. Fit SHat ====
shat = SHat()
shat.set_params({'alpha': 0.001, 'use_hsic': False})
shat.fit(X_train_all, y_train_all, params={
    'n_samples_per_task': [dataset.n] * dataset.n_task
})

# ==== 4. Evaluate ====
print("Selected Features:", shat.selected_features)

# Optional: Compare against true causal indices (first p_s features)
p_s = dataset.p_s
print("True causal features: indices 0 to", p_s - 1)

# ==== 5. Plot feature importances ====
import matplotlib.pyplot as plt

if shat.selected_features is not None:
    mask = np.zeros(dataset.p, dtype=bool)
    mask[shat.selected_features] = True
    plt.bar(range(dataset.p), mask, label="Selected by SHat")
    plt.axvline(p_s - 0.5, color='red', linestyle='--', label='End of causal')
    plt.title("SHat selected features vs. true causal range")
    plt.xlabel("Feature Index")
    plt.ylabel("Selected (1) or not (0)")
    plt.legend()
    plt.show()
