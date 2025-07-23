import numpy as np
from sklearn import linear_model
import argparse
import re_subset_search as subset_search
import pickle
import os
from data import *
from utils import *
import gc
from plotting import *
from method import *

np.random.seed(1234)

s_range_pcv = 0.001
e_range_pcv = 0.999

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir", default=f"Experiment_01_top_left"
)
parser.add_argument("--n_task", default=10)
parser.add_argument("--merge_dica", default=0)
parser.add_argument("--n", default=4000)
parser.add_argument("--p", default=12)
parser.add_argument("--p_s", default=6)
parser.add_argument("--p_conf", default=1)
parser.add_argument("--eps", default=2)
parser.add_argument("--g", default=1)
parser.add_argument("--lambd", default=0.5)
parser.add_argument("--lambd_test", default=0.99)
parser.add_argument("--use_hsic", default=0)
parser.add_argument("--alpha_test", default=0.05)
parser.add_argument("--n_repeat", default=1)
parser.add_argument("--max_l", default=100)
parser.add_argument("--n_ul", default=100)
args = parser.parse_args()

save_dir = args.save_dir

if not os.path.exists(save_dir) :
    os.makedirs(save_dir)

n_task = int(args.n_task)
n = int(args.n)

p = int(args.p)
p_s = int(args.p_s)
p_conf = int(args.p_conf)
eps = float(args.eps)
g = float(args.g)

lambd = float(args.lambd)
lambd_test = float(args.lambd_test)

alpha_test = float(args.alpha_test)
use_hsic = bool(int(args.use_hsic))

dataset = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
# x_train = dataset.train['x_train'].astype(np.float32)
# y_train = dataset.train['y_train'].astype(np.float32)
n_ex = dataset.train["n_ex"]

# Define test
x_test = dataset.train["x_test"].astype(np.float32)
y_test = dataset.train["y_test"].astype(np.float32)

n_train_tasks = np.arange(2, n_task)
n_repeat = int(args.n_repeat)

true_s = np.arange(p_s)

pooling = Pooling()
pooling_non_lin_1 = Pooling_RF()
pooling_non_lin_2 = Pooling_LGBM()
pooling_poly = Pooling_poly().set_params(degree=2)
mean = Mean()
sgreedy = SGreedy().set_params({'alpha': 0.001, 'use_hsic': False})
shat = SHat().set_params({'alpha': 0.001, 'use_hsic': False})
shat_rf = SHat_RF().set_params({'alpha': 0.001, 'use_hsic': False})
shat_poly = SHat_poly().set_params(degree=2, params={'alpha': 0.001, 'use_hsic': False})
causal = Causal().set_params()
methods = [pooling, shat, sgreedy, mean, causal]

results = {}
methods_name = [method.name for method in methods]

n_train_tasks = np.arange(p_s + 1)
print(n_train_tasks)
color_dict, markers, legends = get_color_dict()

print("n_rep", n_repeat)
print(n_train_tasks.size)
for m in methods_name:
    results[m] = np.zeros((n_repeat, n_train_tasks.size))

for rep in range(n_repeat):
    print("**************REP", rep, "*******")

    x_train, y_train = dataset.resample(n_task, n)

    x_test = dataset.test["x_test"]
    y_test = dataset.test["y_test"]

    for index, n_ps in np.ndenumerate(n_train_tasks):
    # for index, t in np.ndenumerate(n_train_tasks):
        print("---- Round ", index)
        # x_temp = x_train[0 : np.cumsum(n_ex)[t], :]
        # y_temp = y_train[0 : np.cumsum(n_ex)[t], :]

        params = {
            'n_samples_per_task': n_ex,
        }

        for method in methods:
            print(method.name)

            if method.name == 'causal':
                method.fit(x_train, y_train, params)
                results[method.name][rep, index] = method.evaluate(x_test, y_test)
            else:
                x_temp = x_train[:, n_ps:]
                y_temp = y_train
                method.fit(x_temp, y_temp, params)
                results[method.name][rep, index] = method.evaluate(x_test[:, n_ps:], y_test)
        
    del x_train, y_train, x_test, y_test

save_all = {}
save_all["results"] = results

save_all["plotting"] = [methods_name, color_dict, legends, markers]

# print('RANGE', np.arange(p_s + 1))
save_all["n_train_tasks"] = n_train_tasks#np.arange(p_s + 1) 

# Save pickle file
file_name = ["tl_norm_", str(n_repeat), str(eps), str(g), str(lambd)]
file_name = "_".join(file_name)

with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
    pickle.dump(save_all, f)

# Create plot
plot_tl(os.path.join(save_dir, file_name + ".pkl"))