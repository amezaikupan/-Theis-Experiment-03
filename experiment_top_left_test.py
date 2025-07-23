import numpy as np
import pandas as pd
import argparse
import pickle
import os
from data import *
from utils import *
from plotting import *
from method import *
import gc

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default=f"Experiment_01_top_left")
parser.add_argument("--n_task", default=10)
parser.add_argument("--merge_dica", default=0)
parser.add_argument("--n", default=4000)
parser.add_argument("--p", default=12)
parser.add_argument("--p_s", default=6)
parser.add_argument("--p_conf", default=1)
parser.add_argument("--eps", default=2)
parser.add_argument("--g", default=1)
parser.add_argument("--lambd", default=0.5)
parser.add_argument("--lambd_test", default=0.7)
parser.add_argument("--use_hsic", default=0)
parser.add_argument("--alpha_test", default=0.05)
parser.add_argument("--n_repeat", default=20)
parser.add_argument("--max_l", default=100)
parser.add_argument("--n_ul", default=100)
args = parser.parse_args()

save_dir = args.save_dir
if not os.path.exists(save_dir):
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
n_repeat = int(args.n_repeat)

# Load dataset
dataset = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
n_ex = dataset.train["n_ex"]

x_test = dataset.train["x_test"].astype(np.float32)
y_test = dataset.train["y_test"].astype(np.float32)

n_train_tasks = np.arange(p_s + 1)
color_dict, markers, legends = get_color_dict()

pooling = Pooling()
mean = Mean()
sgreedy = SGreedy().set_params({'alpha': 0.001, 'use_hsic': False})
shat = SHat().set_params({'alpha': 0.001, 'use_hsic': False})
causal = Causal().set_params()
methods = [pooling, shat, sgreedy, mean, causal]
methods_name = [method.name for method in methods]

results = {m: np.zeros((n_repeat, len(n_train_tasks))) for m in methods_name}
feature_log = []

true_causal = set(range(p_s))

for rep in range(n_repeat):
    print("**************REP", rep, "*******")
    x_train, y_train = dataset.resample(n_task, n)
    x_test = dataset.test["x_test"]
    y_test = dataset.test["y_test"]

    for index, n_ps in np.ndenumerate(n_train_tasks):
        print("---- Round ", index)
        params = {'n_samples_per_task': n_ex, 'run_metadata': {'rep': rep, 'n_ps': n_ps}}

        for method in methods:
            print(method.name)

            if method.name == 'causal':
                method.fit(x_train, y_train, params)
                results[method.name][rep, index] = method.evaluate(x_test, y_test)
            else:
                x_temp = x_train[:, n_ps:]  # Remove first n_ps features
                y_temp = y_train
                method.fit(x_temp, y_temp, params)
                results[method.name][rep, index] = method.evaluate(x_test[:, n_ps:], y_test)

                if hasattr(method, "selected_features") and method.selected_features is not None:
                    selected = method.selected_features
                    if hasattr(method, "lasso_mask") and method.lasso_mask is not None:
                        selected = np.where(method.lasso_mask)[0][selected]

                    # Convert selected indices back to original feature space
                    selected_original = selected + n_ps
                    selected_set = set(selected_original)
                    
                    # Causal features that remain after truncation (in original indices)
                    remaining_causal_original = set(range(n_ps, p_s))
                    
                    # Find intersection - causal features that were both retained and selected
                    correct = selected_set.intersection(remaining_causal_original)
                    
                    # Calculate metrics
                    total_causal_remaining = len(remaining_causal_original)  # p_s - n_ps
                    causal_selected = len(correct)
                    
                    feature_log.append({
                        "rep": rep,
                        "method": method.name,
                        # "n_train_tasks": n_task,
                        "total_causal_features": p_s,
                        "causal_removed": n_ps,
                        "causal_remaining": total_causal_remaining,
                        "causal_selected": causal_selected,
                        "selection_accuracy": causal_selected / total_causal_remaining if total_causal_remaining > 0 else 0,
                        "selected_features_original": list(selected_original),
                        "selected_features_truncated": list(selected)
                    })

    del x_train, y_train, x_test, y_test
    gc.collect()

# Save results
save_all = {
    "results": results,
    "plotting": [methods_name, color_dict, legends, markers],
    "n_train_tasks": n_train_tasks
}

file_name = ["tl_norm", str(n_repeat), str(eps), str(g), str(lambd)]
file_name = f"lambd_test_{lambd}_lambd_train_{lambd_test}_".join(file_name)

with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
    pickle.dump(save_all, f)

# Save selected feature log
df_selected = pd.DataFrame(feature_log)
if not df_selected.empty:
    # Convert list columns to string for CSV storage
    df_selected["selected_features_original"] = df_selected["selected_features_original"].apply(lambda x: ",".join(map(str, x)))
    df_selected["selected_features_truncated"] = df_selected["selected_features_truncated"].apply(lambda x: ",".join(map(str, x)))
    df_selected.to_csv(os.path.join(save_dir, file_name + "_features.csv"), index=False)

# Plot
plot_tl(os.path.join(save_dir, file_name + ".pkl"))