import numpy as np 
import scipy as sp 
from sklearn import linear_model
from scipy.linalg import block_diag
import sklearn.metrics as metrics
import itertools
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from pygam import LinearGAM, s
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor 
from lightgbm import LGBMRegressor


def invariant_residual_test(residual, n_samples_per_task_valid, subset=None, save_hist=False, save_dir="Search_plot_gam"):
    """
    Perform residual invariance test across environments using:
    - Wilcoxon test for mean difference
    - Levene's test for variance difference
    - Bonferroni correction on minimum p-value across all one-vs-rest tests

    Parameters:
    - residual: np.ndarray of residuals (1D)
    - n_samples_per_task_valid: list or array of number of samples per environment
    - subset: tuple of variable indices (optional, for saving histogram file name)
    - save_hist: bool, whether to save residual histograms per environment
    - save_dir: str, folder to save plots

    Returns:
    - corrected_pval: float
    """

    from scipy.stats import ranksums, levene
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    env_labels = np.concatenate([
        np.full(n, i) for i, n in enumerate(n_samples_per_task_valid)
    ])  # shape = (N,)
    
    pvals_all_envs = []

    if save_hist:
        os.makedirs(save_dir, exist_ok=True)


    # Optionally plot histogram
    if save_hist and subset is not None:
        plt.figure(figsize=(6, 4))
        for e in np.unique(env_labels):
            res_env = residual[env_labels == e]
            plt.hist(res_env, bins=30, alpha=0.6, label=f"Env {e}", density=True)
        plt.title(f"Residuals by environment for subset {subset}")
        plt.xlabel("Residual")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        subset_str = "_".join(map(str, subset))
        plt.savefig(f"{save_dir}/residual_hist_subset_{subset_str}.png", dpi=300)
        plt.close()

    # Statistical tests per environment
    for e in np.unique(env_labels):
        idx_env = (env_labels == e)
        idx_rest = (env_labels != e)

        res_env = residual[idx_env]
        res_rest = residual[idx_rest]

        try:
            _, pval_wilcoxon = ranksums(res_env, res_rest)
        except:
            pval_wilcoxon = 1.0

        try:
            _, pval_levene = levene(res_env, res_rest)
        except:
            pval_levene = 1.0

        p_combined = 2 * min(pval_wilcoxon, pval_levene)
        p_combined = min(p_combined, 1.0)
        pvals_all_envs.append(p_combined)

    # Bonferroni correction
    min_pval = min(pvals_all_envs)
    corrected_pval = min(1.0, len(np.unique(env_labels)) * min_pval)

    return corrected_pval

def log_loss(labels, pred, eps=1e-15):
    pred_clipped = np.clip(pred, eps, 1 - eps)
    labels_full = np.column_stack([1 - labels, labels])  # [1-p, p] for each sample
    return np.log(-np.sum(labels_full * np.log(pred_clipped), axis=1)[:, np.newaxis])

def mse(true_vals, pred):
    return np.mean((true_vals - pred)**2) 

def levene(residual, n_samples_per_task):
 
    residual = residual.ravel()

    num_tasks = len(n_samples_per_task)
    task_boundary = np.concatenate(([0], np.cumsum(n_samples_per_task)))
    # residual_boundary = [residual[task_boundary[i]:task_boundary[i+1], :] for i in range(num_tasks)]
    residual_boundary = [
        residual[task_boundary[i]:task_boundary[i+1]]
        for i in range(num_tasks)
    ]

    stat, pval = sp.stats.levene(*residual_boundary)
    return stat, pval

def hsic(residual, n_samples_per_task):
    def get_kernel_matrix(X, sX):
        kernel = (X[:, :, np.newaxis] - X.T).T
        kernel = np.exp(-1.0 / (2 * sX) * np.linalg.norm(kernel, axis=1))
        return kernel

    def get_sX(X, y):
        k = X[:, :, np.newaxis] - y.T 
        ls = np.linalg.norm(k, axis=1)
        sX = 0.5 * np.median(ls.flatten())
        return sX

    def get_task_boundray_matrix(n_samples_per_task):
        return block_diag(*[np.ones((n, n)) for n in n_samples_per_task])

    valid_dom = get_task_boundray_matrix(n_samples_per_task)
    sX = get_sX(residual, residual)
    #------- 
    X = residual
    y = valid_dom

    n = X.T.shape[1]
    kernel_X = get_kernel_matrix(X, sX)
    kernel_y = valid_dom
    coef = 1.0 / n

     # The formula can be founded there https://proceedings.neurips.cc/paper_files/paper/2007/file/d5cfead94f5350c12c322b5b664544c1-Paper.pdf
    HSIC = (
        (coef**2) * np.sum(kernel_X * kernel_y)
        + coef**4 * np.sum(kernel_X) * np.sum(kernel_y)
        - 2 * coef**3 * np.sum(np.sum(kernel_X, axis=1) * np.sum(kernel_y, axis=1))
    )

    # Get sums of Kernels
    KXsum = np.sum(kernel_X)
    KYsum = np.sum(kernel_y)

    # Get stats for gamma approx
    xMu = 1.0 / (n * (n - 1)) * (KXsum - n)
    yMu = 1.0 / (n * (n - 1)) * (KYsum - n)
    V1 = (
        coef**2 * np.sum(kernel_X * kernel_X)
        + coef**4 * KXsum**2
        - 2 * coef**3 * np.sum(np.sum(kernel_X, axis=1) ** 2)
    )
    V2 = (
        coef**2 * np.sum(kernel_y * kernel_y)
        + coef**4 * KYsum**2
        - 2 * coef**3 * np.sum(np.sum(kernel_y, axis=1) ** 2)
    )

    meanH0 = (1.0 + xMu * yMu - xMu - yMu) / n
    varH0 = 2.0 * (n - 4) * (n - 5) / (n * (n - 1.0) * (n - 2.0) * (n - 3.0)) * V1 * V2

    # Parameters of the Gamma
    a = meanH0**2 / varH0
    b = n * varH0 / meanH0

    pval = 1.0 - sp.stats.gamma.cdf(n * HSIC, a, scale=b)
    return pval

# Split train and validation sets
def split_train_valid(x, y, n_samples_per_task_list, valid_split=0.4):

    task_boundaries = np.append(0, np.cumsum(n_samples_per_task_list).astype(int))

    n_samples_per_task_train_set, n_samples_per_task_valid_set = [], []
    train_x, train_y, valid_x, valid_y = [], [], [], []

    for i in range(len(n_samples_per_task_list)):  # Means for each task
        start_idx, end_idx = task_boundaries[i], task_boundaries[i + 1]
        x_task = x[start_idx:end_idx]
        y_task = y[start_idx:end_idx]

        indices = np.arange(len(x_task))
        # np.random.shuffle(indices)
        x_task = x_task[indices]
        y_task = y_task[indices]

        n_train = int((1 - valid_split) * len(x_task))

        train_x.append(x_task[:n_train])
        train_y.append(y_task[:n_train])

        valid_x.append(x_task[n_train:])
        valid_y.append(y_task[n_train:])

        n_samples_per_task_train_set.append(n_train)
        n_samples_per_task_valid_set.append(len(x_task) - n_train)

        # n_train_task = int((1 - valid_split) * n_samples_per_task_list[i])

        # train_x.append(x[task_boundaries[i] : task_boundaries[i] + n_train_task])
        # train_y.append(y[task_boundaries[i] : task_boundaries[i] + n_train_task])

        # valid_x.append(x[task_boundaries[i] + n_train_task : task_boundaries[i + 1]])
        # valid_y.append(y[task_boundaries[i] + n_train_task : task_boundaries[i + 1]])

        # n_samples_per_task_train_set.append(n_train_task)
        # n_samples_per_task_valid_set.append(n_samples_per_task_list[i] - n_train_task)

    train_x = np.concatenate(train_x, axis=0)
    valid_x = np.concatenate(valid_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    valid_y = np.concatenate(valid_y, axis=0)

    n_samples_per_task_train_set = np.array(n_samples_per_task_train_set)
    n_samples_per_task_valid_set = np.array(n_samples_per_task_valid_set)

    return (
        train_x,
        train_y,
        valid_x,
        valid_y,
        n_samples_per_task_train_set,
        n_samples_per_task_valid_set,
    )
# ************************* POOLING ***********************
# ********  Full search ****************
def full_search(x,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        x, y, n_samples_per_task, valid_split
    )
    print ('GOO FULL SEARCH')

    # plt.figure(figsize=(6, 3))
    # sns.kdeplot(y, label='Y gốc', fill=True)
    # sns.kdeplot(train_y, label='Train Y', fill=True)
    # sns.kdeplot(valid_y, label='Validation Y', fill=True)
    # plt.title('Phân phối biến đầu ra Y')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # p_train = ks_2samp(y, train_y).pvalue
    # p_valid = ks_2samp(y, valid_y).pvalue
    # print(f"KS p-value (Y gốc vs Train Y) = {p_train}")
    # print(f"KS p-value (Y gốc vs Validation Y) = {p_valid}")

    # for i in range(train_x.shape[1]):
    #     # plt.figure(figsize=(6, 3))
    #     # sns.kdeplot(x[:, i], label='X gốc', fill=True)
    #     # sns.kdeplot(train_x[:, i], label='Train', fill=True)
    #     # sns.kdeplot(valid_x[:, i], label='Validation', fill=True)
    #     # plt.title(f'Phân phối đặc trưng {i}')
    #     # plt.legend()
    #     # plt.tight_layout()
    #     # plt.show()
        
    #     p_train = ks_2samp(x[:, i], train_x[:, i]).pvalue
    #     p_valid = ks_2samp(x[:, i], valid_x[:, i]).pvalue
    #     print(f"Feature {i}: KS p-value (X vs Train) = {p_train:.4f}, (X vs Valid) = {p_valid:.4f}")

    # Search 
    all_sets = []
    all_pvals = []
    all_losses = []
    best_subset = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    best_loss = 1e12
    rang = np.arange(train_x.shape[1])

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):
            currentIndex = rang[np.array(s)]

            regr = linear_model.LinearRegression()
            regr.fit(train_x[:, currentIndex], train_y.flatten())
            pred = regr.predict(valid_x[:, currentIndex])[:, np.newaxis]

            residual = valid_y - pred

            if use_hsic:
                pval = hsic(residual, n_samples_per_task_valid)
            else:
                stat, pval = levene(residual, n_samples_per_task_valid)

            all_sets.append(s)
            all_pvals.append(pval)

            current_loss = np.mean((valid_y -  pred)**2)
            all_losses.append(current_loss)

            if pval > alpha:
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_subset = s 
                    accepted_sets.append(s)
                    accepted_loss.append(current_loss)
                    accepted_pvals.append(pval)


    all_pvals = np.array(all_pvals).flatten()
    if len(accepted_sets) == 0:
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:

            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            all_losses = np.array(all_losses).flatten()
            tied_losses = all_losses[tied_indices]

            idx_max = tied_indices[np.argmin(tied_losses)]
            
        best_subset = all_sets[idx_max]
        best_loss = all_losses[idx_max]
        accepted_sets.append(best_subset)

    # === Write full log to file ===
    import os

    os.makedirs('search_log', exist_ok=True)
    
    with open(f"search_log/shat_n_features_{train_x.shape[1]}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            f.write(f"Subset: {s}, P-Value: {p:.23f}, Loss: {l:.23f}\n")

        f.write("\nAccepted Subsets:\n")
        for s, l in zip(accepted_sets, accepted_loss):
            f.write(f"Subset: {s}, Loss: {l:.23f}\n")

        f.write(f"\nBest Subset: {best_subset}, Best Loss: {best_loss:.23f}\n")
        f.write(f"Accepted pvals: {accepted_pvals}\n")
        f.write(f"Accepted sets: {accepted_sets}\n")

    print("SHAT BEST SUBSET", best_subset)
    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)
    
##### ********** GREEDY_SEARCH *************

def greedy_search(X, y, n_samples_per_task, alpha, 
                  valid_split, is_classification_task=False):
    
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        X, y, n_samples_per_task, valid_split
    )

    all_sets = []
    all_pvals = []
    all_losses = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subset = None

    n_iters = 10 * num_predictors
    stay = 1
    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    bins = []

    while stay == 1:
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        loss_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])

            if selected[p] == 0:
                subset_add = np.append(current_subset, p).astype(int)

                regr = linear_model.LinearRegression()
                regr.fit(train_x[:, subset_add], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_add])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_add))
                all_pvals.append(pval)
                all_losses.append(current_loss)

            if selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0
                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue
                
                regr = linear_model.LinearRegression()
                regr.fit(train_x[:, subset_rem], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_rem])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_rem))
                all_pvals.append(pval)
                all_losses.append(current_loss)

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size > 0:
            best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

            selected[np.where(loss_a == best_loss)] = (
                selected[np.where(loss_a == best_loss)] + 1
            ) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            accepted_sets.append(tuple(accepted_subset))
            accepted_loss.append(best_loss)
            # Find the corresponding p-value for the accepted subset
            best_idx = np.where(loss_a == best_loss)[0][0]
            accepted_pvals.append(pvals_a[best_idx])
            
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)
        else:
            best_pval_arg = np.argmin(statistic_a)
            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1

    # Handle case where no accepted subset was found
    if accepted_subset is None:
        print('Number of accepted sets: 0')

        all_pvals = np.array(all_pvals).flatten()
        all_losses = np.array(all_losses).flatten()

        # Find the subset with the highest p-value
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        # Handle ties in p-values (same logic as full_search)
        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]

        accepted_subset = np.sort(all_sets[idx_max])
        best_loss = all_losses[idx_max]
        
        # Add the final selected subset to accepted sets
        accepted_sets.append(tuple(accepted_subset))
        accepted_loss.append(best_loss)
        accepted_pvals.append(all_pvals[idx_max])
    else:
        print('Number of accepted sets: ', len(accepted_sets))
        best_loss = accepted_loss[-1] if accepted_loss else 0

    # === Write full log to file ===
    import os

    os.makedirs('search_log', exist_ok=True)
    with open(f"search_log/greedy_n_features_{train_x.shape[1]}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("Greedy Search Log\n")
        f.write("=================\n\n")
        
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
        
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, P-Value: {p_val:.23f}, Loss: {l_val:.23f}\n")

        f.write("\nAccepted Subsets (during search):\n")
        for s, l, p in zip(accepted_sets, accepted_loss, accepted_pvals):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, Loss: {l_val:.23f}, P-Value: {p_val:.23f}\n")

        f.write(f"\nFinal Best Subset: {tuple(accepted_subset)}, Best Loss: {float(best_loss):.23f}\n")
        f.write(f"Total iterations: {ind}\n")
        f.write(f"Total subsets tested: {len(all_sets)}\n")
        f.write(f"Number of accepted sets: {len(accepted_sets)}\n")
        f.write(f"Alpha threshold: {alpha}\n")
        f.write(f"Classification task: {is_classification_task}\n")

    print("SGREEDY ACCEPTED SET", accepted_subset)
    return np.array(accepted_subset)


# ***************** POLYMONIAL *****************
# Full search 
def full_search_poly(x,y,n_samples_per_task, 
                alpha, valid_split,
                degree,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    # Split data 
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        x, y, n_samples_per_task, valid_split
    )

    print ('GOO POLY FULL SEARCH')

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # Search 
    all_sets = []
    all_pvals = []
    all_losses = []
    best_subset = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    best_loss = 1e12
    rang = np.arange(train_x.shape[1])
    j = 0

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):
            currentIndex = rang[np.array(s)]
        
            regr = make_pipeline(PolynomialFeatures(degree=degree), linear_model.LinearRegression())
            regr.fit(train_x[:, currentIndex], train_y.flatten())

            pred = regr.predict(valid_x[:, currentIndex])[:, np.newaxis]
            residual = valid_y - pred
            pred = pred.ravel()

            if use_hsic:
                pval = hsic(residual, n_samples_per_task_valid)
            else:
                # print('N samples per task valid', n_samples_per_task_valid)
                stat, pval = levene(residual, n_samples_per_task_valid)
                # pval = invariant_residual_test(
                #     residual=residual,
                #     n_samples_per_task_valid=n_samples_per_task_valid,
                #     subset=s,
                #     save_hist=True,
                #     save_dir = 'search_plot_poly'
                # )


            all_sets.append(s)
            all_pvals.append(pval)

            current_loss = np.mean((valid_y -  pred)**2)
            all_losses.append(current_loss)

            if pval > alpha:
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_subset = s 
                    accepted_sets.append(s)
                    accepted_loss.append(current_loss)
                    accepted_pvals.append(pval)

    print('Number of accepted sets: ', len(accepted_sets))

    all_pvals = np.array(all_pvals).flatten()
    if len(accepted_sets) == 0:
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            # Find the ones with the same max p-value
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]

            all_losses = np.array(all_losses).flatten()
            tied_losses = all_losses[tied_indices]

            idx_max = tied_indices[np.argmin(tied_losses)]
            
        best_subset = all_sets[idx_max]
        best_loss = all_losses[idx_max]
        accepted_sets.append(best_subset)

    # === Write full log to file ===
    with open(f"search_log/poly_shat_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            f.write(f"Subset: {s}, P-Value: {p:.23f}, Loss: {l:.23f}\n")

        f.write("\nAccepted Subsets:\n")
        for s, l in zip(accepted_sets, accepted_loss):
            f.write(f"Subset: {s}, Loss: {l:.23f}\n")

        f.write(f"\nBest Subset: {best_subset}, Best Loss: {best_loss:.23f}\n")
        f.write(f"Accepted pvals: {accepted_pvals}\n")
        f.write(f"Accepted sets: {accepted_sets}\n")

    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)
    
# ******* Poly Greedy search ********
def greedy_search_poly(X,y,n_samples_per_task, 
                alpha, valid_split,
                degree,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        X, y, n_samples_per_task, valid_split
    )

    all_sets = []
    all_pvals = []
    all_losses = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subset = None

    n_iters = 10 * num_predictors
    stay = 1
    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    bins = []

    while stay == 1:
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        loss_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])

            if selected[p] == 0:
                subset_add = np.append(current_subset, p).astype(int)

                regr = make_pipeline(PolynomialFeatures(degree=degree), linear_model.LinearRegression())
                regr.fit(train_x[:, subset_add], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_add])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_add))
                all_pvals.append(pval)
                all_losses.append(current_loss)

            if selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0
                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue
                
                regr = make_pipeline(PolynomialFeatures(degree=degree), linear_model.LinearRegression())
                regr.fit(train_x[:, subset_rem], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_rem])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_rem))
                all_pvals.append(pval)
                all_losses.append(current_loss)

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size > 0:
            best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

            selected[np.where(loss_a == best_loss)] = (
                selected[np.where(loss_a == best_loss)] + 1
            ) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            
            accepted_sets.append(tuple(accepted_subset))
            accepted_loss.append(best_loss)

            best_idx = np.where(loss_a == best_loss)[0][0]
            accepted_pvals.append(pvals_a[best_idx])
            
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)
        else:
            best_pval_arg = np.argmin(statistic_a)
            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1

    # Handle case where no accepted subset was found
    if accepted_subset is None:
        print('Number of accepted sets: 0')

        all_pvals = np.array(all_pvals).flatten()
        all_losses = np.array(all_losses).flatten()

        # Find the subset with the highest p-value
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        # Handle ties in p-values (same logic as full_search)
        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]

        accepted_subset = np.sort(all_sets[idx_max])
        best_loss = all_losses[idx_max]
        
        # Add the final selected subset to accepted sets
        accepted_sets.append(tuple(accepted_subset))
        accepted_loss.append(best_loss)
        accepted_pvals.append(all_pvals[idx_max])
    else:
        print('Number of accepted sets: ', len(accepted_sets))
        best_loss = accepted_loss[-1] if accepted_loss else 0

    # === Write full log to file ===
    with open(f"search_log/poly_greedy_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("Poly Greedy Search Log\n")
        f.write("=================\n\n")
        
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, P-Value: {p_val:.23f}, Loss: {l_val:.23f}\n")

        f.write("\nAccepted Subsets (during search):\n")
        for s, l, p in zip(accepted_sets, accepted_loss, accepted_pvals):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, Loss: {l_val:.23f}, P-Value: {p_val:.23f}\n")

        f.write(f"\nFinal Best Subset: {tuple(accepted_subset)}, Best Loss: {float(best_loss):.23f}\n")
        f.write(f"Total iterations: {ind}\n")
        f.write(f"Total subsets tested: {len(all_sets)}\n")
        f.write(f"Number of accepted sets: {len(accepted_sets)}\n")
        f.write(f"Alpha threshold: {alpha}\n")
        f.write(f"Classification task: {is_classification_task}\n")

    return np.array(accepted_subset)

# ********* RANDOMFOREST ****************
# *** shat ** 
def full_search_rf(x,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):

    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        x, y, n_samples_per_task, valid_split
    )
    from sklearn.ensemble import RandomForestRegressor 

    # Search 
    all_sets = []
    all_pvals = []
    all_losses = []
    best_subset = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    best_loss = 1e12

    rang = np.arange(train_x.shape[1])

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):
            currentIndex = rang[np.array(s)]

            regr = RandomForestRegressor()
            regr.fit(train_x[:, currentIndex], train_y.flatten())
    
            pred = regr.predict(valid_x[:, currentIndex])[:, np.newaxis]
            residual = valid_y - pred

            if use_hsic:
                pval = hsic(residual, n_samples_per_task_valid)
            else:
                stat, pval = levene(residual, n_samples_per_task_valid)
                # pval = invariant_residual_test(
                #     residual=residual,
                #     n_samples_per_task_valid=n_samples_per_task_valid,
                #     subset=s,
                #     save_hist=True,
                #     save_dir = 'search_plot_rf'
                # )

            all_sets.append(s)
            all_pvals.append(pval)

            current_loss = np.mean((valid_y -  pred)**2)
            all_losses.append(current_loss)

            if pval > alpha:
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_subset = s 
                    accepted_sets.append(s)
                    accepted_loss.append(current_loss)
                    accepted_pvals.append(pval)
    print('Number of accepted sets: ', len(accepted_sets))

    all_pvals = np.array(all_pvals).flatten()
    if len(accepted_sets) == 0:
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            all_losses = np.array(all_losses).flatten()
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]
            
        best_subset = all_sets[idx_max]
        best_loss = all_losses[idx_max]
        accepted_sets.append(best_subset)

    # === Write full log to file ===
    with open(f"search_log/random_forest_shat_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            f.write(f"Subset: {s}, P-Value: {p:.23f}, Loss: {l:.23f}\n")

        f.write("\nAccepted Subsets:\n")
        for s, l in zip(accepted_sets, accepted_loss):
            f.write(f"Subset: {s}, Loss: {l:.23f}\n")

        f.write(f"\nBest Subset: {best_subset}, Best Loss: {best_loss:.23f}\n")
        f.write(f"Accepted pvals: {accepted_pvals}\n")
        f.write(f"Accepted sets: {accepted_sets}\n")

    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)
    
# ******* Ramdom forest Greedy search ********
def greedy_search_rf(X,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    
    # Split data
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        X, y, n_samples_per_task, valid_split
    )
    all_sets = []
    all_pvals = []
    all_losses = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subset = None

    n_iters = 10 * num_predictors
    stay = 1
    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    bins = []

    while stay == 1:
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        loss_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])

            if selected[p] == 0:
                subset_add = np.append(current_subset, p).astype(int)

                regr = RandomForestRegressor()
                regr.fit(train_x[:, subset_add], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_add])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_add))
                all_pvals.append(pval)
                all_losses.append(current_loss)

            if selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0
                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue
                
                regr = RandomForestRegressor()
                regr.fit(train_x[:, subset_rem], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_rem])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_rem))
                all_pvals.append(pval)
                all_losses.append(current_loss)

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size > 0:
            best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

            selected[np.where(loss_a == best_loss)] = (
                selected[np.where(loss_a == best_loss)] + 1 ) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            
            accepted_sets.append(tuple(accepted_subset))
            accepted_loss.append(best_loss)

            best_idx = np.where(loss_a == best_loss)[0][0]
            accepted_pvals.append(pvals_a[best_idx])
            
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)
        else:
            best_pval_arg = np.argmin(statistic_a)
            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1

    if accepted_subset is None:
        print('Number of accepted sets: 0')

        all_pvals = np.array(all_pvals).flatten()
        all_losses = np.array(all_losses).flatten()
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]

        accepted_subset = np.sort(all_sets[idx_max])
        best_loss = all_losses[idx_max]
        
        # Add the final selected subset to accepted sets
        accepted_sets.append(tuple(accepted_subset))
        accepted_loss.append(best_loss)
        accepted_pvals.append(all_pvals[idx_max])
    else:
        print('Number of accepted sets: ', len(accepted_sets))
        best_loss = accepted_loss[-1] if accepted_loss else 0

    # === Write full log to file ===
    with open(f"search_log/rf_greedy_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("Ramdom forest Greedy Search Log\n")
        f.write("=================\n\n")
        
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, P-Value: {p_val:.23f}, Loss: {l_val:.23f}\n")

        f.write("\nAccepted Subsets (during search):\n")
        for s, l, p in zip(accepted_sets, accepted_loss, accepted_pvals):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, Loss: {l_val:.23f}, P-Value: {p_val:.23f}\n")

        f.write(f"\nFinal Best Subset: {tuple(accepted_subset)}, Best Loss: {float(best_loss):.23f}\n")
        f.write(f"Total iterations: {ind}\n")
        f.write(f"Total subsets tested: {len(all_sets)}\n")
        f.write(f"Number of accepted sets: {len(accepted_sets)}\n")
        f.write(f"Alpha threshold: {alpha}\n")
        f.write(f"Classification task: {is_classification_task}\n")

    return np.array(accepted_subset)
    
# ************* LGBM ***************
def full_search_lgbm(x,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    # Split data 
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        x, y, n_samples_per_task, valid_split
    )
    # Search 
    all_sets = []
    all_pvals = []
    all_losses = []
    best_subset = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    best_loss = 1e12

    rang = np.arange(train_x.shape[1])    
    j = 0

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):

            currentIndex = rang[np.array(s)]

            regr = LGBMRegressor(
                    objective='regression', 
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31
                )
            regr.fit(train_x[:, currentIndex], train_y.flatten())
            # print('subset test', s)
            pred = regr.predict(valid_x[:, currentIndex])[:, np.newaxis]
            residual = valid_y - pred

            if use_hsic:
                pval = hsic(residual, n_samples_per_task_valid)
            else:
                stat, pval = levene(residual, n_samples_per_task_valid)
                # pval = invariant_residual_test(
                #     residual=residual,
                #     n_samples_per_task_valid=n_samples_per_task_valid,
                #     subset=s,
                #     save_hist=True,
                #     save_dir = 'search_plot_lgbm'
                # )

            all_sets.append(s)
            all_pvals.append(pval)

            current_loss = np.mean((valid_y -  pred)**2)
            all_losses.append(current_loss)

            if pval > alpha:
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_subset = s 
                    accepted_sets.append(s)
                    accepted_loss.append(current_loss)
                    accepted_pvals.append(pval)
    print('Number of accepted sets: ', len(accepted_sets))

    all_pvals = np.array(all_pvals).flatten()
    if len(accepted_sets) == 0:
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            all_losses = np.array(all_losses).flatten()
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]
            
        best_subset = all_sets[idx_max]
        best_loss = all_losses[idx_max]
        accepted_sets.append(best_subset)

    # === Write full log to file ===
    with open(f"search_log/lgbm_shat_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            f.write(f"Subset: {s}, P-Value: {p:.23f}, Loss: {l:.23f}\n")

        f.write("\nAccepted Subsets:\n")
        for s, l in zip(accepted_sets, accepted_loss):
            f.write(f"Subset: {s}, Loss: {l:.23f}\n")

        f.write(f"\nBest Subset: {best_subset}, Best Loss: {best_loss:.23f}\n")
        f.write(f"Accepted pvals: {accepted_pvals}\n")
        f.write(f"Accepted sets: {accepted_sets}\n")

    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)

def greedy_search_lgbm(X,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):

    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        X, y, n_samples_per_task, valid_split
    )

    all_sets = []
    all_pvals = []
    all_losses = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subset = None

    n_iters = 10 * num_predictors
    stay = 1
    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    bins = []

    while stay == 1:
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        loss_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])

            if selected[p] == 0:
                subset_add = np.append(current_subset, p).astype(int)

                regr = LGBMRegressor(
                    objective='regression', 
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31
                )
                regr.fit(train_x[:, subset_add], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_add])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_add))
                all_pvals.append(pval)
                all_losses.append(current_loss)

            if selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0
                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue
                
                regr = linear_model.LinearRegression()
                regr.fit(train_x[:, subset_rem], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_rem])[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_rem))
                all_pvals.append(pval)
                all_losses.append(current_loss)

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size > 0:
            best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

            selected[np.where(loss_a == best_loss)] = (
                selected[np.where(loss_a == best_loss)] + 1) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            
            # Track accepted sets
            accepted_sets.append(tuple(accepted_subset))
            accepted_loss.append(best_loss)
            # Find the corresponding p-value for the accepted subset
            best_idx = np.where(loss_a == best_loss)[0][0]
            accepted_pvals.append(pvals_a[best_idx])
            
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)
        else:
            best_pval_arg = np.argmin(statistic_a)
            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1

    # Handle case where no accepted subset was found
    if accepted_subset is None:
        print('Number of accepted sets: 0')

        all_pvals = np.array(all_pvals).flatten()
        all_losses = np.array(all_losses).flatten()

        # Find the subset with the highest p-value
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        # Handle ties in p-values (same logic as full_search)
        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]

        accepted_subset = np.sort(all_sets[idx_max])
        best_loss = all_losses[idx_max]
        
        # Add the final selected subset to accepted sets
        accepted_sets.append(tuple(accepted_subset))
        accepted_loss.append(best_loss)
        accepted_pvals.append(all_pvals[idx_max])
    else:
        print('Number of accepted sets: ', len(accepted_sets))
        best_loss = accepted_loss[-1] if accepted_loss else 0

    # === Write full log to file ===
    with open(f"search_log/lgbm_greedy_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("Greedy Search Log\n")
        f.write("=================\n\n")
        
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, P-Value: {p_val:.23f}, Loss: {l_val:.23f}\n")

        f.write("\nAccepted Subsets (during search):\n")
        for s, l, p in zip(accepted_sets, accepted_loss, accepted_pvals):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, Loss: {l_val:.23f}, P-Value: {p_val:.23f}\n")

        f.write(f"\nFinal Best Subset: {tuple(accepted_subset)}, Best Loss: {float(best_loss):.23f}\n")
        f.write(f"Total iterations: {ind}\n")
        f.write(f"Total subsets tested: {len(all_sets)}\n")
        f.write(f"Number of accepted sets: {len(accepted_sets)}\n")
        f.write(f"Alpha threshold: {alpha}\n")
        f.write(f"Classification task: {is_classification_task}\n")

    return np.array(accepted_subset)

# ******** GAM **********
from scipy.stats import wilcoxon


def wilcoxon_one_vs_all(residuals, n_samples_per_task):

    residuals = residuals.flatten()
    splits = np.split(residuals, np.cumsum(n_samples_per_task)[:-1])
    pvals = []

    for i, group in enumerate(splits):
        rest = np.concatenate([splits[j] for j in range(len(splits)) if j != i])
        try:
            stat, p = wilcoxon(group, rest[:len(group)])  # truncate if necessary
        except ValueError:
            p = 1.0  # conservative fallback
        pvals.append(p)

    min_p = min(pvals)
    corrected_p = min(1.0, len(pvals) * min_p)  # Bonferroni correction
    return corrected_p


import numpy as np
import itertools

def full_search_gam(x,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    # Split data 
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        x, y, n_samples_per_task, valid_split
    )
    from pygam import s, LinearGAM
    from scipy.stats import ranksums, levene

    print ('GOOOO FULL SEARCH GAM')

    all_sets = []
    all_pvals = []
    all_losses = []
    best_subset = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    best_loss = 1e12

    rang = np.arange(train_x.shape[1])

    for i in range(1, rang.size + 1):
        for subset in itertools.combinations(rang, i):

            print(subset)

            currentIndex = np.array(subset)
            train_subset = train_x[:, currentIndex]
            valid_subset = valid_x[:, currentIndex]

            terms = sum([s(j) for j in range(len(currentIndex))], start=s(0))

            gam = LinearGAM(terms)
          

            gam.fit(train_subset, train_y.flatten())
            pred = gam.predict(valid_subset)

            residual = valid_y.ravel() - pred.ravel()

            _, pval_levene = levene(residual, n_samples_per_task_valid)

            # min_pval = invariant_residual_test(
            #     residual=residual,
            #     n_samples_per_task_valid=n_samples_per_task_valid,
            #     subset=subset,
            #     save_hist=True,
            #     save_dir='search_plot_gam'
            # )

            min_pval= pval_levene

            all_sets.append(subset)
            all_pvals.append(min_pval)

            current_loss = np.mean((valid_y -  pred)**2)
            all_losses.append(current_loss)

            if min_pval > alpha:
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_subset = subset
                    accepted_sets.append(subset)
                    accepted_loss.append(current_loss)
                    accepted_pvals.append(min_pval)


    print('Number of accepted sets: ', len(accepted_sets))

    all_pvals = np.array(all_pvals).flatten()
    if len(accepted_sets) == 0:
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:

            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            all_losses = np.array(all_losses).flatten()
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]
            
        best_subset = all_sets[idx_max]
        best_loss = all_losses[idx_max]
        accepted_sets.append(best_subset)

    # === Write full log to file ===
    with open(f"search_log/GAM_shat_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("All Tested Subsets:\n")
        for subset, p, l in zip(all_sets, all_pvals, all_losses):
            f.write(f"Subset: {subset}, P-Value: {p:.23f}, Loss: {l:.23f}\n")

        f.write("\nAccepted Subsets:\n")
        for subset, l in zip(accepted_sets, accepted_loss):
            f.write(f"Subset: {subset}, Loss: {l:.23f}\n")

        f.write(f"\nBest Subset: {best_subset}, Best Loss: {best_loss:.23f}\n")
        f.write(f"Accepted pvals: {accepted_pvals}\n")
        f.write(f"Accepted sets: {accepted_sets}\n")

    if return_n_best:
        return [np.array(subset) for subset in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)
    
   
# ******* GAM Greedy search ********
def greedy_search_gam(x,y,n_samples_per_task, 
                alpha, valid_split,
                use_hsic=False,
                return_n_best=None,
                is_classification_task = False):
    
    print('GO GAM GREADY SEARCH')
    
    # Split data
    train_x, train_y, valid_x, valid_y, n_samples_per_task_train, n_samples_per_task_valid = split_train_valid(
        x, y, n_samples_per_task, valid_split
    )
    from pygam import s, LinearGAM
    from scipy.stats import ranksums
    all_sets = []
    all_pvals = []
    all_losses = []
    accepted_sets = []
    accepted_loss = []
    accepted_pvals = []

    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subset = None

    n_iters = 10 * num_predictors
    stay = 1
    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    bins = []

    while stay == 1:
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        loss_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])

            if selected[p] == 0:
                subset_add = np.append(current_subset, p).astype(int)

                currentIndex = np.array(subset_add)
                train_subset = train_x[:, subset_add]
                valid_subset = valid_x[:, subset_add]

                terms = sum([s(j) for j in range(len(currentIndex))], start=s(0))

                gam = LinearGAM(terms)

                gam.fit(train_subset, train_y.flatten())
                pred = gam.predict(valid_subset)[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_add))
                all_pvals.append(pval)
                all_losses.append(current_loss)

            if selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0
                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue

                currentIndex = np.array(subset_rem)
                train_subset = train_x[:, subset_rem]
                valid_subset = valid_x[:, subset_rem]

                terms = sum([s(j) for j in range(len(currentIndex))], start=s(0))

                gam = LinearGAM(terms)
                gam.fit(train_subset, train_y.flatten())
                pred = gam.predict(valid_subset)[:, np.newaxis]
                residual = valid_y - pred

                current_loss = np.mean((valid_y - pred)**2)

                stat, pval = levene(residual, n_samples_per_task_valid)

                pvals_a[p] = pval
                statistic_a[p] = stat
                loss_a[p] = current_loss

                all_sets.append(tuple(subset_rem))
                all_pvals.append(pval)
                all_losses.append(current_loss)

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size > 0:
            best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

            selected[np.where(loss_a == best_loss)] = (
                selected[np.where(loss_a == best_loss)] + 1 ) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            
            accepted_sets.append(tuple(accepted_subset))
            accepted_loss.append(best_loss)

            best_idx = np.where(loss_a == best_loss)[0][0]
            accepted_pvals.append(pvals_a[best_idx])
            
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)
        else:
            best_pval_arg = np.argmin(statistic_a)
            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1

    if accepted_subset is None:
        print('Number of accepted sets: 0')

        all_pvals = np.array(all_pvals).flatten()
        all_losses = np.array(all_losses).flatten()
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]
            tied_losses = all_losses[tied_indices]
            idx_max = tied_indices[np.argmin(tied_losses)]

        accepted_subset = np.sort(all_sets[idx_max])
        best_loss = all_losses[idx_max]
        
        # Add the final selected subset to accepted sets
        accepted_sets.append(tuple(accepted_subset))
        accepted_loss.append(best_loss)
        accepted_pvals.append(all_pvals[idx_max])
    else:
        print('Number of accepted sets: ', len(accepted_sets))
        best_loss = accepted_loss[-1] if accepted_loss else 0

    # === Write full log to file ===
    with open(f"search_log/gam_greedy_tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
        f.write("GAM Greedy Search Log\n")
        f.write("=================\n\n")
        
        f.write("All Tested Subsets:\n")
        for s, p, l in zip(all_sets, all_pvals, all_losses):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, P-Value: {p_val:.23f}, Loss: {l_val:.23f}\n")

        f.write("\nAccepted Subsets (during search):\n")
        for s, l, p in zip(accepted_sets, accepted_loss, accepted_pvals):
            # Convert to scalar values to avoid numpy array formatting issues
            p_val = float(np.asarray(p).item()) if hasattr(p, 'item') else float(p)
            l_val = float(np.asarray(l).item()) if hasattr(l, 'item') else float(l)
            f.write(f"Subset: {s}, Loss: {l_val:.23f}, P-Value: {p_val:.23f}\n")

        f.write(f"\nFinal Best Subset: {tuple(accepted_subset)}, Best Loss: {float(best_loss):.23f}\n")
        f.write(f"Total iterations: {ind}\n")
        f.write(f"Total subsets tested: {len(all_sets)}\n")
        f.write(f"Number of accepted sets: {len(accepted_sets)}\n")
        f.write(f"Alpha threshold: {alpha}\n")
        f.write(f"Classification task: {is_classification_task}\n")

    return np.array(accepted_subset)
    

