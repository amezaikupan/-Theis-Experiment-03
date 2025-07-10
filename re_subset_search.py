import numpy as np 
import scipy as sp 
from sklearn import linear_model
from scipy.linalg import block_diag
import sklearn.metrics as metrics
import itertools

def log_loss(labels, pred, eps=1e-15):
    pred_clipped = np.clip(pred, eps, 1 - eps)
    
    # Convert single-column labels to full probability distribution
    labels_full = np.column_stack([1 - labels, labels])  # [1-p, p] for each sample
    
    
    # Calculate cross-entropy
    return np.log(-np.sum(labels_full * np.log(pred_clipped), axis=1)[:, np.newaxis])

def mse(true_vals, pred):
    return np.mean((true_vals - pred)**2) 

# Define independent tests
# Rewrite: they have the same parameter  
def levene(residual, n_samples_per_task):
    num_tasks = len(n_samples_per_task)
    task_boundary = np.concatenate(([0], np.cumsum(n_samples_per_task)))
    residual_boundary = [residual[task_boundary[i]:task_boundary[i+1], :] for i in range(num_tasks)]
    stat, pval = sp.stats.levene(*residual_boundary)
    # print("Stat, pval", stat, pval)
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
    
    # Optimized 
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
        n_train_task = int((1 - valid_split) * n_samples_per_task_list[i])

        train_x.append(x[task_boundaries[i] : task_boundaries[i] + n_train_task])
        train_y.append(y[task_boundaries[i] : task_boundaries[i] + n_train_task])

        valid_x.append(x[task_boundaries[i] + n_train_task : task_boundaries[i + 1]])
        valid_y.append(y[task_boundaries[i] + n_train_task : task_boundaries[i + 1]])

        n_samples_per_task_train_set.append(n_train_task)
        n_samples_per_task_valid_set.append(n_samples_per_task_list[i] - n_train_task)

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

# Full search 
def full_search(x,y,n_samples_per_task, 
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

    if(is_classification_task):
        for i in range(1, rang.size + 1):
            for s in itertools.combinations(rang, i):
                currentIndex = rang[np.array(s)]

                clf = linear_model.LogisticRegression(random_state=42)
                clf.fit(train_x[:, currentIndex], train_y.flatten())
                pred = clf.predict_proba(valid_x[:, currentIndex])
                residual = log_loss(valid_y, pred)

                # Compute domain boundaries

                if use_hsic:
                    pval = hsic(residual, n_samples_per_task_valid)
                else:
                    stat, pval = levene(residual, n_samples_per_task_valid)

                all_sets.append(s)
                all_pvals.append(pval)

                current_loss = metrics.log_loss(valid_y, pred)
                all_losses.append(current_loss)
                
                if pval > alpha:
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_subset = s 
                        accepted_sets.append(s)
                        accepted_loss.append(current_loss)
    else:
        j = 0

        for i in range(1, rang.size + 1):
            for s in itertools.combinations(rang, i):
                currentIndex = rang[np.array(s)]

                regr = linear_model.LinearRegression()
                regr.fit(train_x[:, currentIndex], train_y.flatten())
                mean_x = np.mean(train_x[:, currentIndex], axis=0)
                # print('subset test', s)
                pred = regr.predict(valid_x[:, currentIndex])[:, np.newaxis]
                mean_pred = np.mean(pred, axis=0)
                residual = valid_y - pred

                # if j < 3:
                #     import matplotlib.pyplot as plt
                #     # Plot histograms
                #     task_boundaries = np.append(0, np.cumsum(n_samples_per_task_valid))
                #     plt.figure(figsize=(12, 6))
                #     for i in range(len(n_samples_per_task_valid)):
                #         domain_residual = residual[task_boundaries[i]:task_boundaries[i+1]]
                #         plt.hist(domain_residual, bins=30, alpha=0.5, edgecolor='black', label=f'Domain {i}')

                #     plt.title('Residual Histogram by Domain' + str(list(currentIndex)) + "Mean_subset: " + str(mean_x) + " Mean_pred: " + str(mean_pred))
                #     plt.xlabel('Residual')
                #     plt.ylabel('Frequency')
                #     plt.legend()
                #     plt.grid(True)
                #     plt.show()

                #     j += 1

                if use_hsic:
                    pval = hsic(residual, n_samples_per_task_valid)
                else:
                    # print('N samples per task valid', n_samples_per_task_valid)
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
    

    print('Number of accepted sets: ', len(accepted_sets))

    all_pvals = np.array(all_pvals).flatten()
    if len(accepted_sets) == 0:
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]

        # print("Max pval", all_pvals[idx_max])
        # print(np.sum(all_pvals == all_pvals[idx_max]))

        if np.sum(all_pvals == all_pvals[idx_max]) > 1:
            # Find the ones with the same max p-value
            max_pval = all_pvals[idx_max]
            tied_indices = np.where(all_pvals == max_pval)[0]

            # Retrieve losses for tied sets
            all_losses = np.array(all_losses).flatten()
            tied_losses = all_losses[tied_indices]
            # print(len(tied_losses))
            # kjjkh
            # Pick the index with lowest loss among tied p-values
            idx_max = tied_indices[np.argmin(tied_losses)]
            
        best_subset = all_sets[idx_max]
        best_loss = all_losses[idx_max]
        accepted_sets.append(best_subset)

      # === Write full log to file ===
    with open(f"search_log/tasks_train_{len(n_samples_per_task_train)}_acc_sets_{len(accepted_sets)}.txt", "w") as f:
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
    
def greedy_search(X, y, n_samples_per_task, alpha, 
                  valid_split, is_classification_task=False):
    
    # Split data
    train_x, train_y, valid_x, valid_y, n_samples_per_task, n_samples_per_task_valid = split_train_valid(
        X, y, n_samples_per_task, valid_split
    )

    # Search 
    all_sets = []
    all_pvals = []

    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subset = None

    n_iters = 10 * num_predictors
    stay = 1
    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    bins = []

    if is_classification_task:
        while stay == 1:
            pvals_a = np.zeros(num_predictors)
            statistic_a = 1e10 * np.ones(num_predictors)
            loss_a = np.zeros(num_predictors)

            for p in range(num_predictors):
                current_subset = np.sort(np.where(selected == 1)[0])

                if selected[p] == 0:
                    subset_add = np.append(current_subset, p).astype(int)

                    clf = linear_model.LogisticRegression(random_state=42)
                    clf.fit(train_x[:, subset_add], train_y.flatten())
                    pred = clf.predict_proba(valid_x[:, subset_add])
                    residual = log_loss(valid_y, pred)

                    current_loss = metrics.log_loss(valid_y, pred)

                    stat, pval = levene(residual, n_samples_per_task_valid)

                    pvals_a[p] = pval
                    statistic_a[p] = stat
                    loss_a[p] = current_loss

                    all_sets.append(subset_add)
                    all_pvals.append(pval)

                if selected[p] == 1:
                    acc_rem = np.copy(selected)
                    acc_rem[p] = 0
                    subset_rem = np.sort(np.where(acc_rem == 1)[0])

                    if subset_rem.size == 0:
                        continue
                    
                    clf = linear_model.LogisticRegression(random_state=42)
                    clf.fit(train_x[:, subset_rem], train_y.flatten())
                    pred = clf.predict(valid_x[:, subset_rem])[:, np.newaxis]
                    residual = log_loss(valid_y, pred)

                    # import matplotlib.pyplot as plt 
                    # plt.hist(residual, bins=30, edgecolor='black', alpha=0.7)
                    # plt.title(str(s))
                    # plt.show()
                    
                    current_loss = metrics.log_loss(valid_y, pred)

                    stat, pval = levene(residual, n_samples_per_task_valid)

                    pvals_a[p] = pval
                    statistic_a[p] = stat
                    loss_a[p] = current_loss

                    all_sets.append(subset_rem)
                    all_pvals.append(pval)

            accepted = np.where(pvals_a > alpha)

            if accepted[0].size > 0:

                best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

                selected[np.where(loss_a == best_loss)] = (
                    selected[np.where(loss_a == best_loss)] + 1
                ) % 2

                accepted_subset = np.sort(np.where(selected == 1)[0])
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
    else:
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

                    all_sets.append(subset_add)
                    all_pvals.append(pval)

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

                    all_sets.append(subset_rem)
                    all_pvals.append(pval)

            accepted = np.where(pvals_a > alpha)

            if accepted[0].size > 0:

                best_loss = np.amin(loss_a[np.where(pvals_a > alpha)])

                selected[np.where(loss_a == best_loss)] = (
                    selected[np.where(loss_a == best_loss)] + 1
                ) % 2

                accepted_subset = np.sort(np.where(selected == 1)[0])
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


    # print(loss_a)

    if accepted_subset is None:
        print('Number of accepted sets: 0')

        all_pvals = np.array(all_pvals).flatten()

        max_pvals = np.argsort(all_pvals)[-1]
        accepted_subset = np.sort(all_sets[max_pvals])
    else:
        print('Number of accepted sets: ', len(accepted_subset ))


    return np.array(accepted_subset)


