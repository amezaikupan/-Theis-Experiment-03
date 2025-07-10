import numpy as np
from sklearn import linear_model


def get_color_dict():

    colors = {
        "pool": "red",
        "lasso": "red",
        "shat": "green",
        "sgreed": "green",
        "ssharp": "green",
        "strue": "blue",
        "cauid": "blue",
        "causharp": "blue",
        "cauul": "blue",
        "mean": "black",
        "msda": "orange",
        "mtl": "orange",
        "dica": "orange",
        "dom": "k",
        "naive": "magenta",
        "groundTruth": "purple"
    }

    markers = {
        "pool": "o",
        "lasso": "^",
        "shat": "o",
        "sgreed": "^",
        "strue": "^",
        "ssharp": "d",
        "cauid": "d",
        "causharp": "h",
        "cauul": "^",
        "mean": "o",
        "msda": "o",
        "mtl": "^",
        "dica": "d",
        "dom": "o",
        "naive": "o",
        "groundTruth": "^"

    }

    legends = {
        "pool": r"$\beta^{CS}$",
        "lasso": r"$\beta^{CS(\hat S Lasso)}$",
        "shat": r"$\beta^{CS(\hat S)}$",
        "ssharp": r"$\beta^{CS(\hat S \sharp)}$",
        "strue": r"$\beta^{CS(cau)}$",
        "cauid": r"$\beta^{CS(cau+,id)}$",
        "causharp": r"$\beta^{CS(cau\sharp)}$",
        "cauul": r"$\beta^{CS(cau\sharp UL)}$",
        "sgreed": r"$\beta^{CS(\hat{S}_{greedy})}$",
        "mean": r"$\beta^{mean}$",
        "msda": r"$\beta^{mSDA}$",
        "mtl": r"$\beta^{MTL}$",
        "dica": r"$\beta^{DICA}$",
        "naive": r"$\beta^{naive}$",
        "dom": r"$\beta^{dom}$",
        "groundTruth": "groundTruth"

    }

    return colors, markers, legends


def mse(model, x, y):
    return np.mean((model.predict(x) - y) ** 2)

def mae(model, x, y):
    return np.mean(np.abs(model.predict(x) - y))

def mae_2(x, y):
    """
    Mean Absolute Error between two vectors.
    
    Args:
        x: Predicted values (1D array)
        y: True values (1D array)
    
    Returns:
        Mean Absolute Error
    """
    return np.mean(np.abs(x - y))

def mse_2(x, y):
    """
    Mean Squared Error between two vectors.
    
    Args:
        x: Predicted values (1D array)
        y: True values (1D array)
    
    Returns:
        Mean Squared Error
    """
    return np.mean((x - y) ** 2)


def np_getDistances(x, y):
    K = x[:, :, np.newaxis] - y.T
    return np.linalg.norm(K, axis=1)


# # Select top 11 predictors from Lasso
# def lasso_alpha_search_synt(X, Y):
#     from sklearn import linear_model

#     exit_loop = False
#     alpha_lasso = 0.2
#     step = 0.02
#     num_iters = 1000
#     count = 0
#     n = 11

#     while not exit_loop and count < num_iters:
#         count = count + 1

#         regr = linear_model.Lasso(alpha=alpha_lasso)
#         regr.fit(X, Y.flatten())
#         zeros = np.where(np.abs(regr.coef_) < 0.00000000001)

#         nonzeros = X.shape[1] - zeros[0].shape[0]

#         if nonzeros == n:
#             exit_loop = True
#         if nonzeros < n:
#             alpha_lasso -= step
#         else:
#             step /= 2
#             alpha_lasso += step

#     mask = np.ones(X.shape[1], dtype=bool)
#     mask[zeros] = False
#     genes = []
#     index_mask = np.where(mask == True)[0]
#     return mask


def train_linear_and_eval(x, y, x_test, y_test):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    result = mse(model, x_test, y_test)
    return result, model.coef_

def lasso_alpha_search_synt(X, Y, target_features=11, max_iters=10, 
                            initial_alpha=0.2, initial_step=0.02, tol=1e-11):
    """
    Efficiently find lasso alpha to select target number of features.
    
    Args:
        X: Feature matrix
        Y: Target vector
        target_features: Desired number of non-zero coefficients (default: 11)
        max_iters: Maximum iterations (default: 1000)
        initial_alpha: Starting alpha value (default: 0.2)
        initial_step: Initial step size (default: 0.02)
        tol: Tolerance for zero coefficients (default: 1e-11)
    
    Returns:
        Boolean mask indicating selected features
    """
    # from sklearn.linear_model import Lasso
    from sklearn.linear_model import LassoCV

    import numpy as np
    
    alpha = initial_alpha
    step = initial_step
    
    for _ in range(max_iters):
        # Fit lasso with current alpha
        
        from sklearn.model_selection import KFold
        alphas = np.logspace(-2, 1, 100)  # fine-grained log-scale from 0.0001 to 10

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        lasso = LassoCV(alphas=alphas, cv=cv, n_jobs=-1)
        lasso.fit(X, Y.ravel())
        
        # Count non-zero coefficients
        nonzero_count = np.sum(np.abs(lasso.coef_) >= tol)
        
        if nonzero_count == target_features:
            print(f"Found alpha: {alpha} with {nonzero_count} features")
            break
        elif nonzero_count < target_features:
            alpha -= step
        else:
            step *= 0.5
            alpha += step
    
    # Return mask of selected features

    # Sort features by the absolute value of their coefficients
    print(lasso.coef_)
    print("Number of non-zero coefficients:", np.sum(np.abs(lasso.coef_) >= tol))
    print("Selected features:", np.where(np.abs(lasso.coef_) >= tol)[0])
    print("Top 11 features:", np.argsort(np.abs(lasso.coef_))[:11])
    sorted_indices = np.argsort(np.abs(lasso.coef_))[::-1]  # Descending order
    top_11_indices = sorted_indices[:11]  # Select the top 11 features

    return top_11_indices


def lasso_binary_search_alpha(
    X, y, target_features=11, tol=1e-6,
    alpha_min=1e-4, alpha_max=10.0,
    max_iter=10000, max_search_iter=20,
    return_mask=False, random_state=None
):
    """
    Binary search to find alpha that yields target number of non-zero features using Lasso.

    Args:
        X: Feature matrix (NumPy array or pandas DataFrame)
        y: Target vector
        target_features: Desired number of non-zero coefficients
        tol: Threshold to count non-zero coefficients
        alpha_min: Minimum alpha for binary search
        alpha_max: Maximum alpha for binary search
        max_iter: Max iterations for Lasso
        max_search_iter: Max binary search steps
        return_mask: If True, return boolean mask instead of indices
        scale: If True, apply StandardScaler to X
        random_state: Not used but reserved for future use

    Returns:
        selected_alpha: The alpha yielding desired number of features
        model: Fitted Lasso model
        selected: Indices or boolean mask of selected features
    # """
    # if scale:
    #     scaler = StandardScaler()
    #     X = scaler.fit_transform(X)

    from sklearn.linear_model import Lasso

    selected_alpha = None
    selected = None
    model = None

    for _ in range(max_search_iter):
        alpha = (alpha_min + alpha_max) / 2
        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X, y)
        nonzero_mask = np.abs(model.coef_) > tol
        nonzero_count = np.sum(nonzero_mask)

        if nonzero_count == target_features:
            selected_alpha = alpha
            break
        elif nonzero_count > target_features:
            alpha_min = alpha  # Increase regularization
        else:
            alpha_max = alpha  # Decrease regularization

        selected_alpha = alpha  # Keep track of last tried

    # Select top features
    sorted_indices = np.argsort(np.abs(model.coef_))[::-1]
    top_indices = sorted_indices[:target_features]

    if return_mask:
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[top_indices] = True
        selected = mask
    else:
        selected = top_indices

    return selected
