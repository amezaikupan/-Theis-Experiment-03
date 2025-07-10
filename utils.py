import numpy as np

from sklearn import linear_model
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import solve
import gc
import os
import torch

def train_linear_and_eval(x, y, x_test, y_test):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    result = mse(model, x_test, y_test)

def get_color_dict():
  colors = {
    'pool' : 'red',
    'lasso' : 'red',
    'shat' : 'green',
    'sgreed' : 'green',
    'ssharp' : 'green',
    'strue' : 'blue',
    'cauid' : 'blue',
    'causharp': 'blue',
    'cauul' : 'blue',
    'mean' : 'black',
    'msda' : 'orange',
    'mtl' : 'orange',
    # 'dica' : 'orange',
    'cicm' : 'orange',
    'dom' : 'k',
    'naive' : 'magenta'
  }

  markers = {
    'pool' : 'o',
    'lasso' : '^',
    'shat' : 'o',
    'sgreed' : '^',
    'strue' : '^',
    'ssharp' : 'd',
    'cauid' : 'd',
    'causharp' : 'h',
    'cauul' : '^',
    'mean' : 'o',
    'msda' : 'o',
    'mtl' : '^',
    # 'dica' : 'd',
    'cicm' : 'd',
    'dom' : 'o',
    'naive' : 'o'
  }

  legends = {
              'pool' : r'$\beta^{CS}$',
              'lasso' : r'$\beta^{CS(\hat S Lasso)}$',
              'shat' : r'$\beta^{CS(\hat S)}$',
              'ssharp' : r'$\beta^{CS(\hat S \sharp)}$',
              'strue' : r'$\beta^{CS(cau)}$',
              'cauid' : r'$\beta^{CS(cau+,id)}$',
              'causharp' : r'$\beta^{CS(cau\sharp)}$',
              'cauul' : r'$\beta^{CS(cau\sharp UL)}$',
              'sgreed' :r'$\beta^{CS(\hat{S}_{greedy})}$',
              'mean'   : r'$\beta^{mean}$',
              'msda'   : r'$\beta^{mSDA}$',
              'mtl'   : r'$\beta^{MTL}$',
              'dica'   : r'$\beta^{DICA}$',
              'naive'   : r'$\beta^{naive}$',
              'dom'   : r'$\beta^{dom}$',
              'cicm'   : r'$\beta^{cICM}$',
            }

  return colors, markers, legends

# def mse(model, x, y):
#   return np.mean((model.predict(x)-y)**2)
def mse(model, x, y):
    y_pred = model.predict(x).ravel()
    y_true = y.ravel()
    return np.mean((y_pred - y_true) ** 2)



def compute_rbf_kernel_blockwise(X, Y=None, gamma=1.0, block_size=500, dtype=np.float32):
    if Y is None:
        Y = X

    n_X = X.shape[0]
    n_Y = Y.shape[0]
    K = np.empty((n_X, n_Y), dtype=dtype)

    for i in range(0, n_X, block_size):
        i_end = min(i + block_size, n_X)
        Xi = X[i:i_end]
        K[i:i_end] = rbf_kernel(Xi, Y, gamma=gamma).astype(dtype)

    return K


def compute_rbf_kernel_blockwise_to_memmap(X, Y=None, gamma=1.0, block_size=500, 
                                           dtype=np.float32, mmap_path='kernel.dat'):
  
    if Y is None:
        Y = X

    if Y is None:
        Y = X

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    K = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=(n_X, n_Y))

    for i in range(0, n_X, block_size):
        i_end = min(i + block_size, n_X)
        Xi = X[i:i_end]
        
        # Compute RBF kernel for block
        K_block = rbf_kernel(Xi, Y, gamma=gamma).astype(dtype)

        # Store to memmap file
        K[i:i_end] = K_block
        K.flush()  # Ensure data is written to disk

    return K

def center_kernel_memmap(K_path, shape):

    N = shape[0]
    K = np.memmap(K_path, dtype='float32', mode='r', shape=shape)
    
    # Compute row and column means and total mean
    row_mean = np.zeros(N, dtype='float32')
    for i in range(N):
        row_mean[i] = np.mean(K[i, :])

    col_mean = np.zeros(N, dtype='float32')
    for j in range(N):
        col_mean[j] = np.mean(K[:, j])

    total_mean = np.mean(row_mean)

    # Allocate new memmap for centered matrix
    centered_path = K_path.replace('.dat', '_centered.dat')
    K_centered = np.memmap(centered_path, dtype='float32', mode='w+', shape=(N, N))

    for i in range(N):
        for j in range(N):
            K_centered[i, j] = K[i, j] - row_mean[i] - col_mean[j] + total_mean

    del K, K_centered
    return centered_path

def build_L_memmap(groupIdx, N, L_path='L.dat', dtype=np.float32):
    unique_groups = np.unique(groupIdx)
    G = len(unique_groups)
    NG = np.array([np.sum(groupIdx == g) for g in unique_groups])

    L = np.memmap(L_path, dtype=dtype, mode='w+', shape=(N, N))

    for i in range(N):
        gi = groupIdx[i]
        groupSize_i = NG[unique_groups == gi][0]

        for j in range(N):
            gj = groupIdx[j]
            groupSize_j = NG[unique_groups == gj][0]

            if gi == gj:
                L[i, j] = 1 / (G * groupSize_i**2) - 1 / (G**2 * groupSize_i**2)
            else:
                L[i, j] = -1 / (G**2 * groupSize_i * groupSize_j)

        if i % 100 == 0:
        
            gc.collect()

    L.flush()
    del L
    return L_path


def compute_A_left_blockwise(Kx_path, L_path, out_path, N, lambd, block_size=512, dtype=np.float32):
    # Load memory-mapped inputs
    Kx = np.memmap(Kx_path, dtype=dtype, mode='r', shape=(N, N))
    L = np.memmap(L_path, dtype=dtype, mode='r', shape=(N, N))
    A_left = np.memmap(out_path, dtype=dtype, mode='w+', shape=(N, N))

    # Step 1: Compute T1 = Kx @ L
    base, ext = os.path.splitext(out_path)
    t1_path = base + "_t1" + ext
    T1 = np.memmap(t1_path, dtype=dtype, mode='w+', shape=(N, N))

    # print("Step 1: Computing T1 = Kx @ L ...")

    # T1_a = Kx @ L

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        T1[i:i_end, :] = Kx[i:i_end, :] @ L
        gc.collect()
    T1.flush()

    # Step 2: Compute T2 = T1 @ Kx
    # print("Step 2: Computing T2 = T1 @ Kx ...")
    # T2_a = T1_a @ Kx

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        A_left[i:i_end, :] = T1[i:i_end, :] @ Kx
        gc.collect()
    


    # Step 3: Add Kx to A_left
    # print("Step 3: Adding Kx to A_left ...")
    # T2_a += Kx

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        A_left[i:i_end, :] += Kx[i:i_end, :]
        gc.collect()

   

    # Step 4: Add lambd * I
    # print("Step 4: Adding λ * I to A_left ...")
    # T2_a = T2_a + lambd*np.eye(N)
    for i in range(0, N):
        A_left[i, i] += lambd

    # A = Kx @ L @ Kx + Kx + lambd * np.eye(N)
    A_left.flush()
    del A_left
    del Kx, L, T1
    gc.collect()

    return out_path

def compute_A_right_blockwise(Ky_path, mid_path, out_path, N, block_size=512, dtype=np.float32):
    Ky = np.memmap(Ky_path, dtype=dtype, mode='r', shape=(N, N))
    mid = np.memmap(mid_path, dtype=dtype, mode='r', shape=(N, N))
    A_right = np.memmap(out_path, dtype=dtype, mode='w+', shape=(N, N))  # output

    # print("Computing A_right = Ky @ mid blockwise...")
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        
        # Use float64 internally to reduce numerical error
        Ky_block = Ky[i:i_end, :].astype(np.float64)
        mid_64 = mid.astype(np.float64)
        A_right[i:i_end, :] = (Ky_block @ mid_64).astype(dtype)
        
        gc.collect()

    A_right.flush()
    del Ky, mid, A_right
    gc.collect()
    return out_path

def solve_blockwise(A_left_path, A_right_path, out_path, N, block_size=512, dtype=np.float32):
    A_left = np.memmap(A_left_path, dtype=dtype, mode='r', shape=(N, N))
    A_right = np.memmap(A_right_path, dtype=dtype, mode='r', shape=(N, N))
    A_out = np.memmap(out_path, dtype=dtype, mode='w+', shape=(N, N))  # Output A

    # print("Solving A_left @ A = A_right blockwise...")
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        rhs_block = A_right[:, i:i_end]
        A_block = solve(A_left, rhs_block)  # or 'pos' if A_left is PSD
        A_out[:, i:i_end] = A_block.astype(dtype)
        gc.collect()

    A_out.flush()
    del A_left, A_right, A_out
    gc.collect()
    return out_path



use_gpu = {"enabled": True}

def get_tensor_size_in_bytes(tensor):
    return tensor.element_size() * tensor.nelement()

def has_enough_gpu_memory(tensors):
    total_required = sum(get_tensor_size_in_bytes(t) for t in tensors if isinstance(t, torch.Tensor))
    total_required_gb = total_required / (1024**3)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        return free_mem >= total_required_gb
    return False

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(i, device) for i in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return type(x)(to_numpy(i) for i in x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        return x

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_tensor(i) for i in x)
    elif isinstance(x, dict):
        return {k: to_tensor(v) for k, v in x.items()}
    else:
        return x

def all_tensors(x):
    if isinstance(x, torch.Tensor):
        return [x]
    elif isinstance(x, (list, tuple)):
        return [t for i in x for t in all_tensors(i)]
    elif isinstance(x, dict):
        return [t for v in x.values() for t in all_tensors(v)]
    return []

def try_gpu_then_numpy(fn, *args, **kwargs):
    all_inputs = all_tensors(args) + all_tensors(kwargs)
    input_devices = {t.device.type for t in all_inputs if isinstance(t, torch.Tensor)}


    # If mixed devices or on CPU but memory available, try GPU
    try:
        if 'cpu' in input_devices or len(input_devices) > 1:
            print(f"[Align] Mixed device tensors found: {input_devices}")
            if has_enough_gpu_memory(all_inputs):
                print("[Align] Moving all inputs to GPU.")
                args = to_device(args, 'cuda')
                kwargs = to_device(kwargs, 'cuda')
            else:
                print("[Align] Moving all inputs to CPU.")
                args = to_device(args, 'cpu')
                kwargs = to_device(kwargs, 'cpu')
        return fn(*args, **kwargs)

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f"[OOM] Falling back to NumPy for: {fn.__name__ if hasattr(fn, '__name__') else str(fn)}")
            torch.cuda.empty_cache()
            gc.collect()

            args_np = to_numpy(args)
            kwargs_np = to_numpy(kwargs)

            try:
                print("[INFO] Using NumPy fallback.")
                result_np = fn(*args_np, **kwargs_np)
                result_tensor = to_tensor(result_np)

                try:
                    result_tensor = result_tensor.to("cuda")
                    print("[Recovery] Successfully moved result back to GPU.")
                except RuntimeError:
                    print("[Recovery] GPU move failed. Staying on CPU.")
                    use_gpu["enabled"] = False

                return result_tensor
            except Exception as e_np:
                print(f"[ERROR] NumPy fallback failed: {e_np}")
                raise e_np
        else:
            raise

def try_gpu_safe(fn, *args, **kwargs):
    all_inputs = all_tensors(args) + all_tensors(kwargs)
    input_devices = {t.device.type for t in all_inputs if isinstance(t, torch.Tensor)}

    # Nếu có tensor ở CPU hoặc mix device → thử chuyển sang GPU
    if 'cpu' in input_devices or len(input_devices) > 1:
        if has_enough_gpu_memory(all_inputs):
            print("[Align] Moving all inputs to GPU.")
            args = to_device(args, 'cuda')
            kwargs = to_device(kwargs, 'cuda')
        else:
            print("[Fallback] Moving all inputs to CPU.")
            args = to_device(args, 'cpu')
            kwargs = to_device(kwargs, 'cpu')

    try:
        return fn(*args, **kwargs)

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
            print("[OOM] CUDA memory error. Retrying on CPU...")
            torch.cuda.empty_cache()
            gc.collect()
            args_cpu = to_device(args, 'cpu')
            kwargs_cpu = to_device(kwargs, 'cpu')
            return fn(*args_cpu, **kwargs_cpu)
        else:
            raise


def build_L_matrix(groupIdx, N, dtype=torch.float32, device='cuda'):
    if isinstance (groupIdx, np.ndarray):
        groupIdx = groupIdx.astype(np.int32)
        unique_groups = np.unique(groupIdx)
        G = len(unique_groups)
        group_counts = np.array([(groupIdx == g).sum() for g in unique_groups])

        L = np.zeros((N, N), dtype=np.float32)

        for g, count in zip(unique_groups, group_counts):
            idx = np.where(groupIdx == g)[0]
            val = 1 / (G * count**2) - 1 / (G**2 * count**2)
            L[np.ix_(idx, idx)] = val

        for i, (g1, s1) in enumerate(zip(unique_groups, group_counts)):
            for j, (g2, s2) in enumerate(zip(unique_groups, group_counts)):
                if g1 != g2:
                    idx1 = np.where(groupIdx == g1)[0]
                    idx2 = np.where(groupIdx == g2)[0]
                    val = -1 / (G**2 * s1 * s2)
                    L[np.ix_(idx1, idx2)] = val
        return L
    else:
        groupIdx = groupIdx.to(device)
        unique_groups = torch.unique(groupIdx)
        G = len(unique_groups)
        group_counts = torch.stack([(groupIdx == g).sum() for g in unique_groups])

        L = torch.zeros((N, N), dtype=dtype, device=device)

        for g, count in zip(unique_groups, group_counts):
            idx = (groupIdx == g).nonzero(as_tuple=True)[0]
            val = 1 / (G * count.item()**2) - 1 / (G**2 * count.item()**2)
            L[idx.unsqueeze(1), idx] = val

        for i, (g1, s1) in enumerate(zip(unique_groups, group_counts)):
            for j, (g2, s2) in enumerate(zip(unique_groups, group_counts)):
                if g1 != g2:
                    idx1 = (groupIdx == g1).nonzero(as_tuple=True)[0]
                    idx2 = (groupIdx == g2).nonzero(as_tuple=True)[0]
                    val = -1 / (G**2 * s1.item() * s2.item())
                    L[idx1.unsqueeze(1), idx2] = val

        return L
    
def center_kernel_blockwise(K, block_size=512, device=None):
    if isinstance (K, np.ndarray):
        N = K.shape[0]
        K_mean_row = np.mean(K, axis=1, keepdims=True)
        K_mean_col = np.mean(K, axis=0, keepdims=True)
        K_mean_total = np.mean(K)

        K_centered = np.zeros_like(K)

        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            for j in range(0, N, block_size):
                j_end = min(j + block_size, N)

                block = K[i:i_end, j:j_end].copy()  # copy to avoid modifying original K
                block -= K_mean_row[i:i_end]
                block -= K_mean_col[:, j:j_end]
                block += K_mean_total
                K_centered[i:i_end, j:j_end] = block

        return K_centered
    else: 
        N = K.shape[0]
        K_mean_row = K.mean(dim=1, keepdim=True)
        K_mean_col = K.mean(dim=0, keepdim=True)
        K_mean_total = K.mean()

        K_centered = torch.zeros_like(K)

        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            for j in range(0, N, block_size):
                j_end = min(j + block_size, N)

                block = K[i:i_end, j:j_end]
                block -= K_mean_row[i:i_end]
                block -= K_mean_col[:, j:j_end]
                block += K_mean_total
                K_centered[i:i_end, j:j_end] = block

        return K_centered

def blockwise_mm(A, B, block_size=512, device=None):
    """Blockwise matrix multiplication: A (n x m) @ B (m x p)"""
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        n, m = A.shape
        _, p = B.shape
        result = np.zeros((n, p), dtype=A.dtype)

        for i in range(0, n, block_size):
            i_end = min(i + block_size, n)
            for j in range(0, p, block_size):
                j_end = min(j + block_size, p)
                for k in range(0, m, block_size):
                    k_end = min(k + block_size, m)
                    result[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]

        return result

    else:
        # Assume PyTorch tensors
        n, m = A.shape
        _, p = B.shape
        device = A.device
        dtype = A.dtype
        result = torch.zeros((n, p), device=device, dtype=dtype)

        for i in range(0, n, block_size):
            i_end = min(i + block_size, n)
            for j in range(0, p, block_size):
                j_end = min(j + block_size, p)
                for k in range(0, m, block_size):
                    k_end = min(k + block_size, m)
                    result[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]

        return result

def add_fn(a, b):
    if isinstance(a, np.ndarray):
        print("[INFO] Using NumPy add.")
        return np.add(a, b)
    else:
        print("[INFO] Using Torch add.")
        return torch.add(a, b)

def solve_fn(a, b):
    if isinstance(a, np.ndarray):
        print("[INFO] Using NumPy solve.")
        return np.linalg.solve(a, b)
    else:
        print("[INFO] Using Torch solve.")
        return torch.linalg.solve(a, b)
        
def matmul_fn(a, b):
    if isinstance(a, np.ndarray):
        print("[INFO] Using NumPy matmul.")
        return np.matmul(a, b)
    else:
        print("[INFO] Using Torch matmul.")
        return torch.matmul(a, b)
    
def safe_eye(N, dtype=torch.float32):
    try:
        print("[INFO] Trying Torch eye on GPU.")
        return torch.eye(N, dtype=dtype, device='cuda')
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"[WARNING] Torch eye on GPU failed, switching to CPU: {e}")
        return torch.eye(N, dtype=dtype, device='cpu')


def safe_eigh(a):
    if isinstance(a, np.ndarray):
        print("[INFO] Using NumPy eigh.")
        return np.linalg.eigh(a)
    else:
        print("[INFO] Using Torch eigh.")
        return torch.linalg.eigh(a)
    
# def safe_eig(a):
#     if isinstance(a, np.ndarray):
#         print("[INFO] Using NumPy eig.")
#         return np.linalg.eig(a)
#     else:
#         print("[INFO] Using Torch eig.")
#         return torch.linalg.eig(a)
def safe_eig(a):
    if isinstance(a, np.ndarray):
        print("[INFO] Using NumPy eig.")
        return np.linalg.eig(a)
    try:
        print("[INFO] Using Torch eig on GPU.")
        return torch.linalg.eig(a)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print("[WARNING] Torch eig on GPU failed, switching to CPU:", e)
        a_cpu = a.detach().cpu()
        return torch.linalg.eig(a_cpu)

    
def safe_topk(values, k):
    if isinstance(values, np.ndarray):
        print("[INFO] Using NumPy topk.")
        indices = np.argpartition(-values, k)[:k]
        topk_vals = values[indices]
        # Sort descending
        sorted_idx = np.argsort(-topk_vals)
        indices = indices[sorted_idx]
        return topk_vals[sorted_idx], indices
    else:
        print("[INFO] Using Torch topk.")
        return torch.topk(values, k)
def safe_sqrt(x):
    if isinstance(x, np.ndarray):
        print("[INFO] Using NumPy sqrt.")
        return np.sqrt(x)
    else:
        print("[INFO] Using Torch sqrt.")
        return torch.sqrt(x)
def safe_diag(x):
    if isinstance(x, np.ndarray):
        print("[INFO] Using NumPy diag.")
        return np.diag(x)
    else:
        print("[INFO] Using Torch diag.")
        return torch.diag(x)
def scale_fn(x, scalar):
    if isinstance(x, np.ndarray):
        print("[INFO] Using NumPy scale.")
        return x * scalar
    else:
        print("[INFO] Using Torch scale.")
        return x * scalar
    
def safe_ones(shape, dtype=torch.float32):
    try:
        print("[INFO] Trying Torch ones on GPU.")
        return torch.ones(shape, device='cuda', dtype=dtype)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"[WARNING] Torch ones on GPU failed, switching to CPU: {e}")
        return torch.ones(shape, device='cpu', dtype=dtype)
    
def safe_unique(tensor):
    try:
        print("[INFO] Using Torch unique.")
        return torch.unique(tensor)
    except RuntimeError as e:
        print(f"[WARNING] Torch unique failed, switching to CPU: {e}")
        tensor_cpu = tensor.detach().cpu()
        return torch.unique(tensor_cpu)
def safe_divide(tensor, value):
    try:
        # Try dividing on GPU if possible
        return tensor / value
    except RuntimeError as e:
        print("[INFO] GPU out of memory during division, switching to CPU:", e)
        # Fallback to CPU if GPU allocation fails
        return tensor.to('cpu') / value

def subtract_fn(a, b):
    return a - b
def divide_fn(a, b, unsqueeze_dim=None):
    if not torch.is_tensor(b):
        b = torch.tensor(b, dtype=a.dtype, device=a.device)
    else:
        b = b.to(a.device)

    if unsqueeze_dim is not None:
        b = b.unsqueeze(unsqueeze_dim)

    return a / b

#Select top 11 predictors from Lasso
def lasso_alpha_search_synt(X,Y):

    exit_loop = False
    alpha_lasso = 0.2
    step = 0.02
    num_iters = 1000
    count = 0
    n = 11

    while(not exit_loop and count < num_iters):
            count = count + 1

            regr = linear_model.Lasso(alpha = alpha_lasso)
            regr.fit(X,Y.flatten())
            zeros =  np.where(np.abs(regr.coef_) < 0.00000000001)

            nonzeros = X.shape[1]-zeros[0].shape[0]

            if(nonzeros >= n and nonzeros<n+1):
                    exit_loop = True
            if nonzeros<n:
                    alpha_lasso -= step
            else:
                    step /= 2
                    alpha_lasso += step


    mask = np.ones(X.shape[1],dtype = bool)
    mask[zeros] = False
    genes = []
    index_mask = np.where(mask == True)[0]

    return mask


def intervene_on_p(l_p, sz):
  mask = np.zeros((sz, 1), dtype = bool)
  if len(l_p) > 0:
    mask[l_p] = True
  return mask