import numpy as np
import scipy as sc
from scipy.stats import wishart
from sklearn.cluster import KMeans


def _gen_coef(coef_0, lambd, mask=None):
    if mask is not None:
        mask_compl = ((mask + 1) % 2).astype(bool)
        draw = np.random.normal(0, 1, coef_0.shape)
        ret = (1 - lambd) * coef_0 + lambd * draw
        ret[mask_compl] = coef_0[mask_compl]
        return ret
    return (1 - lambd) * coef_0 + lambd * np.random.normal(0, 1, coef_0.shape)


def _draw_cov(p):
    scale = np.random.normal(0, 1, (p, p))
    scale = np.dot(scale.T, scale)
    cov = scale if p == 1 else wishart.rvs(df=p, scale=scale)

    for i in range(p):
        for j in range(p):
            if i != j:
                cov[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

    np.fill_diagonal(cov, 1)
    return cov


def _generate_covs(n_task, p_s, p_n, mask=None):
    cov_s = [_draw_cov(p_s) for _ in range(n_task)]
    cov_n = []

    fix, ref = -1, None
    if mask is not None:
        fix_mask = np.where(mask == False)[0]
        if len(fix_mask) > 0:
            fix = fix_mask.size
            ref = _draw_cov(fix)

    for k in range(n_task):
        cov_n_k = _draw_cov(p_n)

        if fix > 0:
            cov_n_k[-fix:, -fix:] = ref
            for _ in range(100):
                if np.all(np.linalg.eig(cov_n_k)[0] > 0):
                    break
                samp = np.random.normal(0, 1, (fix, p_n - fix))
                if np.any(np.array(samp.shape) == 1):
                    samp = samp.flatten()
                if fix == 1:
                    cov_n_k[-fix, :p_n - fix] = samp.flatten()
                    cov_n_k[:p_n - fix, -fix] = samp.flatten()
                elif fix == p_n - 1:
                    cov_n_k[-fix:, p_n - fix - 1] = samp.flatten()
                    cov_n_k[p_n - fix - 1, -fix:] = samp.flatten()
                else:
                    cov_n_k[-fix:, :p_n - fix] = samp
                    cov_n_k[:p_n - fix, -fix:] = samp.T
        cov_n.append(cov_n_k)

    return cov_s, cov_n


def _generate_task_data(n_task, n, p_s, p_n, alpha, eps, g, lambd, beta_0, gamma_0, mask=None, nonlinear='quadratic'):
    cov_s, cov_n = _generate_covs(n_task, p_s, p_n, mask)
    gamma = [_gen_coef(gamma_0, lambd, mask) for _ in range(n_task)]
    beta = [_gen_coef(beta_0, lambd) for _ in range(n_task)]

    domain_data = {}
    for k in range(n_task):
        xs = np.random.multivariate_normal(np.zeros(p_s), cov_s[k], n)
        if nonlinear == 'linear':
            y = np.dot(xs, alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'quadratic':
            y = np.dot(xs**2, alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'cubic':
            y = np.dot(xs**3, alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'mixed':
            y = np.dot(xs, alpha) + 0.5 * np.dot(xs**2, alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'sin':
            y = np.dot(np.sin(xs), alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'tanh':
            y = np.dot(np.tanh(xs), alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'exp':
            y = np.dot(np.exp(np.clip(xs, -2, 2)), alpha) + eps * np.random.normal(0, 1, (n, 1))
        elif nonlinear == 'log':
            y = np.dot(np.log(np.abs(xs) + 0.1), alpha) + eps * np.random.normal(0, 1, (n, 1))
        else:
            raise ValueError(f"Unknown nonlinear option: {nonlinear}")

        xn = np.dot(y, gamma[k].T) + g * np.random.multivariate_normal(np.zeros(p_n), cov_n[k], n)
        if beta_0.size > 0:
            p_conf = beta_0.shape[0]
            xn += np.dot(xs[:, p_s - p_conf:], beta[k])

        X = np.concatenate([xs, xn], 1)
        domain_data[f'domain_{k}'] = {'X': X, 'y': y}

    return domain_data, cov_s, cov_n, gamma, beta


class gauss_tl:
    def __init__(self, n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test, mask=None, nonlinear='quadratic'):
        self.is_full = (p_s == p)
        if self.is_full:
            p += 1

        self.n_task, self.n = n_task, n
        self.p, self.p_s, self.p_conf = p, p_s, p_conf
        self.eps, self.g = eps, g
        self.lambd, self.lambd_test = lambd, lambd_test
        self.nonlinear = nonlinear

        p_n = p - p_s

        self.alpha = _gen_coef(np.random.normal(0, 1, (p_s, 1)), 0)
        self.gamma_0 = np.random.normal(0, 1, (p_n, 1))
        self.beta_0 = np.random.normal(0, 1, (p_conf, p_n))

        self._generate_data(lambd, mask)
        self._generate_test_data(lambd_test, mask)

        if self.is_full:
            for ds in [self.train, self.test]:
                for dom in ds['domains'].values():
                    dom['X'] = dom['X'][:, :-1]
            self.p -= 1
            self.alpha = self.alpha[:-1]

    def _generate_data(self, lambd, mask=None):
        p_n = self.p - self.p_s
        domain_data, cov_s, cov_n, gamma, beta = _generate_task_data(
            self.n_task, self.n, self.p_s, p_n, self.alpha, self.eps, self.g,
            lambd, self.beta_0, self.gamma_0, mask, self.nonlinear)
        self.train = {
            'domains': domain_data,
            'n_ex': np.full(self.n_task, self.n),
            'cov_s': cov_s, 'cov_n': cov_n,
            'gamma': gamma, 'beta': beta, 'eps': self.eps
        }

    def _generate_test_data(self, lambd_test, mask=None):
        p_n = self.p - self.p_s
        domain_data, cov_s, cov_n, gamma, beta = _generate_task_data(
            self.n_task, self.n, self.p_s, p_n, self.alpha, self.eps, self.g,
            lambd_test, self.beta_0, self.gamma_0, mask, self.nonlinear)
        self.test = {
            'domains': domain_data,
            'n_ex': np.full(self.n_task, self.n),
            'cov_s': cov_s, 'cov_n': cov_n,
            'gamma': gamma, 'beta': beta, 'eps': self.eps
        }
