import numpy as np
import warnings
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from scipy.interpolate import UnivariateSpline

def RF_from_matrix(M, x, y):
    nx, ny = len(x), len(y)
    assert M.shape == (nx, ny)
    X = np.tile(x, ny)
    Y = np.repeat(y, nx)
    Z = M.flatten(order='F')
    return {'X': X, 'Y': Y, 'Z': Z, 'dim': (nx, ny)}

def cov_model(model, range_=1.0, nu=1.5):
    if model == 'Gauss':
        return lambda h: np.exp(-(np.asarray(h) / range_)**2 / 2.0)
    elif model == 'Exp':
        return lambda h: np.exp(-np.asarray(h) / range_)
    elif model == 'Matern':
        def f(h):
            h = np.asarray(h, dtype=float)
            out = np.ones_like(h)
            pos = h > 0
            hp = h[pos] / range_
            arg = np.sqrt(2.0 * nu) * hp
            out[pos] = (2.0**(1.0 - nu) / gamma(nu)) * arg**nu * kv(nu, arg)
            return out
        return f
    else:
        raise ValueError(f"Modèle inconnu : {model}")

class RandomFieldGenerator:
    def __init__(self, x, y, cov_fun, method='eig', cond_sites=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.cond_sites = cond_sites
        nx, ny = len(self.x), len(self.y)

        Xg, Yg = np.meshgrid(self.x, self.y, indexing='ij')
        grid = np.column_stack([Xg.ravel(order='F'), Yg.ravel(order='F')])
        q = grid.shape[0]

        if cond_sites is not None:
            method = 'eig'
            cs = np.atleast_2d(cond_sites)
            full_grid = np.vstack([grid, cs])
        else:
            full_grid = grid

        cov_mat = cov_fun(cdist(full_grid, full_grid))

        if cond_sites is not None:
            m = cs.shape[0]
            Sigma11 = cov_mat[:q, :q]
            Sigma12 = cov_mat[:q, q:]
            Sigma21 = cov_mat[q:, :q]
            Sigma22 = cov_mat[q:, q:]
            self.reg_coeff = Sigma12 @ np.linalg.solve(Sigma22, np.eye(m))
            cov_mat = Sigma11 - self.reg_coeff @ Sigma21
        else:
            self.reg_coeff = None

        if method == 'chol':
            cov_mat += 1e-10 * np.eye(q)
            self.L = np.linalg.cholesky(cov_mat)
        else:
            vals, vecs = np.linalg.eigh(cov_mat)
            vals = np.maximum(vals, 0.0)
            self.L = vecs @ np.diag(np.sqrt(vals))

    def generate(self, cond_val=None):
        z = np.random.randn(self.L.shape[1])
        mu = self.reg_coeff @ np.asarray(cond_val) if self.reg_coeff is not None else 0.0
        field = self.L @ z + mu
        nx, ny = len(self.x), len(self.y)
        return RF_from_matrix(field.reshape((nx, ny), order='F'), self.x, self.y)

def rpareto(n, alpha=1.0):
    return np.random.uniform(size=n) ** (-1.0 / alpha)

def gauss_above(u):
    if u > 3:
        return u + np.random.exponential(1.0 / u)
    while True:
        w = np.random.randn()
        if w >= u: return w

def student_RF(generator, k=3, thresh=-np.inf):
    while True:
        conds = np.random.randn(k + 1)
        val0 = conds[k] / np.sqrt(np.sum(conds[:k]**2) / k) * np.sqrt((k - 2.0) / k)
        if val0 > thresh: break
    X = generator.generate(cond_val=[conds[k]])
    denom = np.zeros(X['Z'].shape)
    for i in range(k):
        denom += generator.generate(cond_val=[conds[i]])['Z']**2
    X['Z'] = X['Z'] / np.sqrt(denom / k) * np.sqrt((k - 2.0) / k)
    return X

def chi2_RF(generator, k=3, thresh=-np.inf):
    while True:
        conds = np.random.randn(k)
        val0 = (np.sum(conds**2) - k) / np.sqrt(2.0 * k)
        if val0 > thresh: break
    total = np.zeros(generator.L.shape[0])
    for i in range(k):
        total += generator.generate(cond_val=[conds[i]])['Z']**2
    Xi = generator.generate(cond_val=[conds[0]])
    Xi['Z'] = (total - k) / np.sqrt(2.0 * k)
    return Xi

def mixture_RF(generator, thresh=-np.inf, alpha=2.0):
    while True:
        W0 = np.random.randn()
        Lam = rpareto(1, alpha=alpha)[0]
        if W0 * Lam >= thresh: break
    RF = generator.generate(cond_val=[W0])
    RF['Z'] = RF['Z'] * Lam
    return RF

def extent_profile(RF, u, max_dist=np.inf, x0=(0.0, 0.0)):
    dists = np.sqrt((RF['X'] - x0[0])**2 + (RF['Y'] - x0[1])**2)
    mask  = dists <= max_dist
    dists = dists[mask]
    Z     = RF['Z'][mask]

    order = np.argsort(dists)
    dists = dists[order]
    Z     = Z[order]

    uniq, cnt_all = np.unique(dists, return_counts=True)
    cum_all = np.cumsum(cnt_all)

    dist_ex = dists[Z >= u]
    if len(dist_ex) == 0:
        return {'x': uniq, 'y': np.zeros(len(uniq))}

    uniq_ex, cnt_ex = np.unique(dist_ex, return_counts=True)
    cum_ex  = np.cumsum(cnt_ex)

    idx = np.searchsorted(uniq_ex, uniq, side='right')
    idx = np.clip(idx, 0, len(cum_ex))
    cum_ex_at_r = np.where(idx > 0, cum_ex[idx - 1], 0)

    return {'x': uniq, 'y': cum_ex_at_r / cum_all}

def get_extremal_range(ep):
    idx = np.argmax(ep['y'] < 1.0)
    if ep['y'][idx] < 1.0:
        return ep['x'][idx]
    return np.inf

def tail_dependence_fun(ep, new_x=None):
    if new_x is None:
        new_x = ep['x']
    new_x = np.asarray(new_x)

    x_fit = np.concatenate([[0.0], ep['x']])
    y_fit = np.concatenate([[1.0], ep['y']])
    w     = np.concatenate([[1e6], np.ones(len(ep['x']))])

    n = len(x_fit)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spl = UnivariateSpline(x_fit, y_fit, w=w, k=3, s=0.95 * n, ext=3)

    phi      = spl(new_x)
    phi_d    = spl.derivative()(new_x)
    chi_p    = phi + (new_x / 2.0) * phi_d
    slope_at_0 = float(spl.derivative()(0.0))

    return {'x': new_x, 'y': chi_p, 'slope_at_0': slope_at_0}

def density_at_zero(x):
    """
    Estime lim_{r→0+} P(R ≤ r)/r = f(0).
    Correction majeure : Élimination des doublons de coordonnées pour UnivariateSpline.
    """
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    
    if np.all(bad): 
        return 0.0
    elif np.any(bad):
        x[bad] = np.max(x[~bad])

    uniq_x, counts = np.unique(x, return_counts=True)
    ys = np.cumsum(counts) / len(x)

    x_fit = np.concatenate([[0.0], uniq_x])
    y_fit = np.concatenate([[0.0], ys])
    w     = np.concatenate([[1e6], np.ones(len(uniq_x))])

    if len(x_fit) < 4:
        return float(y_fit[-1] / x_fit[-1]) if x_fit[-1] > 0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            spl = UnivariateSpline(x_fit, y_fit, w=w, k=3, s=0.99 * len(x_fit), ext=3)
            return max(0.0, float(spl.derivative()(0.0)))
        except Exception:
            return max(0.0, float(np.polyfit(x_fit[:4], y_fit[:4], 1)[0]))
        



# =============================================================================
# AJOUT À useful_functions.py
# Copier-coller ce bloc à la fin du fichier useful_functions.py
# =============================================================================
#
# Théorie (Proposition 3.11 du rapport) :
#   Pour un champ gaussien avec ρ(h) = ρ_iso(sqrt(h^T A h)),
#   la distance de Mahalanobis sqrt(h^T A h) remplace la distance euclidienne.
#   On obtient celle-ci en transformant la grille par A^{1/2} (Cholesky de A),
#   puis en calculant les distances euclidiennes sur la grille transformée.
#   La matrice A est la "shape matrix" : Λ = λ·A dans la notation du rapport.
# =============================================================================

class AnisotropicRFGenerator:
    """
    Génère des champs gaussiens anisotropes de covariance
        ρ_aniso(h) = ρ_iso( sqrt(h^T A h) ),
    où A est une matrice symétrique définie positive 2×2 (shape matrix).

    Implémentation : on transforme la grille par A^{1/2} (Cholesky),
    ce qui fait que les distances euclidiennes sur la grille transformée
    sont exactement les distances de Mahalanobis :
        ||A^{1/2}(p - q)||_2 = sqrt((p-q)^T A (p-q)).
    Le reste de la génération est identique à RandomFieldGenerator.

    Paramètres
    ----------
    x, y    : ndarray 1D — coordonnées de la grille
    cov_fun : callable — fonction de covariance isotrope f(distance_scalaire)
    A       : ndarray (2,2) — shape matrix (definie positive)
    method  : 'eig' (défaut) ou 'chol'
    """

    def __init__(self, x, y, cov_fun, A, method='eig', cond_sites=None):
        from scipy.spatial.distance import cdist

        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.cond_sites = cond_sites
        nx, ny = len(self.x), len(self.y)

        # Grille 2D → liste de points (q, 2)
        Xg, Yg = np.meshgrid(self.x, self.y, indexing='ij')
        grid   = np.column_stack([Xg.ravel(order='F'), Yg.ravel(order='F')])
        q      = grid.shape[0]

        # ---- Transformation Mahalanobis ----------------------------------------
        # A = A_half @ A_half.T  (Cholesky inférieur)
        # Distance de Mahalanobis(p, q) = ||A_half.T @ (p-q)||_2
        # = distance euclidienne entre A_half.T @ p  et  A_half.T @ q
        A_half    = np.linalg.cholesky(np.asarray(A, dtype=float))
        grid_t    = grid @ A_half.T           # grille transformée (q, 2)

        if cond_sites is not None:
            cs      = np.atleast_2d(cond_sites)
            cs_t    = cs @ A_half.T
            full_t  = np.vstack([grid_t, cs_t])
        else:
            full_t  = grid_t

        # Matrice de covariance sur les distances de Mahalanobis
        cov_mat = cov_fun(cdist(full_t, full_t))

        # ---- Conditionnement (krigeage) ----------------------------------------
        if cond_sites is not None:
            m             = cs.shape[0]
            Sigma11       = cov_mat[:q, :q]
            Sigma12       = cov_mat[:q, q:]
            Sigma21       = cov_mat[q:, :q]
            Sigma22       = cov_mat[q:, q:]
            self.reg_coeff = Sigma12 @ np.linalg.solve(Sigma22, np.eye(m))
            cov_mat        = Sigma11 - self.reg_coeff @ Sigma21
        else:
            self.reg_coeff = None

        # ---- Décomposition spectrale / Cholesky --------------------------------
        if method == 'chol':
            cov_mat += 1e-10 * np.eye(q)
            self.L   = np.linalg.cholesky(cov_mat)
        else:                                    # 'eig' (défaut, plus stable)
            vals, vecs = np.linalg.eigh(cov_mat)
            vals       = np.maximum(vals, 0.0)
            self.L     = vecs @ np.diag(np.sqrt(vals))

    # ---- Génération d'une réalisation ------------------------------------------
    def generate(self, cond_val=None):
        z  = np.random.randn(self.L.shape[1])
        mu = self.reg_coeff @ np.asarray(cond_val) \
             if self.reg_coeff is not None else 0.0
        field = self.L @ z + mu
        nx, ny = len(self.x), len(self.y)
        return RF_from_matrix(field.reshape((nx, ny), order='F'), self.x, self.y)