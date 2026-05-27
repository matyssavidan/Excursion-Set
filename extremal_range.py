import os
import pickle
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from math import comb
from scipy.stats import norm, chi2 as chi2dist, t as tdist, expon
from scipy.special import gamma
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm

from useful_functions import (
    RandomFieldGenerator, cov_model,
    student_RF, chi2_RF, mixture_RF,
    extent_profile, get_extremal_range,
    tail_dependence_fun, density_at_zero,
    RF_from_matrix, gauss_above, rpareto, 
    AnisotropicRFGenerator,
)


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =============================================================================
# Paramètres globaux
# =============================================================================

MATERN_NU  = 2.5
LAMBDA     = MATERN_NU / (MATERN_NU - 1)   # ≈ 1.667

# GRID_N = 41 pour tester rapidement.
# GRID_N = 121 pour reproduire le papier (lent + RAM important).
GRID_N = 121


COLORS  = {'Gaussian': 'red', 'Student': 'blue',
           'Chi-squared': 'orange', 'Mixture': 'violet'}
MARKERS = {'Gaussian': 'o', 'Student': 'D',
           'Chi-squared': '^', 'Mixture': 's'}
LS      = {'Gaussian': 'solid', 'Student': 'dashdot',
           'Chi-squared': 'dashdot', 'Mixture': 'solid'}
LW      = {'Gaussian': 1.2, 'Student': 0.8,
           'Chi-squared': 1.2, 'Mixture': 0.8}

_LABEL  = {'gauss': 'Gaussian', 'student': 'Student',
           'chi2': 'Chi-squared', 'mixture': 'Mixture'}


# =============================================================================
# CDF marginales et conversion vers marges Exp(1)
# =============================================================================

def pmixture(z, alpha=2):
    """
    CDF de W = Λ·G, Λ ~ Pareto(α), G ~ N(0,1).
    P(W ≤ z) = ∫_1^∞ Φ(z/x) α/x^{α+1} dx
    Équivalent R : pmixture(z, alpha)
    """
    def _scalar(zi):
        val, _ = quad(lambda x: norm.cdf(zi / x) * alpha / x**(alpha + 1),
                      1.0, np.inf, limit=200)
        return val
    return np.array([_scalar(zi) for zi in np.atleast_1d(z)])


def marginal_cdf(u, field_type, k=3, alpha_mix=2):
    """
    CDF marginale F_X(u).

    Convention des axes (fidèle au R) :
      'gauss'   : u est le seuil sur N(0,1)
      'student' : u est le seuil du Student NON normalisé (var = k/(k-2))
                  → les seuils stockés (var=1) doivent d'abord être * sqrt(k)
      'chi2'    : u est la valeur du chi-2 NON normalisé (E=k, Var=2k)
                  → les seuils stockés (centrés réduits) doivent d'abord être
                    convertis : u_chi2 = k + u_stored * sqrt(2k)
      'mixture' : u est le seuil sur W directement
    """
    if field_type == 'gauss':
        return norm.cdf(u)
    elif field_type == 'student':
        return tdist.cdf(u, df=k)
    elif field_type == 'chi2':
        return chi2dist.cdf(u, df=k)
    elif field_type == 'mixture':
        return pmixture(u, alpha=alpha_mix)
    else:
        raise ValueError(field_type)


def to_exp_scale(u, field_type, **kw):
    """
    Transforme un seuil u en valeur sur marges Exp(1) : -log(1 - F_X(u)).
    u doit déjà être sur l'échelle naturelle du champ (voir marginal_cdf).
    """
    p = marginal_cdf(u, field_type, **kw)
    return -np.log(np.clip(1.0 - p, 1e-15, None))


def _gauss_to_exp(u_arr):
    """
    Conversion gaussienne → Exp(1) avec extrapolation log-log pour u ≥ 7.
    Identique à gauss_to_exp() dans le R.
    """
    _x   = np.arange(6.0, 7.1, 0.1)
    _y   = -np.log(1.0 - norm.cdf(_x))
    coef = np.polyfit(np.log(_x), np.log(_y), 1)
    def _c(u):
        if u < 7.0:
            return -np.log(max(1.0 - norm.cdf(u), 1e-300))
        return np.exp(coef[1] + coef[0] * np.log(u))
    return np.array([_c(u) for u in u_arr])


# =============================================================================
# Courbes théoriques — Figures 3 / 4  (2 C*_1 / C*_2)
# =============================================================================

def lower_gamma(a, x):
    if x <= 0: return 0.0
    v, _ = quad(lambda t: t**(a-1)*np.exp(-t), 0.0, x)
    return v


def compute_C_star(k, u, alpha, lam):
    """
    LKC de dimension k pour le champ mélange W = Λ·G, Λ ~ Pareto(α).
    Utilisée avec alpha=1 pour les courbes théoriques (comme dans le R).
    Équivalent R : compute_C_star(k, u, alpha, lambda)
    """
    def omega(j): return np.pi**(j/2)/gamma(1+j/2)
    def E_rho(j, u, alpha):
        if   j == 0: return (u**(-2*alpha)*2**(alpha-1)*np.pi**(-0.5)*
                              lower_gamma(alpha+0.5, u**2/2) + 1-norm.cdf(u))
        elif j == 1: return (u**(-2*alpha)*2**(alpha-1)*np.pi**(-1)*
                              alpha*lower_gamma(alpha, u**2/2))
        elif j == 2: return (u**(-2*alpha)*2**(alpha-1)*np.pi**(-1.5)*
                              alpha*lower_gamma(alpha+0.5, u**2/2))
        else: raise ValueError
    j     = 2 - k
    coeff = comb(2, j)*omega(2)/(omega(j)*omega(2-j))
    return coeff * lam**(j/2) * E_rho(j, u, alpha)


def theoretical_slope(field_type, x_vals, k=3, lam=LAMBDA):
    """
    2 C*_1(u) / C*_2(u) pour un vecteur de seuils.

    Axes x (identiques au R) :
      'gauss'   : seuil N(0,1),          range [-2.5, 2.7]
      'student' : seuil Student non norm. (var=k/(k-2)), range [-2.5, 2.7]
      'chi2'    : valeur chi-2 non norm.  (E=k, Var=2k),  range [0, 6.5]
      'mixture' : seuil sur W,  alpha=1 dans compute_C_star, range [-1.5, 6.5]
    """
    x = np.asarray(x_vals)
    if field_type == 'gauss':
        C1 = 0.25 * np.sqrt(lam) * np.exp(-x**2 / 2.0)
        C2 = 1.0 - norm.cdf(x)
    elif field_type == 'student':
        # Transformation interne identique au R : u = u * sqrt((k-2)/k)
        u_in = x * np.sqrt((k-2.0)/k)
        C1   = 0.25*np.sqrt(lam)*(1.0 + u_in**2/(k-2.0))**((1.0-k)/2.0)
        C2   = 1.0 - tdist.cdf(x, df=k)   # x directement (non transformé)
    elif field_type == 'chi2':
        C1 = (np.sqrt(lam*np.pi)/(2.0**((k+1)/2)*gamma(k/2.0)) *
              x**((k-1)/2.0) * np.exp(-x/2.0))
        C2 = 1.0 - chi2dist.cdf(x, df=k)
    elif field_type == 'mixture':
        C1 = np.array([compute_C_star(1, xi, alpha=1, lam=lam) for xi in x])
        C2 = np.array([compute_C_star(2, xi, alpha=1, lam=lam) for xi in x])
    else:
        raise ValueError(field_type)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(C2 > 0, 2.0*C1/C2, np.nan)


def build_theoretical_fig3(lam=LAMBDA):
    """Courbes théoriques pour Figures 3 / 4."""
    return {
        'gauss':   {'x': np.arange(-2.5, 2.7, 0.001),
                    'y': theoretical_slope('gauss',   np.arange(-2.5, 2.7, 0.001), lam=lam)},
        'student': {'x': np.arange(-2.5, 2.7, 0.001),
                    'y': theoretical_slope('student', np.arange(-2.5, 2.7, 0.001), k=3, lam=lam)},
        'chi2':    {'x': np.arange(0.0,  6.5, 0.001),
                    'y': theoretical_slope('chi2',    np.arange(0.0,  6.5, 0.001), k=3, lam=lam)},
        'mixture': {'x': np.arange(-1.5, 6.5, 0.001),
                    'y': theoretical_slope('mixture', np.arange(-1.5, 6.5, 0.001), lam=lam)},
    }


# =============================================================================
# Courbes théoriques — Figure 5  (f'_p(0) = -2 C*_1 / (π C*_2))
# =============================================================================

def build_theoretical_fig5(lam=LAMBDA, k=3, gamma_aniso=2.0):
    """
    Courbes théoriques de f'_p(0) pour la Figure 5.
    Identiques aux calculs 'second_*_x / second_*_y' dans le R.
    Les x retournés sont sur marges Exp(1).
    """
    th = {}

    # ---- Gaussien ----
    # second_gauss_x = qnorm(pexp(seq(0.5,15,0.01))) → seuils gaussiens
    # puis gauss_to_exp(second_gauss_x) = qexp(pnorm(u)) = -log(1-Φ(u)) = même chose
    t_vals = np.arange(0.5, 15.0, 0.01)
    u_g    = norm.ppf(expon.cdf(t_vals))
    
    # CORRECTION : np.exp(-u_g**2 / 2.0) au lieu de norm.pdf(u_g)
    C1_g   = 0.25 * np.sqrt(lam) * np.exp(-u_g**2 / 2.0)
    C2_g   = 1.0 - norm.cdf(u_g)
    
    th['gauss'] = {'x': t_vals,   # qexp(pnorm(u_g)) = t_vals (identité)
                   'y': -2.0*C1_g/C2_g/np.pi}


    # ---- Student ----
    # second_stud_x = seq(1,18,0.01) — seuils Student non normalisés
    # u (interne) = second_stud_x * sqrt((k-2)/k)
    # C2 = 1-pt(second_stud_x, df=k)
    st_x  = np.arange(1.0, 18.0, 0.01)
    u_in  = st_x * np.sqrt((k-2.0)/k)
    C1_s  = 0.25*np.sqrt(lam)*(1.0 + u_in**2/(k-2.0))**((1.0-k)/2.0)
    C2_s  = 1.0 - tdist.cdf(st_x, df=k)
    th['student'] = {'x': to_exp_scale(st_x, 'student', k=k),
                     'y': -2.0*C1_s/C2_s/np.pi}

    # ---- Chi-2 ----
    # second_chi_x = seq(1,30,0.01) — valeurs chi-2 non normalisées
    ch_x  = np.arange(1.0, 30.0, 0.01)
    C1_c  = (np.sqrt(lam*np.pi)/(2.0**((k+1)/2)*gamma(k/2.0)) *
             ch_x**((k-1)/2.0) * np.exp(-ch_x/2.0))
    C2_c  = 1.0 - chi2dist.cdf(ch_x, df=k)
    th['chi2']    = {'x': to_exp_scale(ch_x, 'chi2', k=k),
                     'y': -2.0*C1_c/C2_c/np.pi}

    # ---- Mélange ----
    # second_mix_x = seq(1,100,0.01), alpha=1 dans compute_C_star
    # Prolongé jusqu'à 12.5 (comme dans le R)
    mx_x  = np.arange(1.0, 100.0, 0.01)
    C1_m  = np.array([compute_C_star(1, xi, alpha=1, lam=lam) for xi in mx_x])
    C2_m  = np.array([compute_C_star(2, xi, alpha=1, lam=lam) for xi in mx_x])
    xm    = to_exp_scale(mx_x, 'mixture', alpha_mix=2)
    ym    = -2.0*C1_m/C2_m/np.pi
    th['mixture'] = {'x': np.append(xm, 12.5),
                     'y': np.append(ym, ym[-1])}

    return th

# =============================================================================
# Simulation — Expérience 1 (Figures 3 / 4)
# =============================================================================

def _simulate_one_field(L_path, gen_x, gen_y, reg_coeff,
                        origin_idx, us, field_type, seed):
    """
    Worker joblib (une réalisation, tous les seuils).

    L est chargée via numpy memmap depuis L_path → lecture partagée,
    aucune copie entre processus. Seuls gen_x, gen_y, reg_coeff et les
    scalaires sont sérialisés (négligeable).

    Paramètres
    ----------
    L_path     : str — chemin du fichier .npy contenant la matrice L
    gen_x/y    : ndarray 1D — coordonnées de la grille
    reg_coeff  : ndarray ou None — coefficients de krigeage
    origin_idx : int — index du point le plus proche de (0,0)
    us         : ndarray — seuils (ordre croissant)
    field_type : str
    seed       : int

    Retourne : ndarray de longueur len(us), NaN si X_i(0) ≤ u_j
    """
    np.random.seed(seed)

    # Chargement de L en lecture seule (memmap → pas de copie mémoire)
    L = np.load(L_path, mmap_mode='r')

    # Reconstruction minimale du générateur
    class _Gen:
        pass
    gen           = _Gen()
    gen.L         = L
    gen.x         = gen_x
    gen.y         = gen_y
    gen.reg_coeff = reg_coeff

    def _generate(cond_val=None):
        z   = np.random.randn(gen.L.shape[1])
        mu  = gen.reg_coeff @ np.asarray(cond_val) if gen.reg_coeff is not None else 0.0
        fld = (gen.L @ z + mu)
        nx, ny = len(gen.x), len(gen.y)
        return RF_from_matrix(fld.reshape((nx, ny), order='F'), gen.x, gen.y)
    gen.generate = _generate

    # Génération du champ
    if field_type == 'gaussian':
        RF = gen.generate()
    elif field_type == 'student':
        RF = student_RF(gen, k=3)
    elif field_type == 'chi2':
        RF = chi2_RF(gen, k=3)
    elif field_type == 'mixture':
        RF = mixture_RF(gen, alpha=2)
    else:
        raise ValueError(field_type)

    val0 = RF['Z'][origin_idx]
    row  = np.full(len(us), np.nan)
    for j, u in enumerate(us):
        if val0 > u:
            ep     = extent_profile(RF, u, max_dist=0.2)
            row[j] = get_extremal_range(ep)
        else:
            break   # us croissant → inutile de continuer
    return row


def run_experiment_1(generator, us, n_fields=5000, field_type='gaussian',
                     n_jobs=1, save_path=None):
    """
    Simule n_fields réalisations et calcule l'extremal range pour chaque seuil.

    Paramètres
    ----------
    generator  : RandomFieldGenerator
    us         : ndarray — seuils
    n_fields   : int     — nombre de réalisations (5000 dans le papier)
    field_type : 'gaussian' | 'student' | 'chi2' | 'mixture'
    n_jobs     : int
        1     → boucle séquentielle simple (défaut, sans overhead)
        -1    → tous les cœurs disponibles (recommandé pour n_fields grand)
        k > 1 → k processus
    save_path  : str ou None — chemin pickle de sauvegarde

    Retourne
    --------
    results : ndarray (n_fields, len(us))
    """
    us = np.sort(us)

    # Index du point de la grille le plus proche de (0,0)
    Xg, Yg    = np.meshgrid(generator.x, generator.y, indexing='ij')
    all_x     = Xg.ravel(order='F')
    all_y     = Yg.ravel(order='F')
    origin_idx = int(np.argmin(all_x**2 + all_y**2))

    # Graines reproductibles
    rng   = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=n_fields)

    desc = f"Exp1 [{field_type}]"
    if n_jobs == 1:
        # ---- Boucle séquentielle : simple, pas d'overhead ----
        rows = []
        for i in tqdm(range(n_fields), desc=desc, unit="field"):
            row = _simulate_one_field(
                None, generator.x, generator.y, generator.reg_coeff,
                origin_idx, us, field_type, int(seeds[i]),
                _L_direct=generator.L
            )
            rows.append(row)

    else:
        # ---- Mode parallèle : L sauvegardée dans un fichier temporaire ----
        # Les workers la chargent via memmap → une seule copie en mémoire
        tmpdir = tempfile.mkdtemp()
        L_path = os.path.join(tmpdir, 'L.npy')
        np.save(L_path, generator.L)
        print(f"  L sauvegardée ({generator.L.nbytes / 1e6:.0f} MB), "
              f"partagée entre workers via memmap.")

        rows = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_simulate_one_field)(
                L_path, generator.x, generator.y, generator.reg_coeff,
                origin_idx, us, field_type, int(seeds[i])
            )
            for i in tqdm(range(n_fields), desc=desc, unit="field")
        )

        import gc
        gc.collect()  
        try:
            os.remove(L_path)
            os.rmdir(tmpdir)
        except PermissionError:
            pass 

    results = np.vstack(rows)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({'results': results, 'us': us, 'field_type': field_type}, f)

    return results


# --- Correction : _simulate_one_field doit accepter L direct ou via fichier ---
# On réécrit la signature pour gérer les deux modes (séquentiel / parallèle)

def _simulate_one_field(L_path, gen_x, gen_y, reg_coeff,
                        origin_idx, us, field_type, seed,
                        _L_direct=None):
    """
    Worker commun aux modes séquentiel et parallèle.
    En mode séquentiel, _L_direct est la matrice L numpy directement.
    En mode parallèle, L est chargée depuis L_path via memmap.
    """
    np.random.seed(seed)

    L = _L_direct if _L_direct is not None else np.load(L_path, mmap_mode='r')

    class _Gen: pass
    gen           = _Gen()
    gen.L         = L
    gen.x         = gen_x
    gen.y         = gen_y
    gen.reg_coeff = reg_coeff

    def _generate(cond_val=None):
        z   = np.random.randn(gen.L.shape[1])
        mu  = gen.reg_coeff @ np.asarray(cond_val) if gen.reg_coeff is not None else 0.0
        return RF_from_matrix(
            (gen.L @ z + mu).reshape((len(gen.x), len(gen.y)), order='F'),
            gen.x, gen.y)
    gen.generate = _generate

    if field_type in ('gaussian', 'gaussian_aniso'): RF = gen.generate()
    elif field_type == 'student': RF = student_RF(gen, k=3)
    elif field_type == 'chi2':    RF = chi2_RF(gen, k=3)
    elif field_type == 'mixture': RF = mixture_RF(gen, alpha=2)
    else: raise ValueError(field_type)

    val0 = RF['Z'][origin_idx]
    row  = np.full(len(us), np.nan)
    for j, u in enumerate(us):
        if val0 > u:
            ep     = extent_profile(RF, u, max_dist=0.2)
            row[j] = get_extremal_range(ep)
        else:
            break
    return row


# =============================================================================
# Construction des données empiriques pour les Figures 3 / 4
# =============================================================================

def create_empirical_fig3(results, us, field_type, n_bootstrap=200, k=3):
    """
    Estime 2 C*_1/C*_2 = lim P(R≤r)/r avec IC à 95% bootstrap.

    Transformations de l'axe x (identiques au R) :
      'gauss'   : us as-is
      'student' : us * sqrt(k)          → seuil Student non normalisé
      'chi2'    : k + us * sqrt(2k)     → valeur chi-2 non normalisée
      'mixture' : us as-is

    Retourne {'u_plot', 'density', 'lower', 'upper'}.
    """
    ds, lo, hi = [], [], []
    for j in range(len(us)):
        x = results[:, j]
        x = x[~np.isnan(x)]
        if len(x) == 0 or not np.isfinite(np.min(x)):
            ds.append(0.0); lo.append(0.0); hi.append(0.0)
        else:
            est = density_at_zero(x)
            boots = [density_at_zero(x[np.random.choice(len(x), len(x), replace=True)])
                     for _ in range(n_bootstrap)]
            ds.append(est)
            lo.append(np.quantile(boots, 0.5 - 0.95/2))
            hi.append(np.quantile(boots, 0.5 + 0.95/2))

    u_plot = np.asarray(us, dtype=float).copy()
    if field_type == 'student':
        u_plot = u_plot * np.sqrt(k)            # * sqrt(3)  comme dans le R
    elif field_type == 'chi2':
        u_plot = k + u_plot * np.sqrt(2.0 * k)  # 3 + u*sqrt(6)  comme dans le R
    # gauss et mixture : pas de transformation

    return {'u_plot': u_plot, 'density': np.array(ds),
            'lower': np.array(lo), 'upper': np.array(hi)}


# =============================================================================
# Simulation — Expérience 2 (Figure 5)
# =============================================================================

def _simulate_one_field_exp2(L_path, gen_x, gen_y, reg_coeff, us, field_type, seed, _L_direct=None):
    np.random.seed(seed)
    L = _L_direct if _L_direct is not None else np.load(L_path, mmap_mode='r')
    class _Gen: pass
    gen = _Gen(); gen.L = L; gen.x = gen_x; gen.y = gen_y; gen.reg_coeff = reg_coeff
    def _generate(cond_val=None):
        return RF_from_matrix((gen.L @ np.random.randn(gen.L.shape[1]) + (gen.reg_coeff @ np.asarray(cond_val) if gen.reg_coeff is not None else 0.0)).reshape((len(gen.x), len(gen.y)), order='F'), gen.x, gen.y)
    gen.generate = _generate

    rows = []
    for u in us:
        if field_type in ('gaussian', 'gaussian_aniso'): RF = gen.generate(cond_val=[gauss_above(u)])
        elif field_type == 'student':                    RF = student_RF(gen, k=3, thresh=u)
        elif field_type == 'chi2':                       RF = chi2_RF(gen, k=3, thresh=u)
        elif field_type == 'mixture':                    RF = mixture_RF(gen, thresh=u, alpha=2)
        
        # CORRECTION : On renvoie l'indicateur binaire de dépassement pour le calcul de chi
        rows.append((RF['Z'] > u).astype(float))
    return rows


def run_experiment_2(generator, us, x_radii, n_fields=100, field_type='gaussian', n_jobs=-1):
    seeds = np.random.default_rng(42).integers(0, 2**31, size=n_fields)
    if n_jobs == 1:
        all_rows = [_simulate_one_field_exp2(None, generator.x, generator.y, generator.reg_coeff, us, field_type, int(seeds[i]), _L_direct=generator.L) for i in tqdm(range(n_fields), desc=f"Exp2 [{field_type}]")]
    else:
        tmpdir = tempfile.mkdtemp(); L_path = os.path.join(tmpdir, 'L.npy'); np.save(L_path, generator.L)
        all_rows = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_simulate_one_field_exp2)(L_path, generator.x, generator.y, generator.reg_coeff, us, field_type, int(seeds[i])) for i in tqdm(range(n_fields), desc=f"Exp2 [{field_type}]"))
        import gc, time; gc.collect()
        try: os.remove(L_path); os.rmdir(tmpdir)
        except PermissionError: pass
    
    # Retourne un tableau 3D : (n_fields, n_thresholds, n_pixels)
    return np.array(all_rows)


def create_empirical_fig5(maps_3d, generator, us, field_type, k=3, n_bootstrap=100):
    # maps_3d est de dimension (n_fields, n_thresholds, n_pixels)
    Xg, Yg = np.meshgrid(generator.x, generator.y, indexing='ij')
    dists = np.sqrt(Xg**2 + Yg**2).flatten()
    
    # Pré-calcul pour la moyenne radiale
    order = np.argsort(dists)
    d_sorted = dists[order]
    uniq_d, inverse, counts = np.unique(d_sorted, return_inverse=True, return_counts=True)
    
    # On limite le rayon de fit (max_dist = 0.2 comme pour extent_profile)
    mask_r = uniq_d <= 0.2
    uniq_d_fit = uniq_d[mask_r]
    
    def calc_slope(chi_map_1d):
        # 1. Calcul de phi(r) via la moyenne radiale
        chi_sorted = chi_map_1d[order]
        chi_radial = np.bincount(inverse, weights=chi_sorted) / counts
        
        # 2. On enlève le point 0.0 car tail_dependence_fun l'ajoute en dur avec le poids 1e6
        # Cela évite de faire crasher la spline avec deux points à x=0
        ep = {'x': uniq_d_fit[1:], 'y': chi_radial[mask_r][1:]}
        
        # 3. Fit propre via la spline lissante (comme écrit dans le rapport !)
        res = tail_dependence_fun(ep)
        
        # 4. CRUCIAL : Application du facteur 3/2 de la formule f'_p(0) = 3/2 * phi'(0)
        slope = (3.0 / 2.0) * res['slope_at_0']
        return slope

    slopes, lo, hi = [], [], []
    for j in range(len(us)):
        chi_maps_j = maps_3d[:, j, :] 
        
        # Estimateur principal sur la moyenne de tous les champs
        mean_chi_map = np.mean(chi_maps_j, axis=0)
        slopes.append(calc_slope(mean_chi_map))
        
        # Bootstrap
        boots = []
        for _ in range(n_bootstrap):
            bidx = np.random.choice(chi_maps_j.shape[0], chi_maps_j.shape[0], replace=True)
            boots.append(calc_slope(np.mean(chi_maps_j[bidx, :], axis=0)))
        lo.append(np.quantile(boots, 0.025))
        hi.append(np.quantile(boots, 0.975))

    us_arr = np.asarray(us, dtype=float)
    if field_type == 'student':      u_exp = to_exp_scale(us_arr * np.sqrt(k), 'student', k=k)
    elif field_type == 'chi2':       u_exp = to_exp_scale(k + us_arr * np.sqrt(2*k), 'chi2', k=k)
    elif field_type == 'mixture':    u_exp = to_exp_scale(us_arr, 'mixture', alpha_mix=2)
    else:                            u_exp = _gauss_to_exp(us_arr)
    
    return {'u_plot': u_exp, 'slopes': np.array(slopes), 'lower': np.array(lo), 'upper': np.array(hi)}

# =============================================================================
# Tracé des figures
# =============================================================================

def _legend_handles():
    return [mlines.Line2D([], [], color=COLORS[_LABEL[ft]],
                           linestyle=LS[_LABEL[ft]], linewidth=LW[_LABEL[ft]],
                           marker=MARKERS[_LABEL[ft]], markersize=5,
                           label=_LABEL[ft])
            for ft in ['gauss', 'student', 'chi2', 'mixture']]


def plot_fig3_or_4(th_dict, emp_dict=None, normalize=False, ylim=(0, 6)):
    """
    Figure 3 (normalize=False) ou Figure 4 (normalize=True).

    th_dict  : sortie de build_theoretical_fig3()
    emp_dict : dict {field_type: sortie de create_empirical_fig3()}, optionnel
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for ft in ['gauss', 'student', 'chi2', 'mixture']:
        label = _LABEL[ft]
        color, ls, lw, mk = COLORS[label], LS[label], LW[label], MARKERS[label]
        th  = th_dict[ft]
        x_th = to_exp_scale(th['x'], ft) if normalize else th['x']
        ax.plot(x_th, th['y'], color=color, linestyle=ls, linewidth=lw)

        if emp_dict and ft in emp_dict:
            emp  = emp_dict[ft]
            x_em = to_exp_scale(emp['u_plot'], ft) if normalize else emp['u_plot']
            ax.errorbar(x_em, emp['density'],
                        yerr=[np.maximum(0, emp['density'] - emp['lower']),
                              np.maximum(0, emp['upper']   - emp['density'])],
                        fmt=mk, color=color, markersize=5,
                        elinewidth=lw, capsize=3, zorder=3)
            

    ax.legend(handles=_legend_handles(), title='Field Type', fontsize=9)
    ax.set_xlabel(r'$-\log(1-p)$' if normalize else 'Threshold', fontsize=12)
    ax.set_ylabel('Slope of CDF', fontsize=12)
    ax.set_title('Figure 4' if normalize else 'Figure 3', fontsize=13)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_fig5(th_dict, emp_dict=None):
    """Figure 5 : f'_p(0) vs -log(1-p)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ft in ['gauss', 'student', 'chi2', 'mixture']:
        label = _LABEL[ft]
        color, ls, lw = COLORS[label], LS[label], LW[label]
        th = th_dict[ft]
        ax.plot(th['x'], th['y'], color=color, linestyle=ls, linewidth=lw)
        if emp_dict and ft in emp_dict:
            emp = emp_dict[ft]
            ax.errorbar(emp['u_plot'], emp['slopes'],
                        yerr=[np.maximum(0, emp['slopes'] - emp['lower']),
                              np.maximum(0, emp['upper']  - emp['slopes'])],
                        fmt=MARKERS[label], color=color, markersize=5,
                        elinewidth=lw, capsize=3, zorder=3)

    ax.legend(handles=_legend_handles(), title='Field Type', fontsize=9)
    ax.set_xlabel(r'$-\log(1-p)$', fontsize=12)
    ax.set_ylabel(r"$f'_p(0)$", fontsize=12)
    ax.set_title('Figure 5', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# EXÉCUTION (GARDÉE À TON FORMAT, AVEC SAUVEGARDE ANISOTROPE)
# =============================================================================

print("Chargement des générateurs...")
with open('gen.pkl', 'rb') as f: gen = pickle.load(f)
print("Générateur isotrope chargé depuis 'gen.pkl'.")



# --- CONFIGURATION DES SEUILS ADAPTÉS ---
US_PER_TYPE_EXP1 = {
    'gaussian':       np.arange(-2.0, 2.6, 0.5),
    'student':        np.arange(-1.0, 2.6, 0.5),
    'chi2':           np.arange(-1.0, 2.6, 1.0),
    'mixture':        np.arange(-1.0, 2.6, 1.0)
}

x_radii = np.linspace(0, 0.2, 50)
US_PER_TYPE_EXP2 = {
    'gaussian':       np.array([norm.ppf(expon.cdf(t)) for t in np.linspace(2, 12, 10)]),
    'student':        np.arange(1.0, 11.0),
    'chi2':           np.arange(1.0, 11.0),
    'mixture':        np.linspace(10, 300, 10)
}

# # # --- AJUSTEMENT DES COURBES THÉORIQUES ---
# th3 = build_theoretical_fig3()
# th5 = build_theoretical_fig5()

# # # Map d'exécution
# ft_map = {'chi2': 'chi2', 'mixture': 'mixture', 'gaussian': 'gauss', 'student': 'student'}
# # 3. Simulation Expérience 1
# emp3 = {}
# for ft, key in ft_map.items():
#     print(f"Running experiment 1 for {ft}...")
#     us_ft = US_PER_TYPE_EXP1[ft]
#     res = run_experiment_1(gen, us_ft, n_fields=5000, field_type=ft, n_jobs=-1)  
#     emp3[key] = create_empirical_fig3(res, us_ft, key)

# # 4. Figures 3 et 4
# fig3, _ = plot_fig3_or_4(th3, emp3, normalize=False)
# fig4, _ = plot_fig3_or_4(th3, emp3, normalize=True)
# fig3.savefig('figure3.png', dpi=150)
# fig4.savefig('figure4.png', dpi=150)

# 5. Simulation Expérience 2 (Figure 5)
# emp5 = {}
# for ft, key in ft_map.items():
#      print(f"Running experiment 2 for {ft}...")
#      us_ft = US_PER_TYPE_EXP2[ft]
#      eps = run_experiment_2(gen, us_ft, x_radii, n_fields=500, field_type=ft, n_jobs=-1)
#      emp5[key] = create_empirical_fig5(eps, gen, us_ft, key)

# fig5, _ = plot_fig5(th5, emp5)
# fig5.savefig('figure5.png', dpi=150)
# print("Calculs et graphiques terminés avec succès.")


COLORS_ANISO = {
    'Gaussian'       : 'red',
    'Aniso (γ=1.5)'  : 'darkred',
    'Aniso (γ=2.0)'  : 'coral',
    'Aniso (γ=3.0)'  : 'tomato',
}
MARKERS_ANISO = {
    'Gaussian'       : 'o',
    'Aniso (γ=1.5)'  : 'v',
    'Aniso (γ=2.0)'  : 's',
    'Aniso (γ=3.0)'  : 'D',
}
LS_ANISO = {
    'Gaussian'       : 'solid',
    'Aniso (γ=1.5)'  : (0, (5, 2)),
    'Aniso (γ=2.0)'  : 'dashed',
    'Aniso (γ=3.0)'  : 'dotted',
}
 
 
# ── Fonctions théoriques ──────────────────────────────────────────────────────
 
def shape_matrix(gamma):
    """
    Construit A = diag(γ, 1/γ) (volume unitaire : det(A) = 1 → pas de biais de
    densité, uniquement un étirement).  On peut passer γ > 1 pour étirer dans la
    direction x.
 
    Si l'on veut modifier la densité de périmètre, utiliser A = diag(a, b) avec
    a · b ≠ 1, par exemple a = b = γ → det(A) = γ² → sqrt(det(A)) = γ.
    """
    return np.array([[gamma, 0.0],
                     [0.0,   gamma]], dtype=float)
    # Pour volume-preserving : np.array([[gamma, 0], [0, 1/gamma]])
 
 
def theoretical_slope_aniso(u_vals, A, lam=None):
    """
    Pente théorique 2C1*(u)/C2*(u) pour le cas gaussien anisotrope.
 
    Paramètres
    ----------
    u_vals : ndarray — seuils gaussiens
    A      : ndarray (2,2) — shape matrix (det(A) donne le facteur de correction)
    lam    : float ou None — second moment spectral λ (None → LAMBDA global)
 
    Retourne
    --------
    slope : ndarray — même longueur que u_vals
    """
    if lam is None:
        lam = LAMBDA
    correction = np.sqrt(np.linalg.det(np.asarray(A, dtype=float)))
    return correction * theoretical_slope('gauss', u_vals, lam=lam)
 
 
def build_theoretical_aniso(gammas=(1.5, 2.0, 3.0), lam=None):
    """
    Construit les courbes théoriques pour plusieurs valeurs de γ.
 
    Paramètre γ : A = diag(γ, γ) → det(A) = γ² → correction = γ.
    Le cas isotrope correspond à γ = 1 (courbe 'Gaussian').
 
    Retourne
    --------
    dict {label: {'x': ..., 'y': ..., 'A': ..., 'gamma': ...}}
    """
    if lam is None:
        lam = LAMBDA
 
    u_vals = np.arange(-2.5, 2.7, 0.001)
    result = {
        'Gaussian': {
            'x'    : u_vals,
            'y'    : theoretical_slope('gauss', u_vals, lam=lam),
            'A'    : np.eye(2),
            'gamma': 1.0,
        }
    }
    for gamma in gammas:
        A     = shape_matrix(gamma)
        label = f'Aniso (γ={gamma})'
        result[label] = {
            'x'    : u_vals,
            'y'    : theoretical_slope_aniso(u_vals, A, lam=lam),
            'A'    : A,
            'gamma': gamma,
        }
    return result
 
 
# ── Création des données empiriques ──────────────────────────────────────────
 
def create_empirical_aniso(results, us):
    """
    Wrapper de create_empirical_fig3 pour le champ anisotrope (= gaussien).
    L'axe x n'a pas de transformation (identique au cas gaussien).
    """
    return create_empirical_fig3(results, us, field_type='gauss')
 
 
# ── Figure comparative isotrope / anisotrope ─────────────────────────────────
 
def plot_fig_aniso(th_aniso, emp_aniso=None, normalize=False, ylim=(0, 8)):
    """
    Trace la pente 2C1*/C2* (ou sur marges Exp(1)) pour le cas isotrope
    et plusieurs cas anisotropes.
 
    Paramètres
    ----------
    th_aniso   : sortie de build_theoretical_aniso()
    emp_aniso  : dict {label: sortie de create_empirical_aniso()}, optionnel
    normalize  : bool — True → axe x en marges Exp(1)
    ylim       : tuple
    """
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for label, data in th_aniso.items():
        color  = COLORS_ANISO.get(label, 'gray')
        ls     = LS_ANISO.get(label, 'solid')
        marker = MARKERS_ANISO.get(label, 'o')
 
        x_th = (_gauss_to_exp(data['x']) if normalize else data['x'])
        ax.plot(x_th, data['y'], color=color, linestyle=ls,
                linewidth=1.4, label=label + ' (théorique)')
 
        if emp_aniso and label in emp_aniso:
            emp = emp_aniso[label]
            x_em = (_gauss_to_exp(emp['u_plot']) if normalize
                    else emp['u_plot'])
            ax.errorbar(
                x_em, emp['density'],
                yerr=[np.maximum(0, emp['density'] - emp['lower']),
                      np.maximum(0, emp['upper']   - emp['density'])],
                fmt=marker, color=color, markersize=5,
                elinewidth=1.0, capsize=3, zorder=3,
                label=label + ' (empirique)',
            )
 
    xlabel = r'$-\log(1-p)$' if normalize else 'Threshold $u$'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r'$2C^*_1(u)\,/\,C^*_2(u)$ — Slope of CDF', fontsize=11)
    title  = ('Pente de la portée extrémale — cas gaussien anisotrope'
              + (' (marges Exp(1))' if normalize else ''))
    ax.set_title(title, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
 
 
def plot_fig_aniso_ratio(th_aniso, ylim=(0, 4)):
    """
    Trace le ratio Slope_aniso / Slope_gauss pour vérifier qu'il vaut sqrt(det(A)).
    Utile comme validation numérique de la correction théorique.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
 
    gauss_y = th_aniso['Gaussian']['y']
    u_vals  = th_aniso['Gaussian']['x']
    mask    = gauss_y > 0
 
    for label, data in th_aniso.items():
        if label == 'Gaussian':
            continue
        gamma  = data['gamma']
        color  = COLORS_ANISO.get(label, 'gray')
        ratio  = np.where(mask, data['y'] / gauss_y, np.nan)
        ax.plot(u_vals, ratio, color=color, linewidth=1.4,
                label=fr'{label} — ratio attendu : $\gamma={gamma:.1f}$')
        # Valeur théorique attendue = sqrt(det(A)) = gamma
        ax.axhline(gamma, color=color, linestyle=':', linewidth=0.8)
 
    ax.set_xlabel('Threshold $u$', fontsize=12)
    ax.set_ylabel(r'$\mathrm{Slope}_\mathrm{aniso}(u)\;/\;\mathrm{Slope}_\mathrm{gauss}(u)$',
                  fontsize=10)
    ax.set_title(r'Vérification : ratio $=\sqrt{\det(A)}=\gamma$', fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax