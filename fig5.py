"""
fix_figure5.py
==============
Correction du bug de l'Expérience 2 (Figure 5).

DIAGNOSTIC DU BUG
-----------------
gen.pkl a été créé avec RandomFieldGenerator(...) sans cond_sites,
donc gen.reg_coeff = None.
Dans _simulate_one_field_exp2, l'appel
    gen.generate(cond_val=[gauss_above(u)])
ignore complètement cond_val → X(0) ~ N(0,1) libre, PAS conditionné sur X(0) > u.
Les chi-maps obtenus sont donc faux pour TOUS les types de champs.

FIX : Conditioning by kriging (Journel & Huijbregts 1978)
----------------------------------------------------------
Pour conditionner X sur X(0) = z0 SANS recalculer gen.pkl :

    X_cond(s) = X(s) + sigma12(s) * (z0 - X(0))

où sigma12(s) = Cov(X(s), X(0)) = (L @ L.T)[:, origin_idx].

Vérification :
    X_cond(0) = X(0) + sigma12(0) * (z0 - X(0))
              = X(0) + 1 * (z0 - X(0))     [car Var(X(0)) = 1 → sigma12(0) = 1]
              = z0  ✓

Complexité : calcul de sigma12 en O(n²) UNE SEULE FOIS (matrice-vecteur),
puis conditioning en O(n) par réalisation.
"""

import os
import tempfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import norm, chi2 as chi2dist, t as tdist, expon
from scipy.special import gamma
from scipy.integrate import quad
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports depuis les fichiers existants
# ---------------------------------------------------------------------------
from useful_functions import (
    RF_from_matrix, gauss_above, rpareto,
    extent_profile, get_extremal_range, tail_dependence_fun,
)
from extremal_range import (
    LAMBDA, MATERN_NU,
    COLORS, MARKERS, LS, LW, _LABEL,
    pmixture, marginal_cdf, to_exp_scale, _gauss_to_exp,
    build_theoretical_fig5, plot_fig5,
)


# ---------------------------------------------------------------------------
# Préparation : calcul de sigma12 depuis gen.pkl existant
# ---------------------------------------------------------------------------

def compute_sigma12(generator):
    """
    Calcule le vecteur de krigeage sigma12 = Cov(X(s), X(0)) pour tous les
    points s de la grille.

    Utilise l'identité :
        Cov(X(s), X(0)) = (L @ L.T)[:, origin_idx]
                        = L @ L[origin_idx, :]

    Paramètre
    ---------
    generator : RandomFieldGenerator (avec reg_coeff = None ou non)

    Retourne
    --------
    sigma12    : ndarray (n_pixels,)  — vecteur de krigeage
    origin_idx : int                  — index du point le plus proche de (0,0)
    """
    Xg, Yg    = np.meshgrid(generator.x, generator.y, indexing='ij')
    all_x     = Xg.ravel(order='F')
    all_y     = Yg.ravel(order='F')
    origin_idx = int(np.argmin(all_x**2 + all_y**2))

    # Produit matrice-vecteur : O(n^2) mais une seule fois, rapide avec BLAS
    sigma12 = generator.L @ generator.L[origin_idx, :]

    # Vérification : sigma12[origin_idx] doit valoir 1 (= Var(X(0)))
    assert abs(sigma12[origin_idx] - 1.0) < 1e-6, \
        f"sigma12[origin] = {sigma12[origin_idx]:.6f} ≠ 1 (variance non unitaire ?)"

    return sigma12, origin_idx


# ---------------------------------------------------------------------------
# Worker corrigé pour l'Expérience 2
# ---------------------------------------------------------------------------

def _worker_exp2_fixed(L_path, gen_x, gen_y, sigma12, origin_idx,
                       us, field_type, seed, _L_direct=None):
    """
    Worker joblib corrigé.

    Utilise le conditioning by kriging pour garantir X(0) > u :
        X_cond(s) = X(s) + sigma12(s) * (z0 - X(0))
    avec z0 tiré au-dessus du seuil u.

    Retourne
    --------
    rows : list of ndarray (n_pixels,) — indicatrice {X_cond > u} pour chaque u
    """
    np.random.seed(seed)
    L = _L_direct if _L_direct is not None else np.load(L_path, mmap_mode='r')
    nx, ny = len(gen_x), len(gen_y)

    # ── Fonctions de base ────────────────────────────────────────────────────

    def uncond():
        """Génère un champ gaussien non conditionné."""
        return L @ np.random.randn(L.shape[1])

    def condition(field, z0):
        """
        Conditioning by kriging : ajuste le champ pour que field[origin_idx] = z0.
        X_cond(s) = X(s) + sigma12(s) * (z0 - X(origin))
        """
        return field + sigma12 * (z0 - field[origin_idx])

    def to_RF(field):
        return RF_from_matrix(
            field.reshape((nx, ny), order='F'), gen_x, gen_y)

    # ── Simulation par seuil ─────────────────────────────────────────────────
    rows = []

    for u in us:

        # ── Champ gaussien ───────────────────────────────────────────────────
        if field_type in ('gaussian', 'gaussian_aniso'):
            z0  = gauss_above(u)                 # tirage au-dessus de u
            RF  = to_RF(condition(uncond(), z0))

        # ── Champ Student ────────────────────────────────────────────────────
        elif field_type == 'student':
            k = 3
            # Rejection sampling sur la valeur en 0
            while True:
                conds = np.random.randn(k + 1)
                val0  = (conds[k] / np.sqrt(np.sum(conds[:k]**2) / k)
                         * np.sqrt((k - 2.) / k))
                if val0 > u:
                    break
            # G4 conditionné sur G4(0) = conds[k]
            f4   = condition(uncond(), conds[k])
            # G1,G2,G3 conditionnés sur Gi(0) = conds[i]
            denom = np.zeros(L.shape[0])
            for i in range(k):
                fi     = condition(uncond(), conds[i])
                denom += fi ** 2
            field_s = f4 / np.sqrt(denom / k) * np.sqrt((k - 2.) / k)
            RF      = to_RF(field_s)

        # ── Champ chi-deux ───────────────────────────────────────────────────
        elif field_type == 'chi2':
            k = 3
            while True:
                conds = np.random.randn(k)
                val0  = (np.sum(conds**2) - k) / np.sqrt(2. * k)
                if val0 > u:
                    break
            total = np.zeros(L.shape[0])
            for i in range(k):
                fi     = condition(uncond(), conds[i])
                total += fi ** 2
            field_c = (total - k) / np.sqrt(2. * k)
            RF      = to_RF(field_c)

        # ── Champ mélange ────────────────────────────────────────────────────
        elif field_type == 'mixture':
            alpha = 2.0
            while True:
                w0  = np.random.randn()
                lam = np.random.uniform() ** (-1. / alpha)
                if w0 * lam >= u:
                    break
            f_g   = condition(uncond(), w0)
            field_m = f_g * lam
            RF      = to_RF(field_m)

        else:
            raise ValueError(f"field_type inconnu : {field_type}")

        rows.append((RF['Z'] > u).astype(float))

    return rows


# ---------------------------------------------------------------------------
# run_experiment_2 corrigé
# ---------------------------------------------------------------------------

def run_experiment_2_fixed(generator, us, n_fields=3000,
                           field_type='gaussian', n_jobs=1):
    """
    Version corrigée de run_experiment_2 utilisant le conditioning by kriging.

    Différence avec la version originale
    -------------------------------------
    • sigma12 est précalculé UNE FOIS depuis generator.L (O(n²), < 1s)
    • Chaque worker utilise X_cond = X + sigma12*(z0 - X(0)) pour garantir
      X(0) = z0 > u exactement, pour tous les types de champs.
    • gen.reg_coeff n'est PAS nécessaire.

    Paramètres
    ----------
    generator  : RandomFieldGenerator (reg_coeff peut être None)
    us         : ndarray — seuils
    n_fields   : int     — nombre de réalisations
    field_type : str     — 'gaussian' | 'student' | 'chi2' | 'mixture'
    n_jobs     : int     — 1 = séquentiel, -1 = tous les cœurs

    Retourne
    --------
    ndarray (n_fields, n_thresholds, n_pixels)
    """
    us = np.asarray(us)

    print(f"  Précalcul de sigma12 depuis gen.L ({generator.L.shape})...")
    sigma12, origin_idx = compute_sigma12(generator)
    print(f"  sigma12 calculé. origin_idx={origin_idx}, "
          f"sigma12[origin]={sigma12[origin_idx]:.6f}")

    seeds = np.random.default_rng(42).integers(0, 2**31, size=n_fields)
    desc  = f"Exp2_fixed [{field_type}]"

    if n_jobs == 1:
        all_rows = [
            _worker_exp2_fixed(
                None, generator.x, generator.y, sigma12, origin_idx,
                us, field_type, int(seeds[i]),
                _L_direct=generator.L,
            )
            for i in tqdm(range(n_fields), desc=desc, unit='field')
        ]
    else:
        tmpdir = tempfile.mkdtemp()
        L_path = os.path.join(tmpdir, 'L.npy')
        np.save(L_path, generator.L)
        print(f"  L sauvegardée ({generator.L.nbytes/1e6:.0f} MB) pour workers.")

        all_rows = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_worker_exp2_fixed)(
                L_path, generator.x, generator.y, sigma12, origin_idx,
                us, field_type, int(seeds[i]),
            )
            for i in tqdm(range(n_fields), desc=desc, unit='field')
        )

        try:
            os.remove(L_path)
            os.rmdir(tmpdir)
        except Exception:
            pass

    return np.array(all_rows)   # (n_fields, n_thresholds, n_pixels)


# ---------------------------------------------------------------------------
# create_empirical_fig5 (identique à l'original, reprise ici pour clarté)
# ---------------------------------------------------------------------------

def create_empirical_fig5_fixed(maps_3d, generator, us, field_type, k=3,
                                n_bootstrap=100):
    """
    Estime f'_p(0) = (3/2) * phi'(0) avec IC bootstrap 95%.

    maps_3d : (n_fields, n_thresholds, n_pixels)

    Deux bugs corrigés vs la version précédente
    -------------------------------------------
    BUG 1 — Facteur 3/2 oublié :
        La formule exacte est f'_p(0) = (3/2) * phi'(0).
        Preuve : chi_p(r) = phi(r) + r/2 * phi'(r)
                 → chi_p'(0) = phi'(0) + 1/2*phi'(0) = 3/2 * phi'(0).
        L'ancienne version renvoyait la pente brute de chi_radial ≈ phi'(0),
        sous-estimant f'_p(0) d'un facteur 3/2 ≈ 33% d'erreur systématique.

    BUG 2 — Dérivation d'une fonction en escalier par polyfit :
        chi_radial est discontinue (grille discrète) → np.polyfit instable.
        Correction : calculer phi(r) = moyenne CUMULATIVE de chi_p dans B(0,r),
        puis ajuster une spline lissante via tail_dependence_fun (avec poids
        élevé en r=0 pour imposer phi(0)=1), et lire phi'(0) depuis la spline.
    """
    Xg, Yg = np.meshgrid(generator.x, generator.y, indexing='ij')
    dists   = np.sqrt(Xg**2 + Yg**2).flatten()

    # Pré-calcul des indices de tri (identique pour tous les seuils et boots)
    order    = np.argsort(dists)
    d_sorted = dists[order]
    uniq_d, inverse, counts = np.unique(d_sorted, return_inverse=True,
                                        return_counts=True)
    cum_n = np.cumsum(counts)           # nombre de pixels dans B(0, r)
    mask_r0 = uniq_d > 0               # exclure r=0 pour la spline

    def calc_slope(chi_map_1d):
        """
        Calcule f'_p(0) = (3/2) * phi'(0) à partir d'une chi_map.

        Étapes :
        1. Trier chi_map par distance croissante
        2. Calculer phi(r) = moyenne cumulative de chi dans B(0, r)
           → phi(r) = (sum chi_i pour ||s_i|| ≤ r) / (# pixels dans B(0,r))
        3. Ajuster une spline lissante sur phi(r) via tail_dependence_fun
           (force phi(0)=1 par poids élevé, amortit le bruit)
        4. Retourner (3/2) * phi'(0) depuis la spline
        """
        # Étape 1-2 : phi(r) cumulatif
        chi_s    = chi_map_1d[order]
        chi_sum  = np.bincount(inverse, weights=chi_s)
        cum_chi  = np.cumsum(chi_sum)
        phi_at_r = cum_chi / cum_n          # phi(r) pour chaque rayon distinct

        # Étape 3 : spline lissante avec tail_dependence_fun
        # (poids 1e6 en r=0 pour forcer phi(0)=1, lissage s=0.95*n)
        ep = {'x': uniq_d[mask_r0], 'y': phi_at_r[mask_r0]}
        result = tail_dependence_fun(ep)

        # Étape 4 : f'_p(0) = (3/2) * phi'(0)
        phi_prime_0 = result['slope_at_0']   # dérivée de la spline en 0
        return 1.5 * phi_prime_0

    slopes, lo, hi = [], [], []
    for j in range(len(us)):
        chi_maps_j = maps_3d[:, j, :]           # (n_fields, n_pixels)
        mean_chi   = np.mean(chi_maps_j, axis=0)
        slopes.append(calc_slope(mean_chi))

        boots = [
            calc_slope(np.mean(
                chi_maps_j[np.random.choice(chi_maps_j.shape[0],
                                            chi_maps_j.shape[0], replace=True)],
                axis=0,
            ))
            for _ in range(n_bootstrap)
        ]
        lo.append(np.quantile(boots, 0.025))
        hi.append(np.quantile(boots, 0.975))

    us_arr = np.asarray(us, float)
    if   field_type == 'student':  u_exp = to_exp_scale(us_arr * np.sqrt(k), 'student', k=k)
    elif field_type == 'chi2':     u_exp = to_exp_scale(k + us_arr*np.sqrt(2*k), 'chi2', k=k)
    elif field_type == 'mixture':  u_exp = to_exp_scale(us_arr, 'mixture', alpha_mix=2)
    else:                          u_exp = _gauss_to_exp(us_arr)

    return {'u_plot': u_exp, 'slopes': np.array(slopes),
            'lower': np.array(lo), 'upper': np.array(hi)}


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Test rapide N=200 (au lieu de 3000)')
    parser.add_argument('--n_fields', type=int, default=3000)
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    N_FIELDS = 200 if args.quick else args.n_fields

    # ── Chargement du générateur ──────────────────────────────────────────────
    print("Chargement de gen.pkl...")
    with open('gen.pkl', 'rb') as f:
        gen = pickle.load(f)
    print(f"  gen.reg_coeff = {gen.reg_coeff}  "
          f"({'BUG : pas de conditionnement' if gen.reg_coeff is None else 'OK'})")
    print(f"  gen.L.shape = {gen.L.shape}")

    # ── Seuils (identiques à l'original) ─────────────────────────────────────
    US_EXP2 = {
        'gaussian': np.array([norm.ppf(expon.cdf(t))
                               for t in np.linspace(2, 12, 10)]),
        'student':  np.arange(1.0, 11.0),
        'chi2':     np.arange(1.0, 11.0),
        'mixture':  np.linspace(10, 300, 10),
    }

    ft_map = {'gaussian': 'gauss', 'student': 'student',
              'chi2': 'chi2',      'mixture': 'mixture'}

    # ── Simulation corrigée ───────────────────────────────────────────────────
    emp5 = {}
    for ft, key in ft_map.items():
        print(f"\n── {ft} ({'rapide' if args.quick else 'complet'}) ──")
        us_ft = US_EXP2[ft]
        maps  = run_experiment_2_fixed(gen, us_ft, n_fields=N_FIELDS,
                                       field_type=ft, n_jobs=args.n_jobs)
        emp5[key] = create_empirical_fig5_fixed(maps, gen, us_ft, ft)

    # ── Courbes théoriques ────────────────────────────────────────────────────
    th5 = build_theoretical_fig5()

    # ── Figure 5 ─────────────────────────────────────────────────────────────
    fig5, _ = plot_fig5(th5, emp5)
    fig5.savefig('figure5_fixed.png', dpi=150)
    print("\nFigure sauvegardée : figure5_fixed.png")
    plt.show()