"""
run_aniso.py
============
Script de simulation et tracé pour le cas gaussien anisotrope.

Hypothèse : les deux fichiers suivants sont dans le même répertoire :
    - useful_functions.py   (avec AnisotropicRFGenerator ajouté)
    - extremal_range.py     (avec les additions anisotropes ajoutées)
    - gen.pkl               (générateur isotrope pré-calculé, si disponible)

Ou bien lancer tel quel : le générateur isotrope est recalculé à la volée si
gen.pkl est absent (long pour GRID_N=121, rapide pour GRID_N=41).

Usage
-----
    python run_aniso.py               # simulation complète N=1000
    python run_aniso.py --quick       # test rapide N=100, grille 41×41
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Import des fonctions existantes
# ---------------------------------------------------------------------------
from useful_functions import (
    RF_from_matrix, cov_model,
    extent_profile, get_extremal_range,
    density_at_zero, gauss_above,
    tail_dependence_fun,
)
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# AnisotropicRFGenerator  (copie ici pour éviter de modifier useful_functions)
# ---------------------------------------------------------------------------

class AnisotropicRFGenerator:
    """
    Champ gaussien anisotrope : ρ_aniso(h) = ρ_iso( sqrt(h^T A h) ).
    La distance de Mahalanobis est obtenue en transformant la grille par A^{1/2}.

    Lien théorique :
        Λ = λ·A  (λ = second moment spectral Matern)
        C1*(u)   = sqrt(det(Λ)) · (1/4) · exp(−u²/2) = λ·sqrt(det(A)) · (1/4) · exp(−u²/2)
        Slope(u) = sqrt(det(A)) · Slope_isotrope(u)
    """

    def __init__(self, x, y, cov_fun, A, method='eig'):
        self.x  = np.asarray(x)
        self.y  = np.asarray(y)
        nx, ny  = len(self.x), len(self.y)

        Xg, Yg = np.meshgrid(self.x, self.y, indexing='ij')
        grid   = np.column_stack([Xg.ravel(order='F'), Yg.ravel(order='F')])

        # Cholesky de A : A = A_half @ A_half.T
        A_half = np.linalg.cholesky(np.asarray(A, dtype=float))
        grid_t = grid @ A_half.T           # grille dans l'espace de Mahalanobis

        cov_mat = cov_fun(cdist(grid_t, grid_t))

        if method == 'chol':
            cov_mat += 1e-10 * np.eye(grid.shape[0])
            self.L   = np.linalg.cholesky(cov_mat)
        else:
            vals, vecs = np.linalg.eigh(cov_mat)
            vals       = np.maximum(vals, 0.0)
            self.L     = vecs @ np.diag(np.sqrt(vals))

        self.reg_coeff = None   # pas de conditionnement spatial ici

    def generate(self, cond_val=None):
        z     = np.random.randn(self.L.shape[1])
        field = self.L @ z
        nx, ny = len(self.x), len(self.y)
        return RF_from_matrix(field.reshape((nx, ny), order='F'), self.x, self.y)


# ---------------------------------------------------------------------------
# Paramètres globaux
# ---------------------------------------------------------------------------

MATERN_NU = 2.5
LAMBDA    = MATERN_NU / (MATERN_NU - 1)   # ≈ 1.667

# Matrices de forme A = γ·I₂, ratio = sqrt(γ) (half-SA convention)
# Pour ratios visibles : sqrt(γ) ∈ {1, 1.41, 2, 3} → γ ∈ {1, 2, 4, 9}
GAMMAS = [1.0, 2.0, 4.0, 9.0]   # ratios attendus : 1, √2≈1.41, 2, 3

COLORS = {
    1.0: 'red',
    2.0: 'darkorange',
    4.0: 'steelblue',
    9.0: 'purple',
}
LS = {1.0: 'solid', 2.0: 'dashed', 4.0: 'dashdot', 9.0: 'dotted'}


# ---------------------------------------------------------------------------
# Formules théoriques
# ---------------------------------------------------------------------------

def theoretical_slope_gauss(u_vals, lam=LAMBDA):
    """2C1*(u)/C2*(u) pour le cas isotrope."""
    C1 = 0.25 * np.sqrt(lam) * np.exp(-u_vals**2 / 2.0)
    C2 = 1.0  - norm.cdf(u_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(C2 > 0, 2.0 * C1 / C2, np.nan)


def theoretical_slope_aniso(u_vals, gamma, lam=LAMBDA):
    """
    Slope anisotrope pour A = γ·I₂, convention half-surface-area density.

    Rho_aniso(h) = rho_iso(sqrt(γ)*||h||) → λ_eff = γ*LAMBDA
    C1_eff*(u)   = (1/4)*sqrt(γ*λ)*exp(-u²/2) = sqrt(γ) * C1_iso*(u)
    Ratio        = sqrt(γ)   (et NON γ)

    Note : la formule (4.40) du rapport donne ratio = γ = sqrt(det(A))
    car elle utilise la convention Adler-Taylor (LKC plein, pas half-SA).
    La convention du CODE est half-SA → ratio = sqrt(γ).
    Pour des courbes théoriques cohérentes avec les simulations, on utilise sqrt(γ).
    """
    return np.sqrt(gamma) * theoretical_slope_gauss(u_vals, lam=lam)


def gauss_to_exp(u_arr):
    """Transforme un seuil gaussien en marge Exp(1) : −log(1−Φ(u))."""
    return -np.log(np.maximum(1.0 - norm.cdf(np.asarray(u_arr)), 1e-300))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_aniso_experiment(generator, us, n_fields=1000, seed=0):
    """
    Simule n_fields réalisations et calcule l'extremal range pour chaque seuil.

    Retourne
    --------
    results : ndarray (n_fields, len(us)) — NaN si X(0) ≤ u
    """
    us      = np.sort(us)
    rng     = np.random.default_rng(seed)
    seeds   = rng.integers(0, 2**31, size=n_fields)

    # Index du point le plus proche de l'origine
    Xg, Yg     = np.meshgrid(generator.x, generator.y, indexing='ij')
    origin_idx = int(np.argmin(Xg.ravel(order='F')**2 + Yg.ravel(order='F')**2))

    results = np.full((n_fields, len(us)), np.nan)

    for i in range(n_fields):
        np.random.seed(int(seeds[i]))
        RF   = generator.generate()
        val0 = RF['Z'][origin_idx]
        for j, u in enumerate(us):
            if val0 > u:
                ep           = extent_profile(RF, u, max_dist=0.3)
                results[i, j] = get_extremal_range(ep)
            else:
                break   # us triés par ordre croissant
        if (i + 1) % max(1, n_fields // 10) == 0:
            print(f"  {i+1}/{n_fields} réalisations simulées")

    return results


def empirical_slope(results, us, n_bootstrap=200):
    """
    Estime lim_{r→0} P(R≤r)/r avec IC bootstrap à 95%.

    Retourne
    --------
    dict {'u_plot', 'density', 'lower', 'upper'}
    """
    ds, lo, hi = [], [], []
    for j in range(len(us)):
        x = results[:, j]
        x = x[np.isfinite(x)]
        if len(x) < 5:
            ds.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        est   = density_at_zero(x)
        boots = [density_at_zero(x[np.random.choice(len(x), len(x), replace=True)])
                 for _ in range(n_bootstrap)]
        ds.append(est)
        lo.append(np.nanquantile(boots, 0.025))
        hi.append(np.nanquantile(boots, 0.975))
    return {
        'u_plot' : np.asarray(us, float),
        'density': np.array(ds),
        'lower'  : np.array(lo),
        'upper'  : np.array(hi),
    }


# ---------------------------------------------------------------------------
# Figure 1 : pente sur échelle naturelle (style Figure 3 du papier)
# ---------------------------------------------------------------------------

def plot_aniso_fig3(th_dict, emp_dict=None, ylim=(0, 8)):
    """Slope en fonction du seuil u (échelle naturelle)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    u_th    = np.arange(-2.5, 2.7, 0.001)

    for gamma in GAMMAS:
        label = f'$\\gamma={gamma}$  ($\\sqrt{{\\gamma}}={np.sqrt(gamma):.2f}$)' + (' (isotrope)' if gamma == 1.0 else '')
        y_th  = theoretical_slope_aniso(u_th, gamma)
        ax.plot(u_th, y_th, color=COLORS[gamma], linestyle=LS[gamma],
                linewidth=1.4, label=label)

        if emp_dict and gamma in emp_dict:
            emp = emp_dict[gamma]
            mask = np.isfinite(emp['density'])
            ax.errorbar(
                emp['u_plot'][mask], emp['density'][mask],
                yerr=[np.maximum(0, (emp['density'] - emp['lower'])[mask]),
                      np.maximum(0, (emp['upper']   - emp['density'])[mask])],
                fmt=('o' if gamma == 1.0 else 's'),
                color=COLORS[gamma], markersize=5,
                elinewidth=0.9, capsize=3, zorder=4,
            )

    ax.set_xlabel('Threshold $u$', fontsize=12)
    ax.set_ylabel(r'$2C^*_1(u)\,/\,C^*_2(u)$  —  Slope of CDF', fontsize=11)
    ax.set_title('Portée extrémale — cas gaussien anisotrope  '
                 r'[$A = \gamma I_2$,  $\mathrm{Slope} = \gamma \cdot \mathrm{Slope}_\mathrm{iso}$]',
                 fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, title='Shape matrix')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2 : idem sur marges Exp(1) (style Figure 4 du papier)
# ---------------------------------------------------------------------------

def plot_aniso_fig4(th_dict, emp_dict=None, ylim=(0, 8)):
    """Slope en fonction de −log(1−p) (marges Exp(1))."""
    fig, ax = plt.subplots(figsize=(8, 5))
    u_th    = np.arange(-2.5, 4.0, 0.001)
    x_th    = gauss_to_exp(u_th)

    for gamma in GAMMAS:
        label = f'$\\gamma={gamma}$  ($\\sqrt{{\\gamma}}={np.sqrt(gamma):.2f}$)' + (' (isotrope)' if gamma == 1.0 else '')
        y_th  = theoretical_slope_aniso(u_th, gamma)
        ax.plot(x_th, y_th, color=COLORS[gamma], linestyle=LS[gamma],
                linewidth=1.4, label=label)

        if emp_dict and gamma in emp_dict:
            emp  = emp_dict[gamma]
            x_em = gauss_to_exp(emp['u_plot'])
            mask = np.isfinite(emp['density'])
            ax.errorbar(
                x_em[mask], emp['density'][mask],
                yerr=[np.maximum(0, (emp['density'] - emp['lower'])[mask]),
                      np.maximum(0, (emp['upper']   - emp['density'])[mask])],
                fmt=('o' if gamma == 1.0 else 's'),
                color=COLORS[gamma], markersize=5,
                elinewidth=0.9, capsize=3, zorder=4,
            )

    ax.set_xlabel(r'$-\log(1-p)$', fontsize=12)
    ax.set_ylabel(r'$2C^*_1(u)\,/\,C^*_2(u)$  —  Slope of CDF', fontsize=11)
    ax.set_title('Portée extrémale — cas anisotrope (marges Exp(1))', fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, title='Shape matrix')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 3 : ratio empirique / théorique (validation de la correction √det(A))
# ---------------------------------------------------------------------------

def plot_aniso_ratio(th_dict, emp_dict, ylim=(0.5, 4.0)):
    """
    Vérifie que Slope_aniso(u) / Slope_iso(u) → γ = sqrt(det(A)).
    Les lignes pointillées horizontales indiquent les valeurs théoriques attendues.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    u_th    = np.arange(-1.5, 2.7, 0.001)
    gauss_y = theoretical_slope_gauss(u_th)

    for gamma in GAMMAS:
        if gamma == 1.0:
            continue
        color = COLORS[gamma]
        # Rapport théorique (= γ constant)
        ax.axhline(gamma, color=color, linestyle=':', linewidth=1.0,
                   label=fr'Attendu $\gamma={gamma}$')
        # Rapport empirique
        if emp_dict and gamma in emp_dict and 1.0 in emp_dict:
            emp_a = emp_dict[gamma]
            emp_g = emp_dict[1.0]
            mask  = (np.isfinite(emp_a['density']) &
                     np.isfinite(emp_g['density']) &
                     (emp_g['density'] > 0))
            ratio = emp_a['density'][mask] / emp_g['density'][mask]
            ax.scatter(emp_a['u_plot'][mask], ratio,
                       color=color, marker='s', s=25, zorder=4,
                       label=fr'Empirique $\gamma={gamma}$')

    ax.set_xlabel('Threshold $u$', fontsize=12)
    ax.set_ylabel(r'$\mathrm{Slope}_{\mathrm{aniso}}\;/\;\mathrm{Slope}_{\mathrm{iso}}$',
                  fontsize=11)
    ax.set_title(r'Validation de $\mathrm{Slope}_\mathrm{aniso} = \gamma \cdot \mathrm{Slope}_\mathrm{iso}$',
                 fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def compute_sigma12_aniso(gen):
    """Calcule le vecteur de krigeage sigma12 depuis gen.L."""
    Xg, Yg    = np.meshgrid(gen.x, gen.y, indexing='ij')
    origin_idx = int(np.argmin(Xg.ravel(order='F')**2 + Yg.ravel(order='F')**2))
    sigma12    = gen.L @ gen.L[origin_idx, :]
    return sigma12, origin_idx


def theoretical_fp0_aniso(u_vals, gamma, lam=LAMBDA):
    """
    Courbe théorique f'_p(0) = -2*C1*(u) / (π*C2*(u)) pour le cas anisotrope.
    Correction : sqrt(γ) (même convention half-SA que pour la Slope).
    """
    u   = np.asarray(u_vals)
    C1  = 0.25 * np.sqrt(lam) * np.exp(-u**2 / 2.0)
    C2  = 1.0 - norm.cdf(u)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(C2 > 0, -2.0 * np.sqrt(gamma) * C1 / (np.pi * C2), np.nan)


def run_exp2_aniso(gen, us, n_fields=1000, seed=0):
    """
    Expérience 2 pour un champ gaussien anisotrope.
    Utilise le conditioning by kriging pour garantir X(0) > u.

    Retourne maps_3d : (n_fields, n_thresholds, n_pixels)
    """
    us = np.asarray(us)

    # Précalcul des coefficients de krigeage (une fois)
    sigma12, origin_idx = compute_sigma12_aniso(gen)

    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_fields)

    Xg, Yg = np.meshgrid(gen.x, gen.y, indexing='ij')
    nx, ny  = len(gen.x), len(gen.y)

    all_rows = []
    for i in range(n_fields):
        np.random.seed(int(seeds[i]))
        rows = []
        for u in us:
            # Tirer z0 > u (rejection sampling sur la marge gaussienne)
            z0    = gauss_above(u)
            # Champ non conditionné
            field = gen.L @ np.random.randn(gen.L.shape[1])
            # Conditioning by kriging : X_cond(0) = z0 exactement
            field_cond = field + sigma12 * (z0 - field[origin_idx])
            RF = RF_from_matrix(field_cond.reshape((nx, ny), order='F'),
                                gen.x, gen.y)
            rows.append((RF['Z'] > u).astype(float))
        all_rows.append(rows)
        if (i + 1) % max(1, n_fields // 5) == 0:
            print(f"    {i+1}/{n_fields}")

    return np.array(all_rows)   # (n_fields, n_thresholds, n_pixels)


def empirical_fp0(maps_3d, gen, us):
    """
    Estime f'_p(0) = (3/2) * phi'(0) depuis les chi-maps.
    Identique à create_empirical_fig5_fixed mais pour le cas anisotrope.
    """
    Xg, Yg = np.meshgrid(gen.x, gen.y, indexing='ij')
    dists   = np.sqrt(Xg**2 + Yg**2).flatten()

    order    = np.argsort(dists)
    d_sorted = dists[order]
    uniq_d, inverse, counts = np.unique(d_sorted, return_inverse=True,
                                        return_counts=True)
    cum_n   = np.cumsum(counts)
    mask_r0 = uniq_d > 0

    def calc_slope(chi_map_1d):
        chi_s    = chi_map_1d[order]
        chi_sum  = np.bincount(inverse, weights=chi_s)
        cum_chi  = np.cumsum(chi_sum)
        phi_at_r = cum_chi / cum_n
        ep       = {'x': uniq_d[mask_r0], 'y': phi_at_r[mask_r0]}
        result   = tail_dependence_fun(ep)
        return 1.5 * result['slope_at_0']   # f'_p(0) = (3/2)*phi'(0)

    slopes, lo, hi = [], [], []
    for j in range(len(us)):
        chi_maps_j = maps_3d[:, j, :]
        mean_chi   = np.mean(chi_maps_j, axis=0)
        slopes.append(calc_slope(mean_chi))
        boots = [
            calc_slope(np.mean(
                chi_maps_j[np.random.choice(chi_maps_j.shape[0],
                                            chi_maps_j.shape[0], replace=True)],
                axis=0))
            for _ in range(100)
        ]
        lo.append(np.nanquantile(boots, 0.025))
        hi.append(np.nanquantile(boots, 0.975))

    return {
        'u_plot' : gauss_to_exp(np.asarray(us, float)),
        'slopes' : np.array(slopes),
        'lower'  : np.array(lo),
        'upper'  : np.array(hi),
    }


def plot_fig5_aniso(th_fp0, emp_fp0=None, ylim=(-9, 0)):
    """
    Figure 5 style pour le cas anisotrope.
    f'_p(0) en fonction de -log(1-p).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    t_vals  = np.linspace(0.5, 15.0, 500)
    u_th    = norm.ppf(1 - np.exp(-t_vals))

    for gamma in GAMMAS:
        if gamma == 1.0:
            label = r'$\gamma=1.0$ (isotrope)'
        else:
            label = fr'$\gamma={gamma}$  ($\sqrt{{\gamma}}={np.sqrt(gamma):.2f}$)'

        y_th = theoretical_fp0_aniso(u_th, gamma)
        ax.plot(t_vals, y_th, color=COLORS[gamma], linestyle=LS[gamma],
                linewidth=1.4, label=label)

        if emp_fp0 and gamma in emp_fp0:
            emp  = emp_fp0[gamma]
            mask = np.isfinite(emp['slopes'])
            ax.errorbar(
                emp['u_plot'][mask], emp['slopes'][mask],
                yerr=[np.maximum(0, (emp['slopes'] - emp['lower'])[mask]),
                      np.maximum(0, (emp['upper']   - emp['slopes'])[mask])],
                fmt=('o' if gamma == 1.0 else 's'),
                color=COLORS[gamma], markersize=5,
                elinewidth=0.9, capsize=3, zorder=4,
            )

    ax.set_xlabel(r'$-\log(1-p)$', fontsize=12)
    ax.set_ylabel(r"$f'_p(0)$", fontsize=12)
    ax.set_title(r'Dépendance de queue — cas anisotrope  '
                 r'[$f^\prime_p(0) = \sqrt{\gamma}\cdot f^\prime_{p,\mathrm{iso}}(0)$]',
                 fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, title='Shape matrix')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Test rapide : grille 41×41, N=100 champs')
    parser.add_argument('--n_fields', type=int, default=1000)
    args = parser.parse_args()

    GRID_N   = 41  if args.quick else 121
    N_FIELDS = 1000 if args.quick else args.n_fields

    # ── Grille et covariance ──────────────────────────────────────────────────
    x_grid   = np.linspace(-0.5, 0.5, GRID_N)
    y_grid   = np.linspace(-0.5, 0.5, GRID_N)
    cov_fun  = cov_model('Matern', nu=MATERN_NU)

    # ── Seuils ───────────────────────────────────────────────────────────────
    us = np.arange(-2.0, 2.6, 0.5)

    # ── Simulation pour chaque γ ─────────────────────────────────────────────
    emp_dict = {}
    gen_dict = {}

    for gamma in GAMMAS:
        A     = gamma * np.eye(2)        
        label = gamma
        pkl   = f'gen_aniso_g{gamma:.1f}.pkl'

        # Charger ou calculer le générateur
        if os.path.exists(pkl):
            print(f"Chargement {pkl}...")
            with open(pkl, 'rb') as f:
                gen = pickle.load(f)
        else:
            print(f"Calcul générateur γ={gamma} (grille {GRID_N}×{GRID_N})...")
            gen = AnisotropicRFGenerator(x_grid, y_grid, cov_fun, A)
            with open(pkl, 'wb') as f:
                pickle.dump(gen, f)
            print(f"  → sauvegardé dans {pkl}")

        gen_dict[gamma] = gen

        print(f"Simulation γ={gamma} (N={N_FIELDS} champs)...")
        results = run_aniso_experiment(gen, us, n_fields=N_FIELDS, seed=42)
        emp_dict[gamma] = empirical_slope(results, us, n_bootstrap=200)
        print(f"  → terminé (excès à seuil médian : "
              f"{np.sum(np.isfinite(results[:, len(us)//2]))}/{N_FIELDS})")

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nTracé des figures...")

    fig3, _ = plot_aniso_fig3(None, emp_dict)
    fig3.savefig('aniso_fig3_natural.png', dpi=150)
    print("  → aniso_fig3_natural.png")

    fig4, _ = plot_aniso_fig4(None, emp_dict)
    fig4.savefig('aniso_fig4_exp1.png', dpi=150)
    print("  → aniso_fig4_exp1.png")

    fig_r, _ = plot_aniso_ratio(None, emp_dict)
    fig_r.savefig('aniso_ratio_validation.png', dpi=150)
    print("  → aniso_ratio_validation.png")


    # ── Expérience 2 — Dépendance de queue (style Figure 5) ─────────────────
    from scipy.stats import expon
    us_exp2 = np.array([norm.ppf(expon.cdf(t)) for t in np.linspace(2, 10, 8)])

    emp_fp0 = {}
    for gamma in GAMMAS:
        print(f"\nExp2 gamma={gamma} (N={N_FIELDS} champs conditionnés)...")
        maps = run_exp2_aniso(gen_dict[gamma], us_exp2,
                              n_fields=N_FIELDS, seed=42)
        emp_fp0[gamma] = empirical_fp0(maps, gen_dict[gamma], us_exp2)

    fig5, _ = plot_fig5_aniso(None, emp_fp0)
    fig5.savefig('aniso_fig5.png', dpi=150)
    print("  → aniso_fig5.png")

    plt.show()
    print("\nTerminé.")


# =============================================================================
# EXPÉRIENCE 2 — Fonction de dépendance de queue f'_p(0) (style Figure 5)
# =============================================================================
#
# Formule théorique pour le cas anisotrope (convention half-SA density) :
#   f'_p(0) = -2*C1*(u) / (π * C2*(u))
#   Pour A = γI₂ : C1_aniso = sqrt(γ) * C1_iso
#   => f'_p(0)_aniso = sqrt(γ) * f'_p(0)_iso
#
# Conditioning by kriging pour garantir X(0) > u :
#   X_cond(s) = X(s) + sigma12(s) * (z0 - X(0))
#   sigma12 = L @ L[origin_idx, :]  (calculé une fois par générateur)
# =============================================================================