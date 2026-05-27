# Étude des Extremal Ranges : Géométrie et Probabilités des Phénomènes Extrêmes

Ce dépôt contient le code source (R et Python) et les ressources numériques associés au projet de recherche sur la portée extrémale (*extremal range*), réalisé dans le cadre du M1 MMAS à l'Université Paris Cité (2025-2026).

Ce projet s'appuie sur le cadre théorique introduit par Cotsakis, Di Bernardino & Opitz (2024) et propose des redémonstrations détaillées, des simulations numériques comparatives, ainsi qu'une contribution originale étendant ce cadre aux champs gaussiens anisotropiques.

## Auteurs et Encadrement
* **Auteurs :** Khalil Bejaoui & Matys Savidan
* **Encadrants :** Anne Estrade & Jose Gregorio Gomez-Garcia
* **Laboratoire :** MAP5, Université Paris Cité

## Objectif du projet
La modélisation des extrêmes spatiaux dépasse la simple intensité locale : il est crucial de comprendre leur étendue (clusters). Contrairement aux modèles max-stables qui imposent une dépendance asymptotique stricte, la **portée extrémale** est une statistique locale géométrique permettant de quantifier l'étendue spatiale des dépassements de seuil, aussi bien en régime de dépendance qu'en régime d'indépendance asymptotique.

Notre travail articule :
1.  Le lien entre la portée extrémale, la dépendance de queue et les courbures de Lipschitz-Killing (LKC).
2.  Une étude comparative de simulations sur 4 types de champs aléatoires.
3.  **Contribution originale :** La formalisation et la simulation de la portée extrémale pour un champ gaussien soumis à une anisotropie spatiale linéaire (matrice de précision $\Lambda$).

## Structure du projet et Correspondances R / Python

L'implémentation a été pensée traduite du `R` en `Python`. Les scripts principaux (`useful_functions.R` et `useful_functions.py`) contiennent les équivalences suivantes pour la génération de champs et l'analyse spatiale :


## Champs aléatoires simulés

Le code permet de simuler et d'analyser le comportement asymptotique sur grille ($121 \times 121$) pour les champs suivants :
* **Champ Gaussien Isotrope :** Indépendance asymptotique, contraction des clusters en $1/u$.
* **Champ de Student ($k=3$) :** Queues lourdes, stabilisation macroscopique sans renormalisation (autosimilitude).
* **Champ Chi-deux ($k=3$) :** Contraction spatiale sous-gaussienne en $1/\sqrt{u}$.
* **Champ Mélange de Pareto ($\alpha=2$) :** Modèle d'échelle à queue lourde avec convergence stable.
* **Champ Gaussien Anisotropique :** Introduction d'une matrice de forme modifiant la densité de périmètre (LKC dimension 1) d'un facteur $\sqrt{\det(\Lambda)}$.

## Installation et Utilisation

### Prérequis Python
L'environnement Python requiert le calcul scientifique standard et la parallélisation pour gérer les grandes matrices de covariance :
```bash
pip install numpy scipy matplotlib joblib