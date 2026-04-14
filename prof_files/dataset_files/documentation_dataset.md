# Documentation du dataset `dataset_subset_3000.h5`

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Description du dataset](#2-description-du-dataset)
3. [Structure du fichier HDF5](#3-structure-du-fichier-hdf5)
4. [Les paramètres météorologiques de chaque scénario](#4-les-paramètres-météorologiques-de-chaque-scénario)
5. [Les 9 champs 3D par scénario](#5-les-9-champs-3d-par-scénario)
6. [Distribution et diversité des scénarios](#6-distribution-et-diversité-des-scénarios)
7. [La tâche d'apprentissage](#7-la-tâche-dapprentissage)
8. [Encodages géométriques : Voxelisation vs SDF](#8-encodages-géométriques--voxelisation-vs-sdf)
9. [Fonction de perte physique (PINN)](#9-fonction-de-perte-physique-pinn)
10. [Protocole expérimental](#10-protocole-expérimental)
11. [Recommandations sur les modèles](#11-recommandations-sur-les-modèles)
12. [Considérations pratiques](#12-considérations-pratiques)

---

## 1. Contexte et objectif

### 1.1 Une problématique opérationnelle concrète

Imaginez un responsable qualité de l'air dans un grand aéroport, un matin de faible vent. Sur son écran, aucune information en temps réel sur la concentration de NOx dans les différentes zones du tarmac. Les équipes de maintenance travaillent à côté des pistes, les techniciens effectuent des opérations de roulage prolongé, et les avions font tourner leurs moteurs au ralenti. Personne ne sait, à cet instant précis, quelles zones sont exposées à des concentrations de polluants dangereuses.

Le NOx (oxydes d'azote) est un groupe de gaz irritants produits par la combustion à haute température, notamment par les réacteurs d'avions. À forte concentration, il est nocif pour les voies respiratoires et contribue à la formation de smog. Dans un aéroport, les zones proches des pistes et des postes de stationnement sont les plus exposées, mais la concentration varie enormément selon le vent, les bâtiments environnants et les opérations en cours.

Ce responsable aimerait pouvoir, en quelques secondes, visualiser un champ de concentration en temps réel et répondre à des questions simples mais vitales :

- Quelle zone du tarmac est actuellement la plus exposée aux émissions de NOx ?
- Doit-on rediriger les équipes de maintenance vers un autre secteur ?
- Peut-on décaler le démarrage moteur d'un appareil, ou le déplacer vers une autre position de stationnement pour éviter d'exposer du personnel ?

Cette capacité de décision rapide, basée sur une vision instantanée de la qualité de l'air, n'existe pas aujourd'hui dans les outils opérationnels des aéroports, et c'est précisément le problème que ce projet cherche à adresser.

### 1.2 Pourquoi les outils actuels ne suffisent pas

Les modèles physiques de référence pour la simulation de dispersion de polluants en environnement aéroportuaire, tels que **LASPORT** ou **OpenALAQS**, produisent des résultats d'une grande précision. Ils modélisent fidèlement les effets de la turbulence atmosphérique, les sillages des aéronefs, les interactions avec les bâtiments et les variations des conditions météorologiques.

Mais cette précision a un coût : une simulation complète pour un scénario donné demande plusieurs heures de calcul sur des serveurs dédiés. On ne peut pas exécuter LASPORT ou OpenALAQS en réponse à un changement de conditions météo survenu il y a dix minutes. Ces outils sont conçus pour des études préalables, des bilans annuels, des études d'impact, pas pour guider une décision opérationnelle en temps réel.

Il existe donc un fossé entre la précision des modèles physiques et la réactivité qu'exige le terrain.

### 1.3 L'approche par deep learning

L'idée de ce projet est de combler ce fossé en entraînant un modèle de deep learning capable de reproduire, en quelques millisecondes, ce qu'un simulateur physique mettrait des heures à calculer. Le modèle n'est pas destiné à remplacer LASPORT ou OpenALAQS pour les études réglementaires, il est conçu pour opérer aux côtés de ces outils, dans un contexte d'aide à la décision rapide.

Concrètement, étant donné les conditions météorologiques courantes (vent, stabilité atmosphérique) et la géométrie du site (bâtiments, pistes, obstacles), le modèle prédit le champ volumique de concentration en NOx sur l'ensemble du domaine aéroportuaire, comme illustré ci-dessous :

```
Entrée :  champ de vent 3D  +  représentation 3D des obstacles
                    ↓
            Modèle de deep learning
                    ↓
Sortie :  champ de concentration en NOx 3D  (µg/m³)
```

La question centrale que vous devez investiguer est la suivante :

> **Comment représenter la géométrie des obstacles (bâtiments, infrastructures aéroportuaires) pour en tirer le meilleur parti dans un modèle de deep learning ?**

Deux manières de représenter cette géométrie sont comparées tout au long de ce projet : la **voxelisation** (masque binaire) et le **Signed Distance Field (SDF)**. Ces deux approches seront détaillées dans la section [8](#8-encodages-géométriques--voxelisation-vs-sdf).

---

## 2. Description du dataset

Le fichier `dataset_subset_3000.h5` contient **3 000 scénarios** de simulation de dispersion de NOx. Chaque scénario correspond à une configuration imaginaire d'aéroport soumise à des conditions météorologiques particulières. Ces scénarios ont été générés par un modèle physique Lagrangien, du même type que LASPORT, en faisant varier à la fois la géométrie des sites (taille de l'aéroport, nombre de bâtiments, nombre de pistes) et les paramètres atmosphériques (vitesse et direction du vent, stabilité de l'atmosphère).

L'objectif de cette diversité est de couvrir un maximum de situations réelles pour que les modèles de deep learning entraînés soient capables de généraliser, c'est-à-dire de donner de bonnes prédictions sur des configurations jamais vues pendant l'entraînement.

Chaque simulation modélise la dispersion du NOx pendant 1 heure, avec 1 million de particules et une résolution spatiale de 5 m × 5 m × 3 m.

---

## 3. Structure du fichier HDF5

### Qu'est-ce que le format HDF5 ?

HDF5 (Hierarchical Data Format 5) est un format de fichier scientifique très utilisé en physique, météorologie et machine learning. Il permet de stocker de grandes quantités de données numériques de façon organisée et compressée, avec un accès rapide sans avoir à charger tout le fichier en mémoire. On peut le voir comme un "fichier ZIP intelligent" qui contient une arborescence de tableaux numériques, similaire à un système de fichiers.


### Organisation du fichier

Le fichier est organisé en 3 000 groupes, un par scénario. Chaque groupe porte le nom du scénario et contient exactement 9 datasets :

```
dataset_subset_3000.h5              ← Fichier racine
│
├── scenario_00004/                 ← Scénario n°4
│   ├── concentration_grid          ← Champ de concentration NOx [64 × 140 × 140]
│   ├── wind_u                      ← Composante est-ouest du vent [64 × 140 × 140]
│   ├── wind_v                      ← Composante nord-sud du vent  [64 × 140 × 140]
│   ├── wind_w                      ← Composante verticale du vent [64 × 140 × 140]
│   ├── obstacle_mask               ← Masque binaire des obstacles  [64 × 140 × 140]
│   ├── obstacle_sdf                ← Distance aux obstacles (SDF)  [64 × 140 × 140]
│   ├── gse_sdf                     ← Distance au sol (SDF)         [64 × 140 × 140]
│   ├── lto_sdf                     ← Distance aux zones de piste   [64 × 140 × 140]
│   └── emissions_total             ← Total des émissions           [64 × 140 × 140]
│
├── scenario_00009/                 ← Scénario n°9 (aéroport moyen, vent fort...)
│   └── ...
│
└── scenario_XXXXX/                 ← 3 000 groupes au total
    └── ...
```

> **Note :** Les identifiants de scénarios ne sont pas consécutifs (on passe de `scenario_00004` à `scenario_00009`, etc.) car ce fichier est un sous-ensemble d'un dataset plus large. Cela n'a aucune importance pour votre travail.

### Paramètres globaux du fichier

Certaines informations sont communes à tous les scénarios. Elles sont stockées dans les attributs du fichier racine (accessibles via `f.attrs` en h5py) :

| Attribut | Valeur | Ce que ça signifie |
|---|---|---|
| `pollutant` | `"NOx"` | Le gaz simulé |
| `concentration_unit` | `"µg/m³"` | L'unité des valeurs de concentration |
| `sdf_unit` | `"m"` | L'unité des champs SDF (mètres) |
| `n_particles` | 1.000.000 | Nombre de particules dans la simulation physique |
| `dt_s` | 0.5 s | Pas de temps de la simulation (une image toutes les 0,5s) |
| `T_steps` | 7 200 | Nombre total de pas (7200 × 0,5s = 3600s = 1h) |
| `Nz` | 64 | Nombre de couches verticales dans la grille |
| `dx_m`, `dy_m` | 5.0 m | Résolution horizontale : chaque cellule fait 5m × 5m |
| `dz_m` | 3.0 m | Résolution verticale : chaque couche fait 3m d'épaisseur |

La résolution de 5 m × 5 m × 3 m est suffisamment fine pour capturer les effets aérodynamiques autour des bâtiments (tourbillons, sillages, zones de recirculation).

---

## 4. Les paramètres météorologiques de chaque scénario

En plus des 9 tableaux 3D, chaque scénario possède des métadonnées scalaires qui décrivent les conditions de la simulation. Ces valeurs sont stockées dans les attributs du groupe (accessibles via `f["scenario_00004"].attrs`). Il ne s'agit pas de tableaux mais de simples valeurs numériques qui caractérisent le scénario dans sa globalité.

| Attribut | Unité | Description |
|---|---|---|
| `Nx`, `Ny` | — | Dimensions horizontales de la grille (140, 300 ou 500 selon la catégorie) |
| `T_sim_s` | s | Durée de simulation (toujours 3 600 s = 1h) |
| `category` | — | Catégorie de l'aéroport (1 à 4, voir section [6](#6-distribution-et-diversité-des-scénarios)) |
| `U_10` | m/s | Vitesse du vent à 10 m d'altitude au-dessus du sol |
| `wind_dir` | degrés | Direction d'où vient le vent (0° = Nord, 90° = Est) |
| `u_star` | m/s | Vitesse de frottement, mesure l'intensité de la turbulence proche du sol |
| `z0` | m | Longueur de rugosité, caractérise la rugosité de la surface (asphalte, herbe...) |
| `L_M` | m | Longueur de Monin-Obukhov (voir ci-dessous) |
| `stability_class` | — | Classe de stabilité atmosphérique (voir section [6](#6-distribution-et-diversité-des-scénarios)) |
| `n_buildings` | — | Nombre de bâtiments dans le domaine simulé |
| `n_runways` | — | Nombre de pistes |
| `n_active_particles` | — | Nombre de particules encore en mouvement en fin de simulation |

### Comprendre la stabilité atmosphérique et la longueur de Monin-Obukhov

La stabilité atmosphérique décrit la tendance de l'air à se mélanger verticalement. C'est un paramètre fondamental pour la dispersion des polluants :

- Par **temps ensoleillé et vent faible** (atmosphère instable), l'air chaud monte rapidement et crée de forts mouvements verticaux. Les polluants se dispersent vite et sur une grande hauteur, c'est favorable pour la qualité de l'air au niveau du sol.
- Par **nuit calme** (atmosphère stable), les mouvements verticaux sont bloqués. Les polluants restent piégés près du sol et s'accumulent, c'est la situation la plus dangereuse.

La **longueur de Monin-Obukhov** (`L_M`) est le paramètre mathématique qui quantifie cette stabilité :

- **L_M < 0** : atmosphère instable (mélange vertical fort, polluants dilués rapidement)
- **L_M > 0** (valeur finie) : atmosphère stable (polluants confinés près du sol)
- **L_M = 99 999** : valeur conventionnelle indiquant une atmosphère parfaitement neutre (ni stable ni instable)

---

## 5. Les 9 champs 3D par scénario

### Qu'est-ce qu'un "champ 3D" dans ce contexte ?

Chaque tableau de données est ce qu'on appelle un **champ volumique** : imaginez une boîte rectangulaire découpée en petits cubes (appelés **voxels**, l'équivalent 3D des pixels). Chaque voxel contient une valeur numérique, la concentration de NOx (la vitesse du vent, etc.), au point correspondant dans l'espace.

Dans notre dataset, cette boîte fait toujours 64 couches de hauteur (axe Z, de 0 à ~192m au-dessus du sol) et Nx × Ny cellules horizontalement (140×140, 300×300 ou 500×500 selon la taille de l'aéroport).

Tous les tableaux sont stockés en float32, compressés avec gzip pour réduire la taille du fichier. En Python/NumPy/PyTorch, ils apparaissent comme des tenseurs de forme `[64, Nx, Ny]`.

---

### 5.1 La variable cible : `concentration_grid`

```
Shape : [64, Nx, Ny]   |   Unité : µg/m³
```

C'est la donnée que votre modèle doit apprendre à prédire. Elle représente la concentration moyenne en NOx dans chaque voxel du domaine, moyennée sur toute la durée de la simulation (1 heure). Plus la valeur est élevée, plus la zone est polluée.

Un voxel à la valeur 50 µg/m³ signifie que dans le cube de 5m × 5m × 3m correspondant, la concentration moyenne de NOx a été de 50 µg/m³ sur l'heure de simulation. À titre de référence, la valeur limite horaire réglementaire en Europe pour le NO₂ est de 200 µg/m³.

> **Attention :** environ 5,6% des scénarios présentent un champ de concentration entièrement à zéro. Cela correspond aux cas où aucune particule n'était active pendant la simulation, ce qui peut arriver sous certaines conditions de vent extrêmes. Vous devrez décider si vous incluez ou excluez ces scénarios de votre entraînement, et justifier votre choix.

---

### 5.2 Les champs de vent : `wind_u`, `wind_v`, `wind_w`

```
Shape : [64, Nx, Ny]   |   Unité : m/s
```

Le vent est une grandeur vectorielle : en chaque point de l'espace, il a une direction et une intensité. Pour le représenter dans une grille 3D, on le décompose en trois composantes indépendantes selon les trois axes de l'espace :

- `wind_u` : composante **est-ouest** (positive vers l'est)
- `wind_v` : composante **nord-sud** (positive vers le nord)
- `wind_w` : composante **verticale** (positive vers le haut)

Ces trois champs ne représentent pas le vent en altitude loin de l'aéroport, mais bien le champ de vent perturbé par les bâtiments et les pistes : ils encodent donc implicitement les tourbillons, sillages et zones de recirculation créés par les obstacles. C'est pourquoi ils constituent, avec la représentation géométrique des obstacles, les entrées principales de vos modèles.

---

### 5.3 Les représentations géométriques des obstacles

C'est ici que réside le cœur scientifique de ce projet. Les obstacles physiques présents dans l'aéroport (bâtiments, terminaux, hangars, etc.) sont représentés de deux façons différentes dans le dataset. Votre travail est précisément de comparer l'impact de ce choix de représentation sur la qualité des prédictions. Ces deux représentations sont décrites en détail dans la section [8](#8-encodages-géométriques--voxelisation-vs-sdf) ; voici leur description technique :

#### `obstacle_mask`, le masque binaire (voxelisation)

```
Shape : [64, Nx, Ny]   |   Valeurs : 0.0 ou 1.0 (binaire)
```

Pour chaque voxel, la valeur est simplement **1** si le voxel est occupé par un bâtiment solide, **0** si c'est de l'air. C'est l'équivalent 3D d'une image noir et blanc où le blanc représente les murs et le noir représente l'espace libre.

#### `obstacle_sdf`, la distance signée aux obstacles

```
Shape : [64, Nx, Ny]   |   Unité : m
```

Au lieu d'un simple 0 ou 1, chaque voxel contient la distance en mètres à la surface de l'obstacle le plus proche, avec un signe positif si le voxel est dans l'air et négatif s'il est à l'intérieur d'un obstacle. Un voxel à 10m d'un mur aura la valeur +10, un voxel au cœur d'un bâtiment aura une valeur négative.

#### `gse_sdf`, la distance signée au sol

```
Shape : [64, Nx, Ny]   |   Unité : m
```

Même principe que `obstacle_sdf`, mais pour la surface du sol. Ce champ encode la hauteur au-dessus du terrain, utile pour représenter la topographie du site.

#### `lto_sdf`, la distance signée aux zones de piste

```
Shape : [64, Nx, Ny]   |   Unité : m
```

Distance signée aux zones d'opérations LTO (Landing and Take-Off), c'est-à-dire les pistes et les voies de circulation des aéronefs. Ce champ encode la proximité des sources d'émissions principales, là où les moteurs sont actifs.

---

### 5.4 `emissions_total`

```
Shape : [64, Nx, Ny]
```

Ce champ représente la distribution spatiale des émissions de NOx dans le domaine : il indique, voxel par voxel, quelle quantité totale de polluant a été injectée dans la simulation à cet endroit. C'est en quelque sorte la "carte des sources". Les zones de forte émission correspondent aux positions des moteurs actifs pendant la simulation.

---

## 6. Distribution et diversité des scénarios

### Catégories d'aéroports

Pour couvrir différentes tailles d'aéroports, le dataset définit 4 catégories. La catégorie détermine la taille du domaine spatial simulé :

| Catégorie | Taille de grille (Nx × Ny) | Superficie couverte | Nb de scénarios | Description |
|---|---|---|---|---|
| 1 | 140 × 140 | 700m × 700m | 750 | Petit aéroport, 1 piste |
| 2 | 300 × 300 | 1,5km × 1,5km | 750 | Aéroport moyen, 2 pistes |
| 3 | 500 × 500 | 2,5km × 2,5km | 750 | Grand aéroport, 3 à 4 pistes |
| 4 | 140 × 140 | 700m × 700m | 750 | Variante de catégorie 1, géométrie différente |

La distribution est parfaitement équilibrée : 750 scénarios par catégorie, ce qui évite tout biais lié à la taille de l'aéroport.

> **Point important :** les catégories 1 et 4 ont la même taille de grille (140×140) mais des configurations géométriques différentes. Ne vous fiez donc pas à la taille de grille pour identifier la catégorie, utilisez l'attribut `category` du scénario.

### Classes de stabilité atmosphérique

La classification de Pasquill-Gifford est le système standard pour décrire les conditions de dispersion dans l'atmosphère. Le dataset couvre 6 classes, avec exactement 500 scénarios par classe :

| Classe | Conditions typiques | Effet sur la dispersion |
|---|---|---|
| I | Journée très ensoleillée, vent faible (< 2 m/s) | Dispersion très forte, polluants dilués rapidement en altitude |
| II | Journée ensoleillée, vent modéré | Bonne dispersion verticale |
| III/1 | Ciel légèrement couvert ou vent moyen | Dispersion légèrement supérieure à neutre |
| III/2 | Fin de journée, vent modéré | Légère tendance à la stabilisation |
| IV | Nuit nuageuse, vent modéré | Stabilité marquée, polluants restent proches du sol |
| V | Nuit claire, vent faible | Forte stabilité, accumulation maximale des polluants |

Cette diversité est cruciale : un modèle qui ne fonctionne qu'en conditions instables (classe I) serait inutile lors des nuits calmes (classe V), qui sont précisément les situations les plus dangereuses pour la qualité de l'air.

### Diversité des conditions météorologiques

| Variable | Min | Max | Moyenne |
|---|---|---|---|
| Vitesse du vent U₁₀ (m/s) | 1,00 | 14,99 | 3,62 |
| Nombre de bâtiments par scénario | 0 | 54 | 8,7 |
| Nombre de pistes | 1 | 4 | — |
| Particules actives en fin de simulation | 0 | 498 088 | 10 210 |

---

## 7. La tâche d'apprentissage

### Ce que le modèle doit faire, concrètement

On est face à un problème de régression volumique : à partir de plusieurs champs 3D en entrée, le modèle doit produire un champ 3D en sortie. C'est l'équivalent 3D d'une segmentation d'image, mais au lieu de prédire une classe par pixel, on prédit une valeur continue (la concentration) par voxel.

La fonction que le modèle doit apprendre est :

```
f( wind_u, wind_v, wind_w, [représentation géométrique] )  →  concentration_grid
```

### Entrées et sorties selon l'encodage choisi

Selon que vous utilisez la voxelisation ou le SDF, le nombre de canaux d'entrée change.

**Cas 1 — Voxelisation (4 canaux) :**

| # Canal | Tableau | Rôle |
|---|---|---|
| 1 | `wind_u` | Vent est-ouest |
| 2 | `wind_v` | Vent nord-sud |
| 3 | `wind_w` | Vent vertical |
| 4 | `obstacle_mask` | Géométrie binaire des obstacles |

**Cas 2 — SDF (6 canaux) :**

| # Canal | Tableau | Rôle |
|---|---|---|
| 1 | `wind_u` | Vent est-ouest |
| 2 | `wind_v` | Vent nord-sud |
| 3 | `wind_w` | Vent vertical |
| 4 | `obstacle_sdf` | Distance signée aux bâtiments |
| 5 | `gse_sdf` | Distance signée au sol |
| 6 | `lto_sdf` | Distance signée aux zones de piste |

Dans les deux cas, le tenseur d'entrée a la forme `[C, 64, Nx, Ny]` (C canaux, 64 couches verticales, Nx × Ny cellules horizontales), et la sortie est `[1, 64, Nx, Ny]`.

### Le problème des tailles variables

Tous les scénarios n'ont pas la même taille de grille. Un scénario de catégorie 1 produit un tenseur `[C, 64, 140, 140]`, un scénario de catégorie 3 produit `[C, 64, 500, 500]`. La plupart des architectures de deep learning nécessitent des entrées de taille fixe.

Pour gérer cela, plusieurs stratégies sont possibles, à vous de choisir et de justifier la vôtre :

- **Entraînement séparé par catégorie** : un modèle distinct pour chaque catégorie. Simple, mais vous obtenez 4 modèles au lieu d'un.
- **Rembourrage (padding)** : toutes les grilles sont complétées avec des zéros jusqu'à la taille maximale (500×500). Simple à implémenter, mais coûteux en mémoire.
- **Découpage (cropping)** : on extrait des sous-volumes de taille fixe. Possible mais on perd de l'information sur les grandes grilles.
- **Architectures flexibles** : certains modèles comme le FNO gèrent nativement des tailles d'entrée variables.

---

## 8. Encodages géométriques : Voxelisation vs SDF

### Pourquoi la représentation de la géométrie est-elle cruciale ?

La dispersion des polluants dans un aéroport est massivement influencée par les bâtiments. Un terminal qui fait obstacle au vent crée un sillage protégé d'un côté et une zone d'accélération de l'autre. Une rangée de hangars peut canaliser le vent comme une buse. Ces effets sont déterminants pour savoir où se concentre le NOx.

Le modèle de deep learning doit donc comprendre la géométrie pour bien prédire la dispersion. Mais comment lui communiquer cette géométrie ? C'est là que les deux encodages divergent.

### La voxelisation : simple mais limitée

La voxelisation est intuitive : on divise l'espace en petits cubes et on marque chaque cube comme "solide" (1) ou "vide" (0). C'est exactement ce que ferait quelqu'un qui dessinerait un plan 3D en pixels.

```
Vue de côté d'un bâtiment (voxelisation) :

  Air   Air   Air   Air   Air
  Air  [MUR] [MUR] [MUR]  Air
  Air  [MUR] [MUR] [MUR]  Air
  Air  [MUR] [MUR] [MUR]  Air
  Sol   Sol   Sol   Sol   Sol

Valeurs : 0   1     1     1    0
          0   1     1     1    0
          0   1     1     1    0
```

**Le problème :** pour le réseau de neurones, tous les voxels "Air" ont la même valeur 0, que ce soit à 1 mètre du mur ou à 100 mètres. Or ces deux situations sont physiquement très différentes : près d'un mur, le vent est fortement perturbé, la couche limite est fine, la concentration peut être très élevée. Loin du mur, l'écoulement est libre. Le réseau doit *inférer* cette information de proximité uniquement à partir du contexte spatial, c'est un effort d'apprentissage supplémentaire.

### Le Signed Distance Field (SDF) : riche et continu

Le SDF résout ce problème en donnant à chaque voxel une valeur qui encode directement sa distance à la surface la plus proche, avec un signe qui indique si on est à l'intérieur ou à l'extérieur de l'obstacle.

```
Vue de côté du même bâtiment (SDF) :

  +5    +4    +3    +4    +5
  +4   [-1]  [-2]  [-1]  +4
  +3   [-1]  [-3]  [-1]  +3
  +4   [-1]  [-2]  [-1]  +4
  +3    +2    +1    +2    +3

Valeurs positives = air libre (distance à l'obstacle)
Valeurs négatives = intérieur d'un obstacle
```

Avec le SDF, un voxel à 1m d'un mur a la valeur +1, un voxel à 50m a la valeur +50. Le réseau reçoit directement l'information de proximité sans avoir à l'inférer. C'est une représentation continue et beaucoup plus informative, particulièrement adaptée à la modélisation des phénomènes de couche limite et d'effets de paroi qui gouvernent la dispersion des polluants.

### Ce que vous devez comparer

Vous comparerez les deux encodages sur les dimensions suivantes :

1. **Qualité de prédiction** : RMSE et MAE sur le champ de concentration prédit
2. **Généralisation** : est-ce que l'avantage d'un encodage se confirme sur toutes les catégories d'aéroports et toutes les classes de stabilité ?
3. **Coût** : l'encodage SDF ajoute 2 canaux d'entrée supplémentaires, est-ce que cela justifie le gain en performance ?

---

## 9. Fonction de perte physique (PINN)

### Qu'est-ce qu'un PINN ?

Un **Physics-Informed Neural Network (PINN)** est un modèle de deep learning qui ne se contente pas d'apprendre à minimiser une erreur de prédiction classique. Il est également contraint, pendant l'entraînement, à respecter des lois physiques connues du problème qu'il résout. Ces contraintes sont encodées directement dans la fonction de perte (loss function) : si le modèle produit une prédiction qui viole une loi physique, il est pénalisé en proportion.

Pourquoi est-ce important ici ? Sans contrainte physique, un réseau de neurones peut produire des champs de concentration qui semblent visuellement proches de la réalité mais qui contiennent des absurdités physiques, par exemple, du NOx qui s'accumule à l'intérieur d'un mur, ou un champ avec des discontinuités brutales impossibles dans un écoulement turbulent réel. Ces artefacts ne seraient jamais produits par un solveur physique comme LASPORT. Le PINN garantit que les prédictions restent physiquement cohérentes, ce qui est essentiel pour un outil destiné à guider des décisions opérationnelles.

> **Ce n'est pas optionnel.** Vous devez impérativement utiliser la fonction de perte composite fournie avec ce projet pour tous vos entraînements, quelle que soit l'architecture choisie.

### La loss composite : formule générale

La loss totale que vous minimisez à chaque itération est une combinaison de quatre termes :

```
L_total = L_mse + λ_mass · (L_mass + L_solid) + λ_tv · L_TV
```

avec par défaut λ_mass = 0,1 et λ_tv = 0,01. Chaque terme correspond à une contrainte physique précise, décrite ci-dessous.

---

### Composante 1 — MSE pondérée near-wall (`L_mse`)

La première composante est une MSE (Mean Squared Error) classique, mais spatialement pondérée. Elle mesure l'écart entre la concentration prédite et la concentration réelle, voxel par voxel.

La différence avec une MSE ordinaire est que les erreurs commises près des parois (murs de bâtiments, surface du sol) sont pénalisées plus fort que les erreurs en plein air. Le poids attribué à chaque voxel décroît exponentiellement avec sa distance à l'obstacle le plus proche : un voxel à 1m d'un mur a un poids beaucoup plus élevé qu'un voxel à 50m.

Cette pondération est physiquement motivée : c'est précisément près des surfaces que les gradients de concentration sont les plus intenses et les plus difficiles à apprendre. C'est aussi là que se trouvent les travailleurs sur le terrain. Mieux prédire ces zones est donc doublement prioritaire, pour la précision physique et pour la pertinence opérationnelle.

La pondération fonctionne de la même façon que l'encodage géométrique soit un SDF ou un masque binaire : dans les deux cas, la proximité d'une paroi est détectée à partir du canal géométrique fourni en entrée du modèle.

---

### Composante 2 — Conservation de masse (`L_mass`)

Cette contrainte traduit un principe fondamental de la physique : la masse se conserve. Le NOx émis par les moteurs ne peut pas disparaître ou apparaître spontanément, la quantité totale de polluant dans le domaine doit être cohérente entre la prédiction et la simulation de référence.

Concrètement, on somme la concentration prédite sur tous les voxels d'air libre et on compare cette somme à celle de la concentration réelle. Si le modèle prédit globalement moins de NOx qu'il n'en existe réellement (ou plus), il est pénalisé.

Cette contrainte évite un biais fréquent dans les réseaux de neurones appliqués aux champs physiques : le modèle peut apprendre à "faire joli" visuellement tout en sous-estimant ou sur-estimant la quantité totale de polluant, ce qui serait catastrophique pour un outil de gestion de la qualité de l'air.

---

### Composante 3 — Herméticité des obstacles (`L_solid`)

Cette contrainte impose une condition aux limites physique fondamentale : les obstacles solides (bâtiments, murs) sont imperméables. Autrement dit, la concentration de NOx à l'intérieur d'un bâtiment doit être nulle, le polluant ne peut pas traverser les murs.

Sans cette contrainte, un réseau de neurones peut très bien apprendre à fuiter de la concentration dans les solides, surtout au début de l'entraînement ou pour des architectures qui ne voient pas bien les frontières. La pénalité d'herméticité force le modèle à respecter cette condition limite en le pénalisant chaque fois qu'une valeur de concentration non nulle est prédite dans un voxel solide.

---

### Composante 4 — Lissage spatial physique, Total Variation (`L_TV`)

La quatrième contrainte s'appuie sur une propriété fondamentale des champs de dispersion turbulente : les panaches de polluants sont spatialement continus et lisses. Un champ de concentration réel ne peut pas présenter de discontinuités brutales ou de variations en dents de scie voxel à voxel, la turbulence atmosphérique mélange les particules progressivement.

La **Total Variation (TV) Loss** pénalise les variations abruptes du champ prédit en calculant les différences entre voxels adjacents dans les trois directions de l'espace. Elle est pondérée de façon anisotrope : la direction verticale (couche de 3m) est pénalisée plus fortement que les directions horizontales (cellule de 5m), car les gradients verticaux dans la couche atmosphérique basse sont naturellement plus intenses.

Cette contrainte est particulièrement importante pour certaines architectures comme le FNO, qui peut produire des artefacts spectraux non physiques sous forme d'oscillations dans le champ prédit.

---

### Récapitulatif des composantes

| Terme | Nom | Contrainte physique encodée |
|---|---|---|
| `L_mse` | MSE pondérée near-wall | Précision accrue près des surfaces où les gradients sont forts |
| `L_mass` | Conservation de masse | Quantité totale de NOx cohérente entre prédiction et réalité |
| `L_solid` | Herméticité des obstacles | Concentration nulle dans les solides (condition aux limites) |
| `L_tv` | Total Variation | Continuité spatiale du panache (diffusion turbulente) |

### Suivre les composantes pendant l'entraînement

La fonction de perte renvoie non seulement la loss totale mais aussi chaque composante séparément. Il est fortement recommandé de logger chaque terme individuellement dans votre outil de suivi (TensorBoard, WandB, etc.) pendant l'entraînement. Cela vous permettra de détecter si une contrainte physique particulière est mal satisfaite, par exemple, une `L_solid` qui ne descend pas indique que le modèle continue à fuiter de la concentration dans les obstacles malgré l'entraînement.

Des valeurs indicatives en régime stable : `L_mass` ∈ [0, 0,3] et `L_solid` ∈ [0, 0,1].

---

## 10. Protocole expérimental

### Division du dataset en train / validation / test

Avant tout entraînement, vous devez diviser les 3000 scénarios en trois sous-ensembles :

- **Train (entraînement)** : les données sur lesquelles le modèle apprend.
- **Validation** : les données utilisées pendant l'entraînement pour surveiller la progression et ajuster les hyperparamètres (learning rate, architecture...) sans biaiser le test final.
- **Test** : les données réservées uniquement à l'évaluation finale. Elles ne doivent jamais être vues par le modèle pendant l'entraînement ni pendant le réglage.

Une répartition raisonnable pour 3000 exemples :

| Split | Proportion | Nb de scénarios |
|---|---|---|
| Entraînement | 70% | ~2 100 |
| Validation | 15% | ~450 |
| Test | 15% | ~450 |

**Recommandation importante :** effectuez un split stratifié par catégorie et par classe de stabilité. Cela signifie que chaque split doit contenir approximativement la même proportion de chaque catégorie (1, 2, 3, 4) et de chaque classe de stabilité (I à V). Un tirage purement aléatoire risque de déséquilibrer les sous-ensembles et de fausser l'évaluation. En Python, `sklearn.model_selection.train_test_split` avec le paramètre `stratify` permet de faire cela facilement.

### Métriques d'évaluation obligatoires

Vous devez reporter au minimum le **RMSE** et le **MAE** sur votre set de test :

```
RMSE = sqrt( mean( (C_prédit - C_réel)² ) )
MAE  = mean( |C_prédit - C_réel| )
```

Le RMSE pénalise plus fortement les grandes erreurs (il est sensible aux pics de concentration mal prédits), tandis que le MAE donne une erreur moyenne plus robuste. Les deux sont complémentaires.

Ces métriques doivent être calculées et reportées :

- **Globalement** sur l'ensemble du set de test
- **Par catégorie** (1, 2, 3, 4) pour vérifier si le modèle généralise bien sur toutes les tailles d'aéroport
- **Par classe de stabilité** pour vérifier la robustesse aux conditions météorologiques

Vous êtes libres d'ajouter d'autres métriques (SSIM pour la similarité structurelle, erreur relative, performance sur les zones de forte concentration...) si vous jugez qu'elles apportent une information pertinente.

### Tableau de résultats attendu

Chaque expérience doit être systématiquement comparée à la baseline **U-Net 3D** avec l'encodage correspondant. En fin de projet, vous devez produire un tableau de la forme :

| Modèle | Encodage | RMSE ↓ | MAE ↓ | Nb paramètres |
|---|---|---|---|---|
| U-Net 3D (baseline) | Voxelisation | — | — | — |
| U-Net 3D (baseline) | SDF | — | — | — |
| Votre modèle | Voxelisation | — | — | — |
| Votre modèle | SDF | — | — | — |

La flèche ↓ signifie "plus c'est bas, mieux c'est". Ce tableau est le cœur de votre analyse comparative.

---

## 11. Recommandations sur les modèles

### La contrainte matérielle

Vos machines ne permettent pas d'entraîner des modèles volumineux sur des données 3D. La contrainte stricte est : moins de 2 millions de paramètres. À titre de référence, un U-Net 3D standard avec 64 filtres de base dépasse facilement les 10M paramètres, il faut donc travailler avec des architectures allégées.

### La baseline obligatoire : U-Net 3D

Le **U-Net** est une architecture de réseau de neurones convolutif initialement conçue pour la segmentation d'images médicales (Ronneberger et al., 2015). Son principe repose sur deux parties :

- Un **encodeur** qui compresse progressivement les données (comme un entonnoir) pour extraire des représentations de plus en plus abstraites
- Un **décodeur** qui reconstruit progressivement un champ de sortie à la résolution originale
- Des **connexions de saut** (skip connections) qui relient directement chaque niveau de l'encodeur au niveau correspondant du décodeur, permettant de préserver les détails fins

Le **U-Net 3D** est simplement la version où toutes les convolutions sont en 3D au lieu de 2D, ce qui lui permet de traiter des volumes (nos tenseurs [C, 64, Nx, Ny]) plutôt que des images.

Pour rester sous 2M paramètres, utilisez un U-Net 3D avec 16 ou 32 filtres de base au lieu des 64 habituels.

### Alternatives envisageables

Vous pouvez proposer l'architecture de votre choix, à condition de respecter la contrainte en paramètres. Voici quelques pistes adaptées à ce problème :

**Factorized FNO (Fourier Neural Operator factorisé)** — Le FNO est une architecture conçue spécifiquement pour les problèmes physiques. Au lieu d'apprendre des filtres locaux comme une convolution, il apprend des filtres dans l'espace de Fourier, ce qui lui permet de capturer des dépendances à longue portée (un polluant émis en un point peut se retrouver loin sous l'effet du vent). La version "factorisée" décompose la transformée 3D en transformées séparées sur chaque axe, ce qui divise drastiquement le nombre de paramètres.

**U-Net avec convolutions séparables** — On remplace chaque convolution 3D classique par une convolution "depthwise separable" : d'abord une convolution spatiale sur chaque canal séparément, puis une convolution 1×1×1 pour mélanger les canaux. Cette variante réduit le nombre de paramètres d'un facteur ~9 à précision comparable.

**Attention U-Net compact** — On ajoute des mécanismes d'attention aux connexions de saut du U-Net. L'attention permet au modèle de se focaliser sur les régions les plus pertinentes (typiquement, les zones proches des obstacles), ce qui peut améliorer la précision sans augmenter beaucoup le nombre de paramètres.

Quel que soit votre choix, justifiez-le (pourquoi cette architecture est-elle adaptée au problème ?) et comparez-le systématiquement à la baseline U-Net 3D.

---

## 12. Considérations pratiques

### Accès aux données avec h5py

Le format HDF5 permet de lire les données à la demande, sans charger les 116 Go en mémoire. Le principe est le suivant : vous ouvrez le fichier, naviguez jusqu'au dataset qui vous intéresse, et ne lisez que les données nécessaires.

```python
import h5py
import numpy as np

# Ouvrir le fichier en lecture seule
with h5py.File("dataset_subset_3000.h5", "r") as f:

    # Lister tous les scénarios disponibles
    scenario_names = list(f.keys())   # ['scenario_00004', 'scenario_00009', ...]

    # Accéder à un scénario
    scenario = f["scenario_00004"]

    # Lire les attributs (métadonnées scalaires)
    category    = scenario.attrs["category"]    # → 1
    U_10        = scenario.attrs["U_10"]        # → 1.36 m/s
    stab_class  = scenario.attrs["stability_class"]  # → 'IV'

    # Lire un tableau 3D ([:] charge tout en mémoire comme un numpy array)
    wind_u      = scenario["wind_u"][:]         # shape : (64, 140, 140)
    conc        = scenario["concentration_grid"][:]  # shape : (64, 140, 140)
```

### Construction d'un Dataset PyTorch

Dans un projet de deep learning, vous aurez besoin d'un `torch.utils.data.Dataset` qui charge les données à la volée. La règle d'or avec HDF5 et PyTorch est d'ouvrir le fichier dans `__getitem__` et non dans `__init__`, car le handle HDF5 ne peut pas être partagé entre les workers du DataLoader :

```python
import h5py
import torch
from torch.utils.data import Dataset

class AirportDataset(Dataset):
    def __init__(self, h5_path, scenario_names, encodage="voxel"):
        self.h5_path = h5_path
        self.scenario_names = scenario_names  # liste des scénarios pour ce split
        self.encodage = encodage              # "voxel" ou "sdf"

    def __len__(self):
        return len(self.scenario_names)

    def __getitem__(self, idx):
        name = self.scenario_names[idx]
        with h5py.File(self.h5_path, "r") as f:
            s = f[name]
            wind_u = s["wind_u"][:]
            wind_v = s["wind_v"][:]
            wind_w = s["wind_w"][:]
            target = s["concentration_grid"][:]

            if self.encodage == "voxel":
                geom = s["obstacle_mask"][:]
                x = np.stack([wind_u, wind_v, wind_w, geom], axis=0)
            else:
                obs_sdf = s["obstacle_sdf"][:]
                gse_sdf = s["gse_sdf"][:]
                lto_sdf = s["lto_sdf"][:]
                x = np.stack([wind_u, wind_v, wind_w, obs_sdf, gse_sdf, lto_sdf], axis=0)

        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(target[np.newaxis], dtype=torch.float32)
```

### Mémoire et batch size

La mémoire requise par scénario dépend fortement de la catégorie :

| Catégorie | Taille grille | Mémoire par tenseur | Mémoire pour les 9 tenseurs |
|---|---|---|---|
| 1 et 4 | 140 × 140 | ~0,5 Mo | ~4,5 Mo |
| 2 | 300 × 300 | ~2,3 Mo | ~21 Mo |
| 3 | 500 × 500 | ~6,4 Mo | ~58 Mo |

Un batch de 4 scénarios de catégorie 3 représente ~230 Mo de données seules, sans compter les gradients et les activations intermédiaires du modèle. Ajustez votre `batch_size` en conséquence, commencez par 1 ou 2 pour la catégorie 3.

### Normalisation des données

Les valeurs de concentration varient sur plusieurs ordres de grandeur (de 0 à des milliers de µg/m³) et sont très asymétriques (beaucoup de zéros, quelques pics élevés). Sans normalisation, l'entraînement sera instable.

Voici quelques approches courantes :

- **Normalisation log** : `log(1 + concentration)`, efficace pour les distributions très asymétriques
- **Min-Max par canal** : ramener chaque canal entre 0 et 1 en utilisant les min/max calculés sur le set d'entraînement
- **Standardisation (Z-score)** : soustraire la moyenne et diviser par l'écart-type, canal par canal

Quelle que soit votre approche, calculez les statistiques de normalisation uniquement sur le set d'entraînement et appliquez-les ensuite à la validation et au test. Ne "fuitez" pas d'information du test dans la normalisation.
