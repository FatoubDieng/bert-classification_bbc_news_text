# BBC News Text Classifier — Fine-tuning BERT

**Classification de texte : Fine-tuning de BERT**
**Master2 2026 — Dakar Institute of Technology**
**Binômes** Fatou DIENG & Zana Baba stephane Coulibaly

##  Dataset choisi

**Nom** : BBC News Text Complexity Summarization
**Fichier** : `bbc_news_text_complexity_summarization.csv`
**Tâche** : Classification de la colonne `text` selon la colonne `labels`
**Lien WandB** : https://wandb.ai/bintousalih-dakar-institute-of-technology/dit-2026-bert-nlp

### Statistiques du dataset

| Caractéristique         |  Valeur             |
------------------------------------------------
| Nombre total d'exemples | **2 127**           |
-----------------------------------------------
| Nombre de colonnes      | 7 (`text`, `labels`, 
    `no_sentences`, `Flesch Reading Ease Score`, 
    `Dale-Chall Readability Score`, 
    `text_rank_summary`, `lsa_summary`)         |
-----------------------------------------------
| Nombre de classes       | **5**               |
-----------------------------------------------
| Classes                 | `business`, 
                           `entertainment`, 
                           `politics`, 
                           `sport`, `tech`      |
------------------------------------------------
| Longueur min (mots)    | **89**               |
------------------------------------------------
| Longueur max (mots)    | **4 432**            |
------------------------------------------------
| Longueur moyenne (mots) | **384**             |
------------------------------------------------
| Longueur médiane (mots) | **331**             |
------------------------------------------------
| 95e percentile (mots)   | **732**             |
------------------------------------------------
| Valeurs manquantes      | **0**               |
------------------------------------------------

### Distribution des classes

| Classe  | Exemples | %    |
-----------------------------
| sport   |    505   | 23.7% |
------------------------------
| business| 503      | 23.6% |
------------------------------
| politics| 403      | 18.9% |
------------------------------
|entertainment| 369  |17.3%  |
------------------------------
| tech | 347         | 16.3% |

> ![alt text](image.png)

**Équilibre des classes** : ratio max/min = **1.46** ≤ 2:1 : classes équilibrées 
Des poids de classes ont néanmoins été appliqués dans `CrossEntropyLoss` par précaution :
`[4.23, 5.77, 5.28, 4.21, 6.12]` pour `[business, entertainment, politics, sport, tech]`.

### Distribution des longueurs de textes

> ![alt text](image-1.png)

Le 95e percentile est à 732 mots → **max_length = 512 tokens** choisi
(les textes BBC sont longs, truncation appliquée aux textes dépassant 512 tokens).

### 5 exemples de textes

| # | Label | Extrait du texte |

| 1 | **business** | *"EU 'too slow' on economic reforms — Most EU countries have failed to put in place policies aimed at making Europe the world's most competitive economy..."* |
| 2 | **tech** | *"BBC web search aids odd queries — The BBC's online search engine was used a record amount in 2004, helping with enquires both simple and strange..."* |
| 3 | **sport** | *"Serena becomes world number two — Serena Williams has moved up five places to second in the world rankings after her Australian Open win..."* |
| 4 | **politics** | *"Boothroyd calls for Lords speaker — Betty Boothroyd has said the House of Lords needs its own Speaker..."* |
| 5 | **tech** | *"Fast lifts rise into record books — Two high-speed lifts at the world's tallest building have been officially recognised as the planet's fastest..."* |

---

##  Architecture du modèle

### Modèle de base

| Paramètre     | Valeur | Justification |
------------------------------------------
| **Modèle**    | `bert-base-uncased` | Dataset en anglais → BERT anglais |
---------------------------------------------------------------------------
| **Tokenizer** | `AutoTokenizer` HuggingFace | Compatible BERT, gère WordPiece |
--------------------------------------------------------------------------------
| **Max length**| `512` tokens | 95e percentile à 732 mots → 512 tokens nécessaires |
---------------------------------------------------------------------------------
| **Tête de classification** | `Linear(768, 5)` | Classifieur sur token [CLS] |
-------------------------------------------------------------------------------
| **Dropout**   | `0.3` | Régularisation avant la couche de sortie            |
------------------------------------------------------------------------------
| **Total paramètres** | **109 486 085** | Tous entraînables (fine-tuning complet) |

### Pourquoi BERT ?

BERT (Bidirectional Encoder Representations from Transformers, Devlin et al. 2018)
est pré-entraîné sur Wikipedia + BookCorpus (3,3 milliards de mots).
Son encodage bidirectionnel capture le contexte gauche et droite de chaque mot,
ce qui le rend particulièrement efficace pour la classification de texte.

Le token `[CLS]` (premier token) encode la représentation globale de la phrase
et est utilisé comme entrée de la tête de classification.


## Choix techniques et hyperparamètres

| Hyperparamètre | Valeur | Justification |
------------------------------------------
| Learning rate | `2e-5` | LR typique BERT — > 5e-5 → catastrophic forgetting |
------------------------------------------------------------------------------
| Batch size | `16` | Limite mémoire CPU |
-----------------------------------------
| Epochs | `4` | BERT converge vite — 97% dès l'epoch 3 |
---------------------------------------------------------
| Optimiseur | `AdamW` | Adam + weight decay — meilleur pour BERT |
---------------------------------------------------------------------
| Weight decay | `0.01` | Régularisation L2 (sauf biais et LayerNorm) |
----------------------------------------------------------------------
| Scheduler | Linéaire + warmup (10%) | 42 warmup steps / 428 total steps |
-------------------------------------------------------------------------
| Loss | `CrossEntropyLoss` | Classification multi-classes (5 classes) |
-----------------------------------------------------------------------
| Gradient clipping | `max_norm=1.0` | Évite les gradients explosifs |
----------------------------------------------------------------------
| Seed | `42` | Reproductibilité |
---------------------------------
| Split | `80/20 stratifié` | 1 701 train / 426 val |

### Mapping labels index

```python
{'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
```

---

## Structure du projet

```
bert-classification-bbc-news-text
├── data/
│   └── bbc_news_text_complexity_summarization.csv
├── checkpoints/
│   ├── best_model_bert_bbc_classification.pth   ← meilleur modèle (epoch 4)
│   ├── tokenizer/                               ← tokenizer sauvegardé
│   └── label2idx.json                           ← mapping classes → index
├── results/
│   ├── confusion_matrix_bert_bbc_classification.png
│   └── learning_curves_bert_bbc_classification.png
├── dataset.py     ← TextClassificationDataset + inspection + split
├── model.py       ← BertClassifier + load_model
├── train.py       ← train_epoch / eval_epoch / main (boucle PyTorch manuelle)
├── utils.py       ← métriques, seed, matrice de confusion, courbes
├── demo.py        ← interface Gradio interactive
├── requirements.txt
└── README.md
```


##  Installation et exécution

### 1 — Cloner le repo

```bash
git clone https://github.com/[votre-username]/bert-classification-bbc.git
cd bert-classification-bbc
```

### 2 — Créer un environnement virtuel

```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux / Mac
python -m venv env
source env/bin/activate
```

### 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4 — Placer le dataset

```
Créer le dossier data/ et y copier :
bbc_news_text_complexity_summarization.csv
```

### 5 — Inspecter le dataset

```bash
python dataset.py
```

### 6 — Entraîner le modèle

```bash
wandb login   # Entrer la clé API depuis https://wandb.ai/authorize
python train.py
```

### 7 — Lancer la démo Gradio

```bash
python demo.py
# Ouvrir : http://localhost:7860
```


##  Résultats

### Métriques finales

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
----------------------------------------------------------------
| 1     | 1.1017     | 62.43%    | 0.2743   | 95.54%  | 95.26% |
-----------------------------------------------------------------
| 2     | 0.1768     | 97.00%    | 0.1069   | 96.95%  | 96.84%  |
-----------------------------------------------------------------
| 3     | 0.0673     | 98.77%    | 0.0847   | 97.42%  | 97.34%  |
----------------------------------------------------------------
| **4** |**0.0376**  |**99.29%** |**0.0842**|**97.42%**|**97.30%**|

| Métrique finale           | Valeur     |
-------------------------------------------
| **Meilleure val_loss**    | **0.0842**(epoch 4) |
-------------------------------------------
| **Val Accuracy**          | **97.42%** |
------------------------------------------
| **Val F1-score (macro)**  | **97.30%** |
-----------------------------------------
| **Train Accuracy**        | **99.29%** |
-----------------------------------------
| **Meilleur epoch**        | **4**      |

### Courbes d'apprentissage
> ![alt text](image-2.png)

**Observations** :
- BERT atteint **95.54% de val accuracy dès l'epoch 1** — démonstration spectaculaire du transfer learning
- La val_loss descend régulièrement de 0.2743 -> 0.0842 sur 4 epochs
- La train_loss chute rapidement de 1.10 -> 0.04 — convergence très rapide
- Léger overfitting visible à l'epoch 4 : train_acc (99.29%) > val_acc (97.42%)
  -> acceptable et prévu pour un fine-tuning BERT sur 4 epochs

### Matrice de confusion

> ![alt text](image-3.png)

### Lien WandB

Toutes les courbes sont visibles sur :
```
https://wandb.ai/bintousalih-dakar-institute-of-technology/dit-2026-bert-nlp/runs/c43la5cy
```

## Interface Gradio

> ![alt text](image-4.png) ![alt text](image-5.png)

L'interface permet de :
- Saisir un texte BBC News en anglais
- Voir la classe prédite avec les probabilités des 5 classes
- Tester 5 exemples pré-remplis couvrant toutes les catégories

Lancement :
```bash
python demo.py
# http://localhost:7860
```

---

## Analyse et interprétation

### Apport du pré-entraînement (Transfer Learning NLP)

BERT atteint **95.54% dès la première epoch** — là où un modèle from scratch
nécessiterait des dizaines d'epochs pour approcher ce niveau.
Cela démontre la puissance du transfer learning en NLP :
les représentations contextuelles bidirectionnelles apprises sur 3,3 milliards de mots
capturent parfaitement le vocabulaire et le style des articles BBC News.

Avec seulement **1 701 exemples d'entraînement** et **4 epochs**,
le modèle atteint 97.42% de val accuracy — un résultat remarquable.

### Analyse overfitting

Un léger overfitting est détectable à l'epoch 4 :
- `train_accuracy = 99.29%` vs `val_accuracy = 97.42%` -> gap de 1.87 points
- La val_loss se stabilise à 0.0842 tandis que la train_loss continue de chuter



## Difficultés rencontrées

- **Warning symlinks Windows** : HuggingFace Hub utilise des symlinks non supportés
  par défaut sur Windows. Résolu en ignorant le warning (fonctionnel sans symlinks,
  légèrement plus d'espace disque utilisé).
- **Premier run sans WandB** : le premier entraînement a tourné sans connexion WandB.
  Résolu en relançant `python train.py` après `wandb login`.
- **Temps d'entraînement sur CPU** : BERT (109M paramètres) est lent sur CPU
  (~30 min par epoch). Solution envisagée : Google Colab avec GPU T4.
- **max_length** : le 95e percentile à 732 mots impose max_length=512,
  ce qui tronque ~5% des textes les plus longs.

-
## Dépendances

| Package       | Usage                      |
----------------------------------------------
| `torch`       | Deep Learning              |
----------------------------------------------
| `transformers`| BERT + tokenizer           |
----------------------------------------------
| `gradio`      | Interface de démonstration |
----------------------------------------------
| `wandb`       | Tracking expériences       |
----------------------------------------------
| `scikit-learn`| F1-score, matrice confusion, split |
----------------------------------------------
| `pandas`      | Chargement CSV             |
----------------------------------------------
| `tqdm`        |Barres de progression       |
---------------------------------------------
| `matplotlib`  | Visualisations             |

#Répartition du travail

Chargement et inspection du dataset TextClassificationDataset (dataset.py), Architecture BertClassifier (model.py),Boucles train_epoch / eval_epoch Pipeline principal + WandB (train.py),Métriques et visualisations (utils.py) par **Fatou DIENG**
Interface Gradio (demo.py) par **Zana Baba Stephane Coulibaly**





*Projet Academique:Dakar Institute of Technology — Master 2 DIT 2026*
