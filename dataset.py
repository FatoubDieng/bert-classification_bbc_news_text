import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer



# Seed - Reproductibilité

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#changement et inspection de notre dataset

def load_and_inspect(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print("=" * 65)
    print("  INSPECTION — BBC News Text Classification")
    print("=" * 65)

    #Aperçu général
    print(f"\n Dimensions         : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f" Colonnes           : {df.columns.tolist()}")
    print(f" Valeurs manquantes : {df.isnull().sum().sum()}")

    # Supprimer les NaN dans les colonnes clés
    df = df.dropna(subset=["text", "labels"]).reset_index(drop=True)
    print(f"Après nettoyage    : {len(df)} exemples")

    #  Classes 
    classes      = sorted(df["labels"].unique().tolist())
    n_classes    = len(classes)
    print(f"\n Nombre de classes  : {n_classes}")
    print(f"   Classes            : {classes}")

    # Distribution des classes 
    print(f"\n{'Classe':<20} {'Exemples':>10}   {'%':>7}")
    print("-" * 42)
    counts = df["labels"].value_counts()
    for cls, cnt in counts.items():
        print(f"{str(cls):<20} {cnt:>10}   {cnt/len(df)*100:>6.1f}%")

    # Détection du déséquilibre
    ratio = counts.max() / counts.min()
    print(f"\n  Ratio max/min : {ratio:.2f}")
    if ratio > 2:
        print("  DÉSÉQUILIBRE > 2:1 !")
        print(" Stratégie : class_weight dans CrossEntropyLoss")
    else:
        print("  Classes équilibrées (ratio ≤ 2:1)")

    #Longueur des textes (en mots — approximation avant tokenization) 
    df["text_len"] = df["text"].apply(lambda x: len(str(x).split()))
    print(f"\n Longueur des textes (en mots) :")
    print(f"   Min    : {df['text_len'].min()}")
    print(f"   Max    : {df['text_len'].max()}")
    print(f"   Moy    : {df['text_len'].mean():.0f}")
    print(f"   Médiane: {df['text_len'].median():.0f}")
    print(f"   95e pc : {df['text_len'].quantile(0.95):.0f}")

    # Recommandation max_length
    p95 = df["text_len"].quantile(0.95)
    if p95 < 100:
        recommande = 128
    elif p95 < 200:
        recommande = 256
    else:
        recommande = 512
    print(f"\n  max_length recommandé : {recommande} tokens")

    #5 exemples visuels
    print(f"\n{'='*65}")
    print("  5 EXEMPLES DE TEXTES")
    print(f"{'='*65}")
    sample = df.sample(5, random_state=SEED)
    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"\n[{i+1}] Label : {row['labels']}")
        texte = str(row['text'])
        print(f"     Texte : {texte[:200]}{'...' if len(texte) > 200 else ''}")

    print("\n" + "=" * 65)

    #Visualisations
    _plot_distribution(df, counts)
    _plot_text_lengths(df)

    return df


def _plot_distribution(df: pd.DataFrame, counts: pd.Series) -> None:
    """Trace la distribution des classes. Sauvegarde class_distribution.png"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    bars    = ax.bar(counts.index.astype(str), counts.values,
                     color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title("Distribution des classes — BBC News", fontsize=13, fontweight='bold')
    ax.set_xlabel("Classe (label)")
    ax.set_ylabel("Nombre d'exemples")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n Distribution sauvegardée class_distribution.png")


def _plot_text_lengths(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["text_len"], bins=50, color='#2E74B5', edgecolor='white', alpha=0.8)
    ax.axvline(df["text_len"].mean(),   color='red',    linestyle='--', label=f'Moy={df["text_len"].mean():.0f}')
    ax.axvline(df["text_len"].quantile(0.95), color='orange', linestyle='--',
               label=f'95e pc={df["text_len"].quantile(0.95):.0f}')
    ax.set_title("Distribution des longueurs de textes (en mots)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Nombre de mots")
    ax.set_ylabel("Fréquence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("text_lengths.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(" Longueurs sauvegardées text_lengths.png")


# SPLIT 80/20 STRATIFIÉ

def split_dataframe(df: pd.DataFrame,
                    train_ratio: float = 0.80,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size    = 1 - train_ratio,
        random_state = seed,
        stratify     = df["labels"]   #Préserve la distribution
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"\n Split stratifié 80/20 :")
    print(f"   Train : {len(train_df):>5} exemples")
    print(f"   Val   : {len(val_df):>5} exemples")
    print(f"   Distribution train : {dict(train_df['labels'].value_counts())}")
    print(f"   Distribution val   : {dict(val_df['labels'].value_counts())}")

    return train_df, val_df

class TextClassificationDataset(Dataset): # Classe Dataset PyTorch personnalisé pour la classification de texte avec BERT
    def __init__(self,
                 df:        pd.DataFrame,
                 tokenizer,
                 max_length: int  = 128,
                 label2idx:  dict = None):

        self.texts     = df["text"].tolist()
        self.labels    = df["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Mapping label index
        if label2idx is None:
            unique_labels = sorted(df["labels"].unique().tolist())
            self.label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def __len__(self) -> int:
        
        return len(self.texts) #Retourne le nombre total d'exemples

    def __getitem__(self, idx: int) -> dict:  #Tokenize un texte et retourne les tenseurs BERT
        text  = str(self.texts[idx])
        label = self.label2idx[self.labels[idx]]

        # Tokenization avec padding et truncation
        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",    # Padding jusqu'à max_length
            truncation      = True,            # Tronquer si trop long
            return_tensors  = "pt"             # Retourner des tenseurs PyTorch
        )

        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),       # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),   # [max_length]
            "label"         : torch.tensor(label, dtype=torch.long)    # scalaire
        }

    def get_class_weights(self) -> torch.Tensor: #fonction pour calculer les poids inverses de fréquence par classe.
        counts = defaultdict(int)
        for lbl in self.labels:
            counts[self.label2idx[lbl]] += 1
        total   = len(self.labels)
        weights = [total / counts[i] for i in range(len(self.label2idx))]
        return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":

    CSV_PATH   = "./data/bbc_news_text.csv"
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128   

    #Inspection
    df = load_and_inspect(CSV_PATH)

    #Split 80/20 stratifié 
    train_df, val_df = split_dataframe(df, train_ratio=0.80)

    #Tokenizer
    print(f"\n Chargement du tokenizer : {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(" Tokenizer chargé")

    # Mapping label index (construit sur le train, appliqué au val)
    unique_labels = sorted(df["labels"].unique().tolist())
    label2idx     = {lbl: i for i, lbl in enumerate(unique_labels)}
    print(f"\n Mapping labels : {label2idx}")

    # Datasets PyTorch 
    train_dataset = TextClassificationDataset(
        train_df, tokenizer, max_length=MAX_LENGTH, label2idx=label2idx
    )
    val_dataset = TextClassificationDataset(
        val_df, tokenizer, max_length=MAX_LENGTH, label2idx=label2idx
    )

    print(f"\n Datasets PyTorch prêts :")
    print(f"   Train : {len(train_dataset):>5} exemples")
    print(f"   Val   : {len(val_dataset):>5} exemples")
    print(f"   Exemple batch  {train_dataset[0].keys()}")

    #Poids de classes
    weights = train_dataset.get_class_weights()
    print(f"\n Class weights : {weights.tolist()}")