import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay


# Seed globale    # Fixe toutes les seeds pour la reproductibilité totale. À appeler une seule fois au début de main()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f" Seed fixée : {seed}")


# Sauvegarde les poids du meilleur modèle selon la val_loss

def save_checkpoint(model, run_name: str, val_loss: float) -> str: # Sauvegarde les poids du meilleur modèle selon la val_loss
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/best_model_{run_name}.pth"
    torch.save(model.state_dict(), path)
    print(f"  ✓ Meilleur modèle sauvegardé → {path}  (val_loss={val_loss:.4f})")
    return path


# Matrice de confusion

def plot_confusion_matrix(all_labels:  list, 
                          all_preds:   list,
                          class_names: list,
                          run_name:    str) -> str: #  Génère, sauvegarde et logue la matrice de confusion dans WandB.
    
    cm   = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = class_names
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(f"Matrice de confusion — {run_name}",
                 fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    save_path = f"results/confusion_matrix_{run_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Log dans WandB
    if wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(save_path)})

    print(f" Matrice de confusion  {save_path}")
    return save_path

# Courbes d'apprentissage

def plot_learning_curves(history: dict, run_name: str) -> str: # Trace les courbes train/val superposées : loss et accuracy.Sauvegarde en local et logue dans WandB.
    
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Courbes d'apprentissage — {run_name}",
                 fontsize=13, fontweight='bold')

    #Loss 
    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             color='#2E74B5', linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss",
             color='#FF6B6B', linewidth=2, linestyle='--')
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    #Accuracy 
    ax2.plot(epochs, history["train_accuracy"], label="Train Accuracy",
             color='#2E74B5', linewidth=2)
    ax2.plot(epochs, history["val_accuracy"], label="Val Accuracy",
             color='#FF6B6B', linewidth=2, linestyle='--')
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    save_path = f"results/learning_curves_{run_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    if wandb.run is not None:
        wandb.log({"learning_curves": wandb.Image(save_path)})

    print(f" Courbes sauvegardées {save_path}")
    return save_path


# comptage des parametres
def count_parameters(model) -> dict: # Compte les paramètres totaux et entraînables du modèle.
    
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"\n Paramètres du modèle :")
    print(f"   Total       : {total:>12,}")
    print(f"   Entraînable : {trainable:>12,}")
    print(f"   Gelés       : {frozen:>12,}")

    return {"total": total, "trainable": trainable, "frozen": frozen}