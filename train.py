import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb

from dataset import TextClassificationDataset, load_and_inspect, split_dataframe
from model   import BertClassifier
from utils   import (set_seed, save_checkpoint, plot_confusion_matrix,
                     plot_learning_curves, count_parameters)


# Boucle d'entrainements

def train_epoch(model,
                loader:    DataLoader,
                criterion: nn.Module,
                optimizer,
                scheduler,
                device:    torch.device) -> tuple[float, float]:
   
    model.train()    # Active Dropout en mode entraînement

    total_loss = 0.0
    correct    = 0
    total      = 0

    loop = tqdm(loader, desc="  Train", leave=False, colour='blue')

    for batch in loop:
        # Envoyer sur le device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        #  Forward pass 
        #  attention_mask 
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels)

        #  Backward pass 
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — évite les gradients explosifs avec BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()   # Scheduler linéaire : step par batch

        # Métriques 
        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy



# Boucle d'evaluation

def eval_epoch(model,
               loader:    DataLoader,
               criterion: nn.Module,
               device:    torch.device) -> tuple[float, float, float]:
  #evalue le model sur le datast val
    model.eval()    # Désactive Dropout en mode évaluation

    total_loss      = 0.0
    correct         = 0
    total           = 0
    all_labels      = []
    all_predictions = []

    loop = tqdm(loader, desc="  Val  ", leave=False, colour='green')

    with torch.no_grad():   # Pas de gradients pendant la validation
        for batch in loop:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            #Forward pass uniquement 
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # Pour F1-score et matrice de confusion
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    f1       = f1_score(all_labels, all_predictions, average='macro') * 100

    return avg_loss, accuracy, f1



# Pipline principal

def main():
    # Hyperparamètres 
    CSV_PATH   = "./data/bbc_news_text.csv"
    MODEL_NAME = "bert-base-uncased"   # Anglais
    MAX_LENGTH = 128     # Justification : 95e percentile < 200 mots -> 128 tokens suffisent
    BATCH_SIZE = 16      # 16 pour éviter les problèmes mémoire CPU/GPU
    N_EPOCHS   = 4       # BERT converge vite pour eviter l'overfitting
    LR         = 2e-5    # LR typique BERT — > 5e-5 -> catastrophic forgetting
    WARMUP     = 0.1     # 10% des steps en warmup
    SEED       = 42
    RUN_NAME   = "bert_bbc_classification"

    # Seed et device 
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    #  WandB init 
    wandb.init(
        project="dit-2026-bert-nlp",
        name=RUN_NAME,
        config={
            "model_name"    : MODEL_NAME,
            "max_length"    : MAX_LENGTH,
            "batch_size"    : BATCH_SIZE,
            "epochs"        : N_EPOCHS,
            "learning_rate" : LR,
            "warmup_ratio"  : WARMUP,
            "optimizer"     : "AdamW",
            "loss"          : "CrossEntropyLoss",
            "seed"          : SEED,
            "device"        : str(device),
        }
    )

    #Chargement et inspection du dataset 
    df = load_and_inspect(CSV_PATH)

    #  Split 80/20 stratifié 
    train_df, val_df = split_dataframe(df, train_ratio=0.80, seed=SEED)

    #  Mapping label index 
    unique_labels = sorted(df["labels"].unique().tolist())
    label2idx     = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx2label     = {v: k for k, v in label2idx.items()}
    n_classes     = len(unique_labels)

    print(f"\n  {n_classes} classes : {label2idx}")
    wandb.config.update({"n_classes": n_classes, "classes": unique_labels})

    #  Tokenizer 
    print(f"\n Chargement du tokenizer : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets et DataLoaders 
    train_dataset = TextClassificationDataset(
        train_df, tokenizer, max_length=MAX_LENGTH, label2idx=label2idx
    )
    val_dataset = TextClassificationDataset(
        val_df, tokenizer, max_length=MAX_LENGTH, label2idx=label2idx
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Modèle BERT 
    print(f"\n Chargement du modèle : {MODEL_NAME}")
    model = BertClassifier(model_name=MODEL_NAME, n_classes=n_classes)
    model = model.to(device)

    params = count_parameters(model)
    wandb.config.update(params)

    #  Loss 
    # Class weights si déséquilibre > 2:1
    weights   = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimiseur AdamW 
    # AdamW : Adam + weight decay (meilleur que Adam pour BERT)
    # weight_decay=0.01 : régularisation L2 sur les poids (sauf biais et LayerNorm)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params"      : [p for n, p in model.named_parameters()
                             if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params"      : [p for n, p in model.named_parameters()
                             if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)

    #  Scheduler linéaire avec warmup 
    total_steps  = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )
    print(f"\n Scheduler : {warmup_steps} warmup steps / {total_steps} total steps")

    #Boucle d'entraînement 
    print(f"\n Début entraînement : {RUN_NAME} | {N_EPOCHS} epochs | {device}\n")

    best_val_loss = float("inf")
    history = {
        "train_loss"     : [],
        "val_loss"       : [],
        "train_accuracy" : [],
        "val_accuracy"   : [],
        "val_f1_score"   : [],
    }

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validation
        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, criterion, device
        )

        # Affichage
        print(
            f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%\n"
            f"  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%"
        )

        # Logging WandB 
        wandb.log({
            "epoch"          : epoch + 1,
            "train_loss"     : train_loss,
            "val_loss"       : val_loss,
            "train_accuracy" : train_acc,
            "val_accuracy"   : val_acc,
            "val_f1_score"   : val_f1,
            "learning_rate"  : optimizer.param_groups[0]["lr"],
        })

        # Historique local
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_f1_score"].append(val_f1)

        #Sauvegarde meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_checkpoint(model, RUN_NAME, best_val_loss)
            wandb.save(path)

    #Matrice de confusion finale
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            preds          = torch.argmax(
                model(input_ids=input_ids, attention_mask=attention_mask), dim=1
            )
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    plot_confusion_matrix(all_labels, all_preds, unique_labels, RUN_NAME)
    plot_learning_curves(history, RUN_NAME)

    #Résumé WandB 
    wandb.summary["best_val_loss"]      = best_val_loss
    wandb.summary["final_val_accuracy"] = history["val_accuracy"][-1]
    wandb.summary["final_val_f1"]       = history["val_f1_score"][-1]

    print(f"\n Entraînement terminé | Meilleure val_loss : {best_val_loss:.4f}")
    wandb.finish()

    # Sauvegarder le tokenizer et le label mapping pour la démo
    tokenizer.save_pretrained("checkpoints/tokenizer")
    import json
    with open("checkpoints/label2idx.json", "w") as f:
        json.dump(label2idx, f)
    print("Tokenizer et labels sauvegardés dans checkpoints/")



if __name__ == "__main__":
    main()