import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig






# Modele Bert classification
class BertClassifier(nn.Module):
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 n_classes:  int = 2,
                 dropout:  float = 0.3):
        super().__init__()

        #Chargement du backbone BERT pré-entraîné 
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels          = n_classes,
            hidden_dropout_prob = dropout,
            ignore_mismatched_sizes = True
        )

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        # outputs.logits : [batch_size, n_classes]
        return outputs.logits


# Utilitaire-Chargement du modèle sauvegardé

def load_model(checkpoint_path: str,
               model_name:      str,
               n_classes:       int,
               device:          torch.device) -> BertClassifier:
    model = BertClassifier(model_name=model_name, n_classes=n_classes)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)
    model.eval()
    print(f" Modèle chargé depuis : {checkpoint_path}")
    return model


# Test du modèle


if __name__ == "__main__":

    MODEL_NAME = "bert-base-uncased"
    N_CLASSES  = 5    # Sera mis à jour après inspection du dataset
    BATCH_SIZE = 2
    MAX_LENGTH = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Chargement du modèle
    print(f"\n Chargement de {MODEL_NAME} ...")
    model = BertClassifier(model_name=MODEL_NAME, n_classes=N_CLASSES)
    model = model.to(device)
    print("Modèle chargé")

    # Test avec des données factices
    dummy_input_ids      = torch.randint(0, 1000, (BATCH_SIZE, MAX_LENGTH)).to(device)
    dummy_attention_mask = torch.ones(BATCH_SIZE, MAX_LENGTH, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)

    print(f"\n Test forward pass :")
    print(f"   Input  : {list(dummy_input_ids.shape)}")
    print(f"   Output : {list(logits.shape)}   ← [batch, n_classes]")

    # Nombre de paramètres
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Paramètres :")
    print(f"   Total       : {total:>12,}")
    print(f"   Entraînable : {trainable:>12,}")