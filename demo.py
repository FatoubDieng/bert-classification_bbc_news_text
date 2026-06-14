
import json
import torch
import gradio as gr
from transformers import AutoTokenizer

from model import load_model


# Configuration
MODEL_NAME      = "bert-base-uncased"
CHECKPOINT_PATH = "checkpoints/best_model_bert_bbc_classification.pth"
TOKENIZER_PATH  = "checkpoints/tokenizer"
LABEL2IDX_PATH  = "checkpoints/label2idx.json"
MAX_LENGTH      = 128

#  Chargement du device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Chargement du label mapping 
with open(LABEL2IDX_PATH, "r") as f:
    label2idx = json.load(f)
idx2label = {int(v): k for k, v in label2idx.items()}
n_classes = len(label2idx)

#  Chargement du tokenizer 
print(f" Chargement du tokenizer depuis {TOKENIZER_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

#  Chargement du modèle 
print(f"Chargement du modèle depuis {CHECKPOINT_PATH} ...")
model = load_model(CHECKPOINT_PATH, MODEL_NAME, n_classes, device)
print(" Modèle prêt pour l'inférence\n")


# Fonction de prediction

def predict(text: str) -> dict:
    if not text or not text.strip():
        return {label: 0.0 for label in label2idx.keys()}

    # Tokenization
    encoding = tokenizer(
        text,
        max_length     = MAX_LENGTH,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inférence
    model.eval()
    with torch.no_grad():
        logits       = model(input_ids=input_ids, attention_mask=attention_mask)
        probs        = torch.softmax(logits, dim=1).squeeze(0)

    # Formatage pour Gradio
    result = {
        idx2label[i]: float(probs[i])
        for i in range(n_classes)
    }

    return result



# Exemples pre-remplis
EXAMPLES = [
    ["Arsenal secured a dramatic late victory against Chelsea last night, "
     "with the winning goal scored in the 89th minute by their top striker. "
     "The result keeps them at the top of the Premier League table."],

    ["The Chancellor announced a series of new measures to tackle inflation, "
     "including changes to interest rates and government spending cuts. "
     "Analysts are divided on whether these policies will be effective."],

    ["Scientists at Oxford University have developed a groundbreaking AI system "
     "capable of diagnosing rare diseases from medical scans with 97% accuracy, "
     "potentially revolutionizing early detection in healthcare."],

    ["The latest blockbuster film broke box office records this weekend, "
     "earning over $200 million globally in its opening days. "
     "Critics have praised the director's vision and the lead actor's performance."],

    ["Apple unveiled its new iPhone lineup featuring enhanced camera capabilities "
     "and a faster processor. The company also announced updates to its "
     "operating system with new AI-powered features for productivity."],
]


#Interface Gradio

def build_interface() -> gr.Blocks:
  
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interface:

        # En-tête 
        gr.Markdown("""
        # BBC News Text Classifier
        **Fine-tuning de BERT pour la classification de textes BBC News**

        Ce modèle a été entraîné sur le dataset **BBC News Text Complexity**
        avec le modèle `bert-base-uncased` fine-tuné en PyTorch pur.

        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Entrée 
                text_input = gr.Textbox(
                    label       = " Entrez un texte BBC News",
                    placeholder = "Collez ou tapez un texte d'actualité en anglais...",
                    lines       = 6,
                    max_lines   = 20,
                )

                with gr.Row():
                    submit_btn = gr.Button(" Analyser", variant="primary", scale=2)
                    clear_btn  = gr.Button(" Effacer",  variant="secondary", scale=1)

            with gr.Column(scale=2):
                #  Sortie 
                label_output = gr.Label(
                    label     = " Probabilités par classe",
                    num_top_classes = n_classes,
                )

        #  Exemples 
        gr.Markdown("### Exemples à tester")
        gr.Examples(
            examples   = EXAMPLES,
            inputs     = text_input,
            outputs    = label_output,
            fn         = predict,
            cache_examples = False,
            label      = "Cliquez sur un exemple pour le charger"
        )

        # Informations de notre modèle 
        gr.Markdown(f"""
        ---
        ### ℹ️ Informations sur le modèle
        | Paramètre | Valeur |
        |---|---|
        | **Modèle de base** | `{MODEL_NAME}` |
        | **Dataset** | BBC News Text Complexity Summarization |
        | **Max length** | {MAX_LENGTH} tokens |
        | **Classes** | {', '.join(label2idx.keys())} |
        | **Auteur** | Fatou DIENG - Zana Baba Stephane Coulibaly |
        """)

        #  Actions 
        submit_btn.click(
            fn      = predict,
            inputs  = text_input,
            outputs = label_output,
        )
        clear_btn.click(
            fn      = lambda: ("", None),
            inputs  = None,
            outputs = [text_input, label_output],
        )
        text_input.submit(
            fn      = predict,
            inputs  = text_input,
            outputs = label_output,
        )

    return interface



# Lancement

if __name__ == "__main__":
    print(" Lancement de l'interface Gradio...")
    print(f"   Classes : {list(idx2label.values())}")
    print(f"   Device  : {device}\n")

    interface = build_interface()
    interface.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = True,    # Mettre True pour un lien public temporaire
        show_error  = True,
    )