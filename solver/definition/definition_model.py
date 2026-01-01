from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch
import os

# Point to: cryptic_solver/data/definition_model
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "definition_model"
)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()


def extract_definition_tokens(clue_text):
    """
    Returns a list of tokens that the model predicts are part of the definition.
    """
    encoded = tokenizer(clue_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    tokens = tokenizer.tokenize(clue_text)
    token_predictions = predictions[1:len(tokens)+1]  # strip CLS/SEP

    def_tokens = [
        tok for tok, lbl in zip(tokens, token_predictions)
        if lbl == 1
    ]
    return def_tokens
print("MODEL_DIR:", MODEL_DIR)
