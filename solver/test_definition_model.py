import json
from torch.utils.data import Dataset
import torch
from transformers import (
    DistilBertForTokenClassification,
    Trainer,
    TrainingArguments
)

TRAIN_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\definition_training.jsonl"
MODEL_OUT = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\definition_model"


class DefDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        return {
            "input_ids": torch.tensor(x["input_ids"]),
            "attention_mask": torch.tensor(x["attention_mask"]),
            "labels": torch.tensor(x["labels"])
        }


class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # pad inputs (already tokenized, so just pad manually)
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
            attention_masks.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels.append(
                torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels)
        }


def main():
    dataset = DefDataset(TRAIN_PATH)

    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # dummy tokenizer substitute (we do manual padding)
    tokenizer = None
    collator = PadCollator(tokenizer)

    args = TrainingArguments(
        output_dir=MODEL_OUT,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        save_steps=2000,          # changed
        save_total_limit=1,
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(MODEL_OUT)

    print("Model saved to:", MODEL_OUT)


if __name__ == "__main__":
    main()
