import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dataset = load_dataset("crows_pairs", split="test")



def process(example):
    return [
        {"text": example["sent_more"], "label": 0},  # biased
        {"text": example["sent_less"], "label": 1},  # less biased
    ]

processed = sum([process(ex) for ex in dataset], [])
texts = [ex["text"] for ex in processed]
labels = [ex["label"] for ex in processed]

df = pd.DataFrame({"text": texts, "label": labels})
split_idx = int(0.9 * len(df))
df_train = df.iloc[:split_idx].reset_index(drop=True)
df_eval = df.iloc[split_idx:].reset_index(drop=True)
df_train.to_csv("crowspair_train.csv", index=False)
df_eval.to_csv("crowspair_eval.csv", index=False)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(df_train["text"].tolist(), truncation=True, padding=True)
eval_encodings = tokenizer(df_eval["text"].tolist(), truncation=True, padding=True)


class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = BiasDataset(train_encodings, df_train["label"].tolist())
eval_dataset = BiasDataset(eval_encodings, df_eval["label"].tolist())

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}



training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

preds_output = trainer.predict(eval_dataset)
probs = torch.nn.functional.softmax(torch.tensor(preds_output.predictions), dim=1)
pred_labels = torch.argmax(probs, dim=1)
confidence_scores = probs.max(dim=1).values

for i in range(10):
    print(f"Text: {df_eval['text'][i]}")
    print(f"Predicted: {'Biased' if pred_labels[i] == 1 else 'Less Biased'} | Confidence: {confidence_scores[i].item():.4f}\n")
