import os
import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "roberta_model"
TOKENIZER_SAVE_PATH = MODELS_DIR / "tokenizer"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = ' '.join(text.split())
    return text

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    print("=" * 60)
    print("Starting RoBERTa Model Training Pipeline (5 Epochs, Max Len 256)")
    print("=" * 60)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[GPU] Device Name: {torch.cuda.get_device_name(0)}")

    # 2. Data Loading & Labeling
    full_dataset_path = DATA_DIR / "fake_reviews_dataset.csv"
    test_reviews_path = DATA_DIR / "TestReviews.csv"

    dfs = []
    if full_dataset_path.exists():
        print(f"[Data] Loading main review dataset ({full_dataset_path.name})...")
        df_full = pd.read_csv(full_dataset_path)
        df_full["label_num"] = df_full["label"].map({"CG": 0, "OR": 1})
        df_full["clean_text"] = df_full["text_"].fillna("").apply(clean_text)
        df_full = df_full.dropna(subset=["label_num"])
        dfs.append(df_full[["clean_text", "label_num"]].rename(columns={"label_num": "label"}))

    if test_reviews_path.exists():
        print(f"[Data] Loading secondary review dataset ({test_reviews_path.name})...")
        df_test = pd.read_csv(test_reviews_path)
        df_test["clean_text"] = df_test["review"].fillna("").apply(clean_text)
        df_test = df_test.rename(columns={"class": "label"})
        dfs.append(df_test[["clean_text", "label"]])

    df = pd.concat(dfs, ignore_index=True)
    df["cleaned_text"] = df["clean_text"]
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    print(f"[Data] Total reviews combined for RoBERTa fine-tuning: {len(df):,}")


    # Train / Test Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )

    print(f"[Data] Training Set Size: {len(train_texts)}, Test Set Size: {len(test_texts)}")

    # 3. Tokenizer & Dataset Initialization
    print("[Tokenizer] Initializing roberta-base tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    batch_size = 16 if torch.cuda.is_available() else 8
    max_len = 256

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model Initialization
    print("[Model] Loading pretrained roberta-base for sequence classification...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)

    epochs = 5
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    TOKENIZER_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

    best_val_accuracy = 0.0

    # 5. Training Loop with Best Model Checkpointing
    print("\n[Training] Starting fine-tuning for 5 epochs...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation after each epoch
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        epoch_acc = accuracy_score(val_targets, val_preds)
        print(f"---> Epoch {epoch+1} Loss: {avg_train_loss:.4f} | Validation Accuracy: {epoch_acc * 100:.2f}%")

        # Save Best Model Checkpoint
        if epoch_acc > best_val_accuracy:
            print(f"[Checkpoint] Validation accuracy improved ({best_val_accuracy * 100:.2f}% -> {epoch_acc * 100:.2f}%). Saving best model...")
            best_val_accuracy = epoch_acc
            model.save_pretrained(MODEL_SAVE_PATH)

    print("=" * 60)
    print(f"[Success] RoBERTa fine-tuning completed. Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    train()
