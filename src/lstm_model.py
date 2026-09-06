from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocabulary(
    texts,
    min_frequency: int = 2,
    max_vocab_size: int = 20_000,
) -> Dict[str, int]:
    """
    Build vocabulary from training text only.
    """
    counter = Counter()

    for text in texts:
        counter.update(text.split())

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, frequency in counter.most_common(
        max_vocab_size - 2
    ):
        if frequency >= min_frequency:
            vocab[word] = len(vocab)

    return vocab


def text_to_sequence(
    text: str,
    vocab: Dict[str, int],
) -> List[int]:
    """
    Convert text into vocabulary indices.
    """
    sequence = [
        vocab.get(word, vocab[UNK_TOKEN])
        for word in text.split()
    ]

    if not sequence:
        sequence = [vocab[UNK_TOKEN]]

    return sequence


def encode_and_pad(
    text: str,
    vocab: Dict[str, int],
    max_len: int,
) -> Tuple[List[int], int]:
    """
    Convert text to indices, truncate and right-pad.
    """
    sequence = text_to_sequence(
        text,
        vocab,
    )

    sequence = sequence[:max_len]

    true_length = len(sequence)

    padded = sequence + [
        vocab[PAD_TOKEN]
    ] * (max_len - true_length)

    return padded, true_length


class MovieReviewDataset(Dataset):
    """
    PyTorch dataset for sentiment classification.
    """

    def __init__(
        self,
        sequences,
        lengths,
        labels,
    ):
        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long,
        )

        self.lengths = torch.tensor(
            lengths,
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.lengths[index],
            self.labels[index],
        )


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary sentiment classifier.
    """

    def __init__(
        self,
        vocabulary_size: int,
        padding_index: int,
        embedding_dimension: int = 100,
        hidden_dimension: int = 128,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_index,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            dropout_rate
        )

        self.fc = nn.Linear(
            hidden_dimension,
            1,
        )

    def forward(
        self,
        reviews,
        lengths,
    ):
        embedded = self.embedding(reviews)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        final_hidden = self.dropout(
            hidden[-1]
        )

        logits = self.fc(
            final_hidden
        )

        return logits.squeeze(1)


def train_lstm(
    model,
    train_loader,
    device,
    epochs: int = 5,
    learning_rate: float = 0.001,
):
    """
    Train the LSTM model (basic version without validation).
    """
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for reviews, lengths, labels in train_loader:
            reviews = reviews.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                reviews,
                lengths,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() *
                len(labels)
            )

        average_loss = (
            total_loss /
            len(train_loader.dataset)
        )

        history.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss,
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {average_loss:.4f}"
        )

    return history


def train_lstm_with_validation(
    model,
    train_loader,
    val_loader,
    device,
    epochs=5,
    learning_rate=0.001,
    patience=3,
):
    """
    Train LSTM with validation loss monitoring and early stopping.
    Returns the best model (state_dict loaded) and training history.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
    }
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        total_train_loss = 0.0
        for reviews, lengths, labels in train_loader:
            reviews = reviews.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(reviews, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(labels)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for reviews, lengths, labels in val_loader:
                reviews = reviews.to(device)
                labels = labels.to(device)
                logits = model(reviews, lengths)
                loss = criterion(logits, labels)
                total_val_loss += loss.item() * len(labels)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss: {avg_train_loss:.4f}, "
            f"val loss: {avg_val_loss:.4f}"
        )

        # ---- Early stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def predict_lstm(
    model,
    data_loader,
    device,
):
    """
    Generate binary predictions from an LSTM model.
    """
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for reviews, lengths, labels in data_loader:
            reviews = reviews.to(device)

            logits = model(
                reviews,
                lengths,
            )

            predicted = (
                logits >= 0
            ).long()

            predictions.extend(
                predicted.cpu().tolist()
            )

            actual.extend(
                labels.long().tolist()
            )

    return (
        np.array(actual),
        np.array(predictions),
    )