import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_classification_metrics


class Trainer:
    """
    Manage model training, evaluation, and checkpointing.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train and evaluate.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    criterion : torch.nn.Module
        Loss function used for optimization.
    device : torch.device
        Device used for training and evaluation.
    output_dir : str
        Directory where checkpoints and history files are saved.

    Attributes
    ----------
    model : torch.nn.Module
        Model being trained and evaluated.
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updates.
    criterion : torch.nn.Module
        Loss function used during training and evaluation.
    device : torch.device
        Device used for computation.
    output_dir : pathlib.Path
        Output directory for saved artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Dataloader for the training split.

        Returns
        -------
        dict
            Dictionary containing average training loss and accuracy
            for the epoch.
        """
        self.model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": running_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset split.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Dataloader for the validation or test split.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics, including loss,
            accuracy, macro F1, confusion matrix, and AUC when available.
        """
        self.model.eval()

        running_loss = 0.0
        total = 0

        all_labels = []
        all_preds = []
        all_probs = []

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics["loss"] = running_loss / total

        return metrics

    def fit(self, train_loader, val_loader, num_epochs: int):
        """
        Train the model and evaluate on the validation set across epochs.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Dataloader for the training split.
        val_loader : torch.utils.data.DataLoader
            Dataloader for the validation split.
        num_epochs : int
            Number of training epochs.

        Returns
        -------
        dict
            Dictionary containing the full training history, best validation
            accuracy, and path to the best saved model checkpoint.
        """
        history = []
        best_val_acc = -1.0
        best_model_path = self.output_dir / "best_model.pt"

        for epoch in range(num_epochs):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            epoch_result = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_result)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f} "
                f"val_auc={val_metrics['auc_ovr'] if val_metrics['auc_ovr'] is not None else 'N/A'}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(self.model.state_dict(), best_model_path)

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return {
            "history": history,
            "best_val_accuracy": best_val_acc,
            "best_model_path": str(best_model_path),
        }
