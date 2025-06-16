import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#############################
# Funciones de visualización#
#############################

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Dibuja las curvas de pérdida y accuracy por época,
    ajustadas al número real de épocas (soporta early stopping).
    """
    epochs_range = range(1, len(train_losses) + 1)

    # Curva de pérdida
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.title("Pérdida por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Curva de accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Accuracy")
    plt.plot(epochs_range, val_accs, label="Val Accuracy")
    plt.title("Accuracy por época")
    plt.xlabel("Época")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_learning_rate(lrs):
    """
    Visualiza la evolución del learning rate a lo largo del entrenamiento.
    Compatible con early stopping (usa la longitud real de `lrs`).
    """
    epochs_range = range(1, len(lrs) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, lrs)
    plt.title("Evolución del Learning Rate")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

#############################
# Funciones de entrenamiento#
#############################

def train_epoch(model: nn.Module,
                device: torch.device,
                train_loader: DataLoader,
                criterion,
                optimizer,
                l1_lambda=None,
                scheduler=None):
    """
    Entrena una época y devuelve (train_loss, train_acc, current_lr).
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader, start=1):
        images, labels, metadata = batch  
        images, labels = images.to(device), labels.to(device)
        metadata = metadata.to(device)

        optimizer.zero_grad()
        outputs = model(images, metadata)  
        loss = criterion(outputs, labels)

        if l1_lambda:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100.0 * correct / total

    # Scheduler (se aplica al final de la época)
    if scheduler:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = optimizer.param_groups[0]['lr']

    return train_loss, train_acc, current_lr


def eval_epoch(model: nn.Module,
               device: torch.device,
               val_loader: DataLoader,
               criterion):
    """
    Evalúa en validación y devuelve (val_loss, val_acc).
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, metadata in val_loader: 
            images, labels = images.to(device), labels.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)  
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = outputs.max(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device):
    """
    Evalúa el modelo en test y devuelve:
    - accuracy (%)
    - all_labels (lista)
    - all_preds (lista)
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, metadata in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            outputs = model(images, metadata)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    return accuracy, all_labels, all_preds










