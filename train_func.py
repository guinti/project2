import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from Classes import RavdessDataset, SERModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device: torch.device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def compute_confusion_matrix(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    return cm

def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
    all_preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
    all_targets = np.concatenate(all_targets) if len(all_targets) > 0 else np.array([])
    loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return loss, acc, all_preds, all_targets

def train_from_ravdess(root_dir: str, epochs: int = 10, batch_size: int = 16, lr: float = 1e-3,
                       feature: str = 'mfcc', model_save_path: str = 'ser_model.pth') -> str:
    dataset = RavdessDataset(root_dir, feature=feature)
    n = len(dataset)
    if n == 0:
        raise RuntimeError('Датасет пуст или не найден')
    idxs = list(range(n))
    random.shuffle(idxs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SERModel(in_channels=1, n_classes=len(TARGET_EMOTIONS)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Эпоха {epoch}/{epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}",  flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'history': history}, model_save_path)

    checkpoint = torch.load(model_save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, DEVICE)
    cm = compute_confusion_matrix(preds, targets, len(TARGET_EMOTIONS))
    print(f"Точность на тесте: {test_acc:.4f}")
    print('Матрица ошибок:')
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=TARGET_EMOTIONS, yticklabels=TARGET_EMOTIONS, ax=ax)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истинно')
    fig.savefig('confusion_matrix.png')
    plt.close(fig)
    return model_save_path
