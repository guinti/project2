import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PyQt5.QtCore import QThread, pyqtSignal
import os
import random
import warnings
from typing import Optional, Callable
import numpy as np
import librosa
from audio import is_audio_file, load_audio_file, save_audio_file, resample_audio, normalize_audio, trim_silence, simple_noise_reduction, ensure_length, extract_mfcc, extract_log_mel_spectrogram

SAMPLE_RATE = 16000 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}


class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, path, epochs, batch, lr, feature, stop_event):
        super().__init__()
        self.path, self.epochs, self.batch, self.lr, self.feature, self.stop_event = path, epochs, batch, lr, feature, stop_event

    def run(self):
        try:
            dataset = RavdessDataset(self.path, feature=self.feature)
            if len(dataset) == 0:
                self.error.emit("Пустой датасет")
                return

            n = len(dataset)
            idxs = list(range(n))
            random.shuffle(idxs)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_idxs = idxs[:n_train]
            val_idxs = idxs[n_train:n_train + n_val]

            train = Subset(dataset, train_idxs)
            val = Subset(dataset, val_idxs)

            model = SERModel().to(DEVICE)
            opt = optim.Adam(model.parameters(), lr=self.lr)
            crit = nn.CrossEntropyLoss()
            save_path = os.path.join(os.getcwd(), "ser_model_from_gui.pth")

            for ep in range(1, max(1, self.epochs) + 1):
                if self.stop_event.is_set():
                    self.error.emit("Обучение остановлено пользователем")
                    return

                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                loader = DataLoader(train, batch_size=self.batch, shuffle=True)
                for batch_idx, (x_batch, y_batch) in enumerate(loader):
                    if self.stop_event.is_set():
                        self.error.emit("Обучение остановлено пользователем")
                        return
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    opt.zero_grad()
                    outputs = model(x_batch)
                    loss = crit(outputs, y_batch)
                    loss.backward()
                    opt.step()

                    running_loss += loss.item() * x_batch.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == y_batch).sum().item()
                    total += x_batch.size(0)

                epoch_progress = int(ep / max(1, self.epochs) * 100)
                self.progress.emit(epoch_progress)

            torch.save({'model_state_dict': model.state_dict()}, save_path)
            self.finished.emit(save_path)

        except Exception as e:

            self.error.emit(str(e))


class RavdessDataset(Dataset):

    def __init__(self, root_dir: str, transform: Optional[callable] = None, sr: int = SAMPLE_RATE,
                 feature: str = 'mfcc', min_seconds: float = 1.0, max_frames: int = 128):
        self.root_dir = root_dir
        self.transform = transform
        self.sr = sr
        self.feature = feature
        self.min_seconds = min_seconds
        self.max_frames = max_frames

        self.filepaths = []
        self.labels = []

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if is_audio_file(f):
                    full = os.path.join(root, f)
                    lab = self._parse_ravdess_label(f)
                    if lab is None:
                        continue
                    if lab not in TARGET_EMOTIONS:
                        continue
                    self.filepaths.append(full)
                    self.labels.append(TARGET_EMOTIONS.index(lab))

        if len(self.filepaths) == 0:
            warnings.warn(f"Не найдено файлов в {root_dir} с эмоциями: {TARGET_EMOTIONS}")

    def _parse_ravdess_label(self, filename: str) -> Optional[str]:
        name = os.path.basename(filename)
        parts = name.split('-')
        if len(parts) < 3:
            return None
        emotion_code = parts[2]
        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        return emotion

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        y, sr = load_audio_file(path, sr=self.sr)
        y = normalize_audio(y)
        y = trim_silence(y)
        y = ensure_length(y, self.sr, self.min_seconds)
        y = simple_noise_reduction(y, self.sr)
        if self.feature == 'mfcc':
            feats = extract_mfcc(y, sr=self.sr)
        else:
            feats = extract_log_mel_spectrogram(y, sr=self.sr)
        if feats.shape[1] < self.max_frames:
            pad_width = self.max_frames - feats.shape[1]
            feats = np.pad(feats, ((0, 0), (0, pad_width)), mode='constant')
        else:
            feats = feats[:, :self.max_frames]
        feats = torch.from_numpy(feats).unsqueeze(0)
        return feats, torch.tensor(label, dtype=torch.long)


class SERModel(nn.Module):

    def __init__(self, in_channels: int = 1, n_classes: int = len(TARGET_EMOTIONS)):
        super(SERModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        self.adaptpool = nn.AdaptiveAvgPool2d((4, 4))
        fc_in = 64 * 4 * 4
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

