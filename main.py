import os
import sys
import io
import csv
import time
import json
import random
import warnings
import threading
import tempfile
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Optional, Dict, Tuple, List
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader, Subset
from matplotlib.backends.backend_pdf import PdfPages
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QTextEdit, QProgressBar, QComboBox,
                             QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from audio import is_audio_file, load_audio_file, save_audio_file, resample_audio, normalize_audio, trim_silence, simple_noise_reduction, ensure_length, extract_mfcc, extract_log_mel_spectrogram

from Classes import TrainingThread, RavdessDataset, SERModel

from generate_report import generate_report_pdf, generate_report_csv

from train_func import train_from_ravdess

SAMPLE_RATE = 16000 

TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def infer_on_audio_file(model: nn.Module, path: str, device: torch.device, feature: str = 'mfcc') -> Dict[str, float]:
    y, sr = load_audio_file(path, sr=SAMPLE_RATE)
    y = normalize_audio(y)
    y = trim_silence(y)
    y = ensure_length(y, sr, min_seconds=1.0)
    y = simple_noise_reduction(y, sr)
    if feature == 'mfcc':
        feats = extract_mfcc(y, sr=sr)
    else:
        feats = extract_log_mel_spectrogram(y, sr=sr)
    max_frames = 128
    if feats.shape[1] < max_frames:
        pad_width = max_frames - feats.shape[1]
        feats = np.pad(feats, ((0, 0), (0, pad_width)), mode='constant')
    else:
        feats = feats[:, :max_frames]
    tensor = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze()
    result = {TARGET_EMOTIONS[i]: float(probs[i]) for i in range(len(TARGET_EMOTIONS))}
    return result


class SERApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Распознавание эмоций в речи')
        self.resize(1000, 700)
        self.model = None
        self.model_path = None
        self.feature = 'mfcc'
        self.current_audio_path = None
        self.current_processed_audio = None
        self.last_prediction = None
        self.training_process = None
        self._setup_ui()
        self.player = QMediaPlayer()
        self.stop_event = threading.Event()

    def _setup_ui(self):
        layout = QVBoxLayout()

        button_stile ="""
    QPushButton {
        background-color: #6495ED;
        color: white;
        padding: 5px 20px;
        font-size: 14px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #4682B4;
    }
"""

        top_layout = QHBoxLayout()
        self.btn_load_audio = QPushButton('Загрузить аудиофайл')
        self.btn_load_audio.setStyleSheet(button_stile)
        self.btn_load_audio.clicked.connect(self.load_audio)
        top_layout.addWidget(self.btn_load_audio)

        self.btn_play_audio = QPushButton('Воспроизвести')
        self.btn_play_audio.setStyleSheet(button_stile)
        self.btn_play_audio.clicked.connect(self.play_audio)
        top_layout.addWidget(self.btn_play_audio)

        self.btn_pause_audio = QPushButton('Пауза')
        self.btn_pause_audio.setStyleSheet(button_stile)
        self.btn_pause_audio.clicked.connect(self.pause_audio)
        top_layout.addWidget(self.btn_pause_audio)

        self.lbl_audio = QLabel('Файл: не выбран')
        self.lbl_audio.setStyleSheet("color: #6495ED;")
        top_layout.addWidget(self.lbl_audio)

        self.btn_load_model = QPushButton('Загрузить модель')
        self.btn_load_model.setStyleSheet(button_stile)
        self.btn_load_model.clicked.connect(self.load_model)
        top_layout.addWidget(self.btn_load_model)

        self.lbl_model = QLabel('Модель: не загружена')
        self.lbl_model.setStyleSheet("color: #6495ED;")
        top_layout.addWidget(self.lbl_model)

        layout.addLayout(top_layout)

        mid_layout = QHBoxLayout()
        self.combo_feature = QComboBox()
        self.combo_feature.addItems(['mfcc', 'mel'])
        self.combo_feature.currentTextChanged.connect(self.on_feature_change)
        mid_layout.addWidget(QLabel('Тип признаков:'))
        mid_layout.addWidget(self.combo_feature)

        self.btn_preprocess = QPushButton('Предобработать аудио')
        self.btn_preprocess.setStyleSheet(button_stile)
        self.btn_preprocess.clicked.connect(self.preprocess_audio)
        mid_layout.addWidget(self.btn_preprocess)

        self.btn_infer = QPushButton('Определить эмоцию')
        self.btn_infer.setStyleSheet(button_stile)
        self.btn_infer.clicked.connect(self.infer)
        mid_layout.addWidget(self.btn_infer)

        self.btn_save_report = QPushButton('Сохранить отчёт')
        self.btn_save_report.setStyleSheet(button_stile)
        self.btn_save_report.clicked.connect(self.save_report)
        mid_layout.addWidget(self.btn_save_report)

        layout.addLayout(mid_layout)

        center_layout = QHBoxLayout()
        self.lbl_plot = QLabel()
        self.lbl_plot.setFixedSize(560, 360)
        self.lbl_plot.setFrameShape(QtWidgets.QFrame.Box)
        center_layout.addWidget(self.lbl_plot)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        center_layout.addWidget(self.text_log)

        layout.addLayout(center_layout)

        train_box = QVBoxLayout()
        train_label = QLabel('Обучение модели на RAVDESS')
        train_label.setStyleSheet('font-weight: bold; color: #6495ED;')
        train_box.addWidget(train_label)

        train_controls = QHBoxLayout()
        self.input_ravdess = QLineEdit()
        self.input_ravdess.setPlaceholderText('Путь к папке ravdess')
        train_controls.addWidget(self.input_ravdess)
        self.btn_browse_ravdess = QPushButton('Обзор')
        self.btn_browse_ravdess.clicked.connect(self.browse_ravdess)
        train_controls.addWidget(self.btn_browse_ravdess)
        train_box.addLayout(train_controls)

        params_layout = QHBoxLayout()
        self.input_epochs = QLineEdit('10')
        self.input_epochs.setFixedWidth(80)
        params_layout.addWidget(QLabel('Эпохи:'))
        params_layout.addWidget(self.input_epochs)

        self.input_batch = QLineEdit('16')
        self.input_batch.setFixedWidth(80)
        params_layout.addWidget(QLabel('Batch:'))
        params_layout.addWidget(self.input_batch)

        self.input_lr = QLineEdit('0.001')
        self.input_lr.setFixedWidth(100)
        params_layout.addWidget(QLabel('LR:'))
        params_layout.addWidget(self.input_lr)
        train_box.addLayout(params_layout)

        train_buttons = QHBoxLayout()
        self.btn_start_train = QPushButton('Запустить обучение')
        self.btn_start_train.setStyleSheet(button_stile)
        self.btn_start_train.clicked.connect(self.run_training_from_gui)
        train_buttons.addWidget(self.btn_start_train)

        self.btn_stop_train = QPushButton('Остановить обучение')
        self.btn_stop_train.setStyleSheet(button_stile)
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)
        train_buttons.addWidget(self.btn_stop_train)
        train_box.addLayout(train_buttons)

        self.train_progress = QProgressBar()
        train_box.addWidget(self.train_progress)

        layout.addLayout(train_box)

        bottom_layout = QHBoxLayout()
        self.progress = QProgressBar()
        bottom_layout.addWidget(self.progress)
        self.lbl_status = QLabel('Готово')
        bottom_layout.addWidget(self.lbl_status)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)
    
    def _cleanup_tmp(self):

        try:
            if getattr(self, "current_processed_audio", None):
                tmp = self.current_processed_audio
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                self.current_processed_audio = None
        except Exception:
            self.current_processed_audio = None

    def log(self, msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        self.text_log.append(f"[{ts}] {msg}")

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Открыть аудиофайл', '', 'Аудио (*.wav *.mp3 *.flac *.ogg)')
        if not path:
            return
        self._cleanup_tmp()

        self.current_audio_path = path
        self.current_processed_audio = None
        self.lbl_audio.setText(f'Файл: {os.path.basename(path)}')
        self.log(f'Загружен аудиофайл: {path}')
        try:
            y, sr = load_audio_file(path, sr=SAMPLE_RATE)
            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=100)
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title('Волновой сигнал (waveform)')
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            pix = QtGui.QPixmap()
            pix.loadFromData(buf.getvalue())
            pix = pix.scaled(self.lbl_plot.size(), Qt.KeepAspectRatio)
            self.lbl_plot.setPixmap(pix)
        except Exception as e:
            self.log(f'Ошибка при загрузке аудио: {e}')

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Открыть модель', '', 'PyTorch модель (*.pth)')
        if not path:
            return
        try:
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    model = SERModel(in_channels=1, n_classes=len(TARGET_EMOTIONS))
                    model.load_state_dict(checkpoint)
                else:
                    raise TypeError
            except TypeError:
                checkpoint = torch.load(path, map_location='cpu')
                model = SERModel(in_channels=1, n_classes=len(TARGET_EMOTIONS))
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

            model.to(DEVICE)
            model.eval()
            self.model = model
            self.model_path = path
            self.lbl_model.setText(f'Модель: {os.path.basename(path)}')
            self.log(f'Модель загружена: {path}')
        except Exception as e:
            self.log(f'Ошибка загрузки модели: {e}')
            QMessageBox.critical(self, 'Ошибка', f'Не удалось загрузить модель:\n{e}')

    def on_feature_change(self, text):
        self.feature = text
        self.log(f'Выбран тип признаков: {text}')

    def preprocess_audio(self):
        if not self.current_audio_path:
            self.log('Сначала загрузите аудиофайл')
            return
        try:
            y, sr = load_audio_file(self.current_audio_path, sr=SAMPLE_RATE)
            self.log(f'Исходный сигнал: {y.shape[0]} сэмплов, sr={sr}')
            y = normalize_audio(y)
            y = trim_silence(y)
            y = ensure_length(y, sr, min_seconds=1.0)
            y = simple_noise_reduction(y, sr)

            self._cleanup_tmp()

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            save_audio_file(tmp_path, y, sr=sr)
            self.current_processed_audio = tmp_path
            self.log(f'Аудио предобработано и сохранено во временный файл: {tmp_path}')

            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=100)
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title('Волновой сигнал (предобработанный)')
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            pix = QtGui.QPixmap()
            pix.loadFromData(buf.getvalue())
            pix = pix.scaled(self.lbl_plot.size(), Qt.KeepAspectRatio)
            self.lbl_plot.setPixmap(pix)
        except Exception as e:
            self.log(f'Ошибка при предобработке: {e}')

    def infer(self):
        if self.model is None:
            self.log('Модель не загружена. Сначала загрузите .pth файл или обучите модель через интерфейс.')
            QMessageBox.information(self, 'Информация', 'Модель не загружена. Загрузите .pth файл или обучите модель.')
            return
        audio_path = self.current_processed_audio or self.current_audio_path
        if not audio_path:
            self.log('Аудиофайл не выбран')
            return
        self.log(f'Запуск инференса на файле: {audio_path}')
        try:
            self.progress.setValue(10)
            pred = infer_on_audio_file(self.model, audio_path, DEVICE, feature=self.feature)
            self.progress.setValue(60)
            self.last_prediction = pred
            
            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=100)
            emotions = list(pred.keys())
            probs = [pred[e] for e in emotions]
            sns.barplot(x=probs, y=emotions, ax=ax)
            ax.set_xlim(0, 1)
            ax.set_title('Вероятности эмоций')
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            pix = QtGui.QPixmap()
            pix.loadFromData(buf.getvalue())
            pix = pix.scaled(self.lbl_plot.size(), Qt.KeepAspectRatio)
            self.lbl_plot.setPixmap(pix)
            self.progress.setValue(100)
            self.log('Инференс завершён. Результаты:')
            for k, v in pred.items():
                self.log(f'  {k}: {v:.4f}')
            top_em = max(pred.items(), key=lambda x: x[1])
            self.log(f"Предполагаемая эмоция: {top_em[0]} (вероятность {top_em[1]:.3f})")
        except Exception as e:
            self.log(f'Ошибка при инференсе: {e}')

    def save_report(self):
        if not self.last_prediction:
            self.log('Нет результата инференса для сохранения')
            QMessageBox.information(self, 'Информация', 'Нет результата инференса для сохранения')
            return
        base, _ = QFileDialog.getSaveFileName(self, 'Сохранить отчёт (выберите имя, создадутся .pdf и .csv)', '', 'PDF файлы (*.pdf)')
        if not base:
            return
        pdf_path = base if base.lower().endswith('.pdf') else base + '.pdf'
        csv_path = os.path.splitext(pdf_path)[0] + '.csv'
        audio_path = self.current_processed_audio or self.current_audio_path
        if not audio_path:
            self.log('Не найден аудиофайл для отчёта')
            QMessageBox.warning(self, 'Предупреждение', 'Не найден аудиофайл для отчёта')
            return
        extra = {'модель': os.path.basename(self.model_path) if self.model_path else 'N/A', 'фича': self.feature}
        generate_report_pdf(pdf_path, audio_path, self.last_prediction, extra_info=extra)
        generate_report_csv(csv_path, audio_path, self.last_prediction)
        self.log(f'Отчёт сохранён: {pdf_path} и {csv_path}')
        QMessageBox.information(self, 'Готово', f'Отчёт сохранён: {pdf_path} и {csv_path}')

    def browse_ravdess(self):
        path = QFileDialog.getExistingDirectory(self, 'Выберите папку RAVDESS')
        if path:
            self.input_ravdess.setText(path)

    def run_training_from_gui(self):
        rav_path = self.input_ravdess.text().strip()
        if not rav_path or not os.path.isdir(rav_path):
            QMessageBox.warning(self, 'Ошибка', 'Укажите корректную папку с RAVDESS')
            return
        try:
            epochs = int(self.input_epochs.text())
            batch = int(self.input_batch.text())
            lr = float(self.input_lr.text())
        except Exception:
            QMessageBox.warning(self, 'Ошибка', 'Неверные параметры обучения')
            return

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.log(f'Запуск обучения: папка={rav_path}, эпохи={epochs}, batch={batch}, lr={lr}')

        self.stop_event.clear()
        self.train_thread = TrainingThread(rav_path, epochs, batch, lr, self.feature, self.stop_event)
        self.train_thread.progress.connect(self.train_progress.setValue)
        self.train_thread.finished.connect(self.on_training_finished)
        self.train_thread.error.connect(self.on_training_error)
        self.train_thread.start()

    def on_training_finished(self, path):
        self.log(f'Обучение завершено. Модель сохранена: {path}')
        QMessageBox.information(self, 'Готово', f'Обучение завершено. Модель сохранена: {path}')
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.train_progress.setValue(100)

    def on_training_error(self, msg):
        self.log(f'Ошибка/остановка обучения: {msg}')
        QMessageBox.warning(self, 'Ошибка', msg)
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
    
    def play_audio(self):
        path = self.current_processed_audio or self.current_audio_path
        if path:
            try:
                try:
                    self.player.stop()
                    self.player.setMedia(QMediaContent())
                    QtCore.QCoreApplication.processEvents()
                except Exception:
                    pass

                self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
                try:
                    self.player.setPosition(0)
                except Exception:
                    pass
                QtCore.QCoreApplication.processEvents()
                self.player.play()
                self.log(f"Воспроизведение: {path}")
            except Exception as e:
                self.log(f"Ошибка воспроизведения: {e}")
        else:
            QMessageBox.warning(self, "Нет файла", "Сначала загрузите аудиофайл.")

    def pause_audio(self):
        self.player.pause()
        self.log("Воспроизведение приостановлено")

    def stop_training(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.log("Остановка обучения запрошена пользователем.")
            self.btn_stop_train.setEnabled(False)
            self.btn_start_train.setEnabled(True)

def main():
    parser = argparse.ArgumentParser(description='Система распознавания эмоций в речи (SER)')
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'train'], help='режим: gui или train')
    parser.add_argument('--ravdess', type=str, default='ravdess', help='путь к датасету RAVDESS')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_out', type=str, default='ser_model.pth')
    args = parser.parse_args()

    if args.mode == 'train':
        print('Режим обучения (CLI)')
        train_from_ravdess(args.ravdess, epochs=args.epochs, batch_size=args.batch_size, model_save_path=args.model_out)
    else:
        app = QApplication(sys.argv)
        window = SERApp()
        window.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()

