import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict
from audio import load_audio_file

SAMPLE_RATE = 16000 
N_MELS = 64 

def generate_report_pdf(output_path: str, audio_path: str, prediction: Dict[str, float], extra_info: Dict = None):
    y, sr = load_audio_file(audio_path, sr=SAMPLE_RATE)
    fig = plt.figure(figsize=(8.27, 11.69)) 
    gs = fig.add_gridspec(3, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1)
    ax1.set_title('Мел-спектрограмма')
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')

    ax2 = fig.add_subplot(gs[1, 0])
    emotions = list(prediction.keys())
    probs = [prediction[e] for e in emotions]
    sns.barplot(x=probs, y=emotions, ax=ax2)
    ax2.set_xlim(0, 1)
    ax2.set_title('Вероятности эмоций')

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')

    info_text = f"Аудиофайл: {os.path.basename(audio_path)}\n"

    if extra_info is not None:
        for k, v in extra_info.items():
            info_text += f"{k}: {v}\n"

    info_text += '\nПредсказание:\n'

    for k, v in prediction.items():
        info_text += f"  {k}: {v:.3f}\n"

    ax3.text(0, 1, info_text, va='top', fontsize=10)

    plt.tight_layout()
    pp = PdfPages(output_path)
    pp.savefig(fig)
    pp.close()
    plt.close(fig)

def generate_report_csv(output_path: str, audio_path: str, prediction: Dict[str, float]):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_file'] + list(prediction.keys()))
        writer.writerow([os.path.basename(audio_path)] + [prediction[k] for k in prediction.keys()])

