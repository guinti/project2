import os
import numpy as np
from typing import Tuple

import librosa
from pydub import AudioSegment
import soundfile as sf

SAMPLE_RATE = 16000
N_MFCC = 40 
N_MELS = 64  
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg'] 

def is_audio_file(filename: str) -> bool: 
    _, ext = os.path.splitext(filename)
    return ext.lower() in AUDIO_EXTENSIONS

def load_audio_file(path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:  
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    name, ext = os.path.splitext(path)
    ext = ext.lower()
    
    if ext in ['.mp3', '.ogg', '.flac']:
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        max_int = float(1 << (8 * audio.sample_width - 1))
        samples = samples / max_int
        return samples, sr
    else:
        
        y, file_sr = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), file_sr

def save_audio_file(path: str, y: np.ndarray, sr: int = SAMPLE_RATE):   
    import soundfile as sf
    sf.write(path, y, sr)

def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:   
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

def normalize_audio(y: np.ndarray) -> np.ndarray:   
    maxv = np.max(np.abs(y))
    if maxv > 0:
        return y / maxv
    return y

def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:   
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt

def simple_noise_reduction(y: np.ndarray, sr: int) -> np.ndarray:
    stft = librosa.stft(y)
    mag, phase = np.abs(stft), np.angle(stft)
    
    noise_floor = np.median(mag, axis=1, keepdims=True)
    
    mask = mag >= (noise_floor * 1.5)
    mag_denoised = mag * mask
    stft_denoised = mag_denoised * np.exp(1j * phase)
    y_d = librosa.istft(stft_denoised)
    
    if y_d.size == 0:
        return y
    return y_d

def ensure_length(y: np.ndarray, sr: int, min_seconds: float = 1.0) -> np.ndarray:    
    min_len = int(min_seconds * sr)
    if y.shape[0] >= min_len:
        return y
    pad = np.zeros(min_len - y.shape[0], dtype=y.dtype)
    return np.concatenate([y, pad])


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:  
    y = librosa.effects.preemphasis(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    return features.astype(np.float32)


def extract_log_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray: 
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2)
    logS = librosa.power_to_db(S, ref=np.max)
    return logS.astype(np.float32)

