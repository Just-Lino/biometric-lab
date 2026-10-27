"""Работа со звуком на чистом NumPy: WAV, VAD, MFCC, высота тона, голосовые признаки,
а также искажения канала (шум, «телефон», реверберация, имитация повторного воспроизведения)."""
from __future__ import annotations

import io
import wave

import numpy as np

SR = 16000
FRAME = int(0.025 * SR)   # 25 мс
HOP = int(0.010 * SR)     # 10 мс
NFFT = 512
N_MELS = 26
N_MFCC = 13               # используем c1..c13 (c0 — громкость — отбрасываем)
FEATURE_DIM = 2 * N_MFCC + 2
MIN_SPEECH_SEC = 0.6


class AudioError(ValueError):
    """Понятная пользователю ошибка обработки звука."""


# ───────────────────────── WAV ─────────────────────────
def read_wav(data: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(data)) as w:
            n_ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise AudioError("Не удалось прочитать WAV. Запись должна идти с кнопки на странице.") from e
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioError("Неподдерживаемая разрядность WAV.")
    if n_ch > 1:
        x = x[: len(x) // n_ch * n_ch].reshape(-1, n_ch).mean(axis=1)
    if sr != SR and len(x) > 1:
        n_new = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n_new), np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(x: np.ndarray, sr: int = SR) -> bytes:
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())
    return buf.getvalue()


# ───────────────────────── кадры, VAD ─────────────────────────
def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def speech_mask(x: np.ndarray) -> np.ndarray:
    """Энергетический VAD: True для кадров с речью."""
    fr = _frames(x)
    e = np.log10((fr ** 2).mean(axis=1) + 1e-10)
    lo, hi = np.percentile(e, 10), np.percentile(e, 95)
    if hi - lo < 0.7:                      # почти нет динамики: тишина или ровный шум
        return np.zeros(len(e), dtype=bool)
    return e > lo + 0.35 * (hi - lo)


# ───────────────────────── MFCC ─────────────────────────
def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_bank(sr=SR, nfft=NFFT, n_mels=N_MELS, fmin=80.0, fmax=7600.0) -> np.ndarray:
    pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_MEL = _mel_bank()
_N = np.arange(N_MELS)
_DCT = np.sqrt(2.0 / N_MELS) * np.cos(np.pi / N_MELS * (_N[None, :] + 0.5) * np.arange(N_MFCC + 1)[:, None])
_DCT[0] *= 1 / np.sqrt(2)
_WIN = np.hamming(FRAME)


def log_mel(x: np.ndarray) -> np.ndarray:
    """Лог-мел-спектрограмма, форма (кадры, N_MELS)."""
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]).astype(np.float64)   # pre-emphasis
    spec = np.abs(np.fft.rfft(_frames(x) * _WIN, NFFT)) ** 2
    return np.log(spec @ _MEL.T + 1e-10)


def mfcc(x: np.ndarray) -> np.ndarray:
    """MFCC c1..c13, форма (кадры, 13)."""
    return (log_mel(x) @ _DCT.T)[:, 1:]


# ───────────────────────── высота тона ─────────────────────────
def pitch_track(x: np.ndarray, fmin=70.0, fmax=400.0) -> np.ndarray:
    """Оценка F0 по автокорреляции; 0 — безголосые кадры."""
    fr = _frames(x.astype(np.float64))
    lag_lo, lag_hi = int(SR / fmax), int(SR / fmin)
    out = np.zeros(len(fr))
    for i, f in enumerate(fr):
        f = f - f.mean()
        if (f ** 2).mean() < 1e-7:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        ac = ac / (ac[0] + 1e-12)
        seg = ac[lag_lo:lag_hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] > 0.35:
            out[i] = SR / (k + lag_lo)
    return out


# ───────────────────────── признаки голоса ─────────────────────────
def analyze(x: np.ndarray) -> dict:
    """Полный разбор записи: VAD → MFCC → F0 → вектор признаков (28 чисел)."""
    if len(x) < SR * 0.5:
        raise AudioError("Запись слишком короткая (меньше 0,5 с).")
    peak = float(np.max(np.abs(x)))
    if peak < 0.01:
        raise AudioError("Запись почти беззвучна. Проверьте микрофон.")
    x = x / peak * 0.9                                  # нормализация громкости
    mask = speech_mask(x)
    speech_sec = float(mask.sum() * HOP / SR)
    if speech_sec < MIN_SPEECH_SEC:
        raise AudioError(f"Речи мало: {speech_sec:.2f} с, нужно хотя бы {MIN_SPEECH_SEC:.1f} с. Говорите чуть дольше и громче.")
    lm = log_mel(x)
    mf = (lm @ _DCT.T)[:, 1:]
    f0 = pitch_track(x)
    n = min(len(mask), len(mf), len(f0))
    mask, mf, f0, lm = mask[:n], mf[:n], f0[:n], lm[:n]
    m = mf[mask]
    voiced = f0[mask & (f0 > 0)]
    if len(voiced) >= 5:
        lf = np.log(voiced)
        f0_mean, f0_std = float(lf.mean()), float(lf.std())
    else:
        f0_mean, f0_std = float("nan"), 0.0
    vec = np.concatenate([m.mean(axis=0), m.std(axis=0), [f0_mean if f0_mean == f0_mean else 5.0, f0_std]])
    clip = float((np.abs(x) > 0.98).mean())
    return {
        "vector": vec.astype(float),
        "speech_sec": speech_sec,
        "total_sec": float(len(x) / SR),
        "f0_hz": float(np.exp(f0_mean)) if f0_mean == f0_mean else None,
        "voiced_ratio": float(len(voiced) / max(mask.sum(), 1)),
        "clipping": clip,
        "wave": x,
        "log_mel": lm,
        "mfcc": mf,
        "mask": mask,
    }


def _downsample_matrix(m: np.ndarray, max_t: int = 120) -> np.ndarray:
    if len(m) <= max_t:
        return m
    idx = np.linspace(0, len(m) - 1, max_t).astype(int)
    return m[idx]


def viz_payload(a: dict) -> dict:
    """Компактные данные для рисования на странице: волна, спектрограмма, MFCC."""
    w = a["wave"]
    nb = 400
    seg = np.array_split(w, nb)
    env = [float(np.max(np.abs(s))) if len(s) else 0.0 for s in seg]
    lm = _downsample_matrix(a["log_mel"])
    mf = _downsample_matrix(a["mfcc"])
    return {
        "waveform": [round(v, 3) for v in env],
        "mel": np.round(lm.T, 2).tolist(),        # (N_MELS, T)
        "mfcc": np.round(mf.T, 2).tolist(),       # (13, T)
        "speech_sec": round(a["speech_sec"], 2),
        "total_sec": round(a["total_sec"], 2),
        "f0_hz": None if a["f0_hz"] is None else round(a["f0_hz"], 1),
        "voiced_ratio": round(a["voiced_ratio"], 2),
        "clipping": round(a["clipping"], 4),
    }


# ───────────────────────── искажения канала ─────────────────────────
def _fft_filter(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def _reverb(x: np.ndarray, rt60: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * rt60)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir /= np.abs(ir).sum() ** 0.5
    L = len(x) + n
    y = np.fft.irfft(np.fft.rfft(x, L) * np.fft.rfft(ir, L), L)[: len(x)]
    return y / (np.abs(y).max() + 1e-9) * np.abs(x).max()


def add_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = (x ** 2).mean() + 1e-12
    n = rng.standard_normal(len(x)) * np.sqrt(p / 10 ** (snr_db / 10))
    return x + n


AUDIO_ATTACKS = {
    "noise": ("Шум в помещении", "SNR, дБ", [30, 20, 10, 5, 0]),
    "telephone": ("Телефонный канал 300–3400 Гц", "степень", [0.0, 1.0]),
    "reverb": ("Эхо (реверберация)", "RT60, с", [0.1, 0.3, 0.6, 1.0]),
    "replay": ("Повторное воспроизведение с динамика (имитация)", "жёсткость", [0.3, 0.6, 1.0]),
}


def apply_attack(x: np.ndarray, kind: str, value: float, seed: int = 0) -> np.ndarray:
    if kind == "noise":
        return add_noise(x, value, seed)
    if kind == "telephone":
        return _fft_filter(x, 300, 3400) if value > 0 else x
    if kind == "reverb":
        return _reverb(x, value, seed)
    if kind == "replay":
        # динамик телефона: обрезка полосы + эхо комнаты + шум; value — сила эффекта 0..1
        y = _fft_filter(x, 200 + 150 * value, 6500 - 1500 * value)
        y = _reverb(y, 0.15 + 0.35 * value, seed)
        return add_noise(y, 28 - 12 * value, seed)
    raise AudioError("Неизвестный тип искажения.")
