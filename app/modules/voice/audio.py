"""Звук на чистом NumPy: чтение WAV, отделение речи (VAD), спектр, мел-фильтры, MFCC, высота тона.
Все функции работают на кадрах по 25 мс с шагом 10 мс при частоте 16 кГц, поэтому массивы признаков имеют одинаковое число кадров."""
import io
import wave

import numpy as np

SR = 16000
FRAME, HOP, NFFT = 400, 160, 512
N_MELS, N_MFCC_MAX = 40, 20
MIN_SPEECH_SEC = 0.6
MAX_SEC = 12.0
_WIN = np.hamming(FRAME)
_FREQS = np.arange(NFFT // 2 + 1) * SR / NFFT


def read_wav(data: bytes) -> np.ndarray:
    """WAV (8, 16 или 32 бита, любое число каналов и частота) -> моно float32 при 16 кГц."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            n_ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        raise ValueError("Не удалось прочитать WAV. Загрузите файл формата WAV или запишите голос кнопкой «Записать».") from None
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Неподдерживаемая разрядность WAV (нужна 8, 16 или 32 бита).")
    if n_ch > 1:
        x = x[: len(x) // n_ch * n_ch].reshape(-1, n_ch).mean(axis=1)
    if sr != SR and len(x) > 1:
        n_new = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n_new), np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(x: np.ndarray) -> bytes:
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())
    return buf.getvalue()


def frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    return x[np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]]


def pre_emphasis(x: np.ndarray) -> np.ndarray:
    """Усиление высоких частот: у голоса они слабее низких, а несут важные особенности тембра."""
    return np.append(x[0], x[1:] - 0.97 * x[:-1])


def _clean(m: np.ndarray, gap: int = 8, min_run: int = 5) -> np.ndarray:
    """Заполняет короткие провалы внутри речи (меньше gap кадров) и убирает короткие всплески (меньше min_run кадров)."""
    m = m.copy()
    for val, limit in ((False, gap), (True, min_run)):
        i = 0
        while i < len(m):
            j = i
            while j < len(m) and m[j] == m[i]:
                j += 1
            if m[i] == val and j - i < limit and i > 0 and j < len(m):
                m[i:j] = not val
            i = j
    return m


def vad(x: np.ndarray, method: str = "adaptive", level: float = 30.0) -> np.ndarray:
    """Маска кадров с речью (True). adaptive - порог между уровнем тишины и громкими кадрами этой записи;
    fixed - всё, что не тише самого громкого кадра более чем на level дБ; zcr - как adaptive, но без шумных кадров (много пересечений нуля)."""
    fr = frames(x.astype(np.float64))
    p = (fr ** 2).mean(axis=1)
    e = 10 * np.log10(np.convolve(p, np.ones(7) / 7, mode="same") + 1e-12)  # сглаживание: в кадре бывает мало импульсов гортани
    if method == "fixed":
        m = e > e.max() - level
    else:
        lo, hi = np.percentile(e, 10), np.percentile(e, 95)
        if hi - lo < 6:  # почти нет динамики: тишина или ровный шум
            return np.zeros(len(e), dtype=bool)
        m = e > lo + max(4.0, 0.3 * (hi - lo))
        if method == "zcr":
            zcr = (np.diff(np.signbit(fr), axis=1) != 0).mean(axis=1)
            m &= zcr < 0.3
    return _clean(m)


def _mel_bank() -> np.ndarray:
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(80.0), mel(7600.0), N_MELS + 2))
    fb = np.zeros((N_MELS, len(_FREQS)))
    for i in range(N_MELS):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((_FREQS - l) / (c - l), (r - _FREQS) / (r - c)))
    return fb


_MEL = _mel_bank()
_DCT = np.sqrt(2.0 / N_MELS) * np.cos(np.pi / N_MELS * (np.arange(N_MELS)[None, :] + 0.5) * np.arange(N_MFCC_MAX + 1)[:, None])


def power_spec(x: np.ndarray) -> np.ndarray:
    """Энергия спектра в каждом кадре, форма (кадры, NFFT/2+1)."""
    return np.abs(np.fft.rfft(frames(pre_emphasis(x.astype(np.float64))) * _WIN, NFFT)) ** 2


def delta(m: np.ndarray, n: int = 2) -> np.ndarray:
    """Скорость изменения коэффициентов во времени (регрессия по 2n+1 кадрам)."""
    pad = np.pad(m, ((n, n), (0, 0)), mode="edge")
    den = 2 * sum(i * i for i in range(1, n + 1))
    return sum(i * (pad[n + i: n + i + len(m)] - pad[n - i: n - i + len(m)]) for i in range(1, n + 1)) / den


def pitch_track(x: np.ndarray, fmin: float = 70.0, fmax: float = 400.0) -> np.ndarray:
    """Высота тона по автокорреляции в каждом кадре (Гц); 0 - безголосый кадр."""
    fr = frames(x.astype(np.float64))
    fr = fr - fr.mean(axis=1, keepdims=True)
    energy = (fr ** 2).mean(axis=1)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * FRAME)) ** 2, 2 * FRAME)[:, :FRAME]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(SR / fmax), int(SR / fmin)
    seg = ac[:, lo:hi]
    k = seg.argmax(axis=1)
    peak = seg[np.arange(len(seg)), k]
    return np.where((peak > 0.35) & (energy > 1e-7), SR / (k + lo), 0.0)


def analyze(x: np.ndarray, vad_method: str = "adaptive", vad_level: float = 30.0, n_mfcc: int = 13) -> dict:
    """Все признаки записи по кадрам. ValueError с понятным текстом, если речи слишком мало."""
    x = (x - x.mean()).astype(np.float32)
    if np.abs(x).max() < 1e-4:
        raise ValueError("Запись пустая: в ней тишина.")
    P = power_spec(x)
    lm = np.log(P @ _MEL.T + 1e-10)
    mf = (lm @ _DCT.T)[:, 1:n_mfcc + 1]  # c0 (громкость) отбрасывается
    mask = vad(x, vad_method, vad_level)
    speech = float(mask.sum() * HOP / SR)
    if speech < MIN_SPEECH_SEC:
        raise ValueError(f"В записи мало речи ({speech:.1f} с, нужно не меньше {MIN_SPEECH_SEC} с). Говорите громче и ближе к микрофону или смените способ выделения речи.")
    tot = P.sum(axis=1) + 1e-12
    cum = np.cumsum(P, axis=1) / tot[:, None]
    fr = frames(x.astype(np.float64))
    return {"x": x, "P": P, "logmel": lm, "mfcc": mf, "dmfcc": delta(mf), "f0": pitch_track(x), "mask": mask, "dur": len(x) / SR, "speech_sec": speech,
            "centroid": (P * _FREQS).sum(axis=1) / tot, "rolloff": _FREQS[(cum < 0.85).sum(axis=1).clip(max=len(_FREQS) - 1)],
            "flatness": np.exp(np.log(P + 1e-12).mean(axis=1)) / (P.mean(axis=1) + 1e-12), "zcr": (np.diff(np.signbit(fr), axis=1) != 0).mean(axis=1)}
