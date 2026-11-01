import hashlib
import itertools
import re
import time
from typing import Any

import cv2
import numpy as np

from app.core.cache import LRUCache
from app.core.imaging import to_data_url
from app.core.metrics import error_curves
from app.core.realdata import matches
from app.modules.base import BiometricModule
from app.modules.uploads import UploadStore
from app.modules.voice import audio, features, render, synthetic
from app.modules.voice.params import VoiceParams

SYNTHETIC = "synthetic"
SYNTHETIC_TITLE = "Синтетические дикторы (12 голосов × 3 записи, одна фраза)"
_ID = re.compile(r"v(\d{2})_r(\d)")
VAD_TITLES = {"adaptive": "Адаптивный порог по энергии", "fixed": "Фиксированный порог (дБ ниже пика, ползунок)", "zcr": "Энергия + пересечения нуля (без шума)"}
VAD_COLORS = {"adaptive": "#e67e22", "fixed": "#2e9e4f", "zcr": "#9b3d9b"}


class VoiceModule(BiometricModule):
    name = "voice"
    title = "Распознавание голоса (диктор)"
    default_threshold = 0.5
    upload_kind = "audio"

    def __init__(self):
        self._uploads = UploadStore()
        self._cache = LRUCache(200)

    # ---------- образцы ----------
    def list_samples(self, source: str | None = None, q: str = "", limit: int = 0) -> list[dict[str, Any]]:
        if source not in (None, "", SYNTHETIC):
            raise KeyError(f"Набор «{source}» не найден. Для своих записей используйте кнопки «Записать» и «Загрузить».")
        items = [{"id": f"v{k:02d}_r{r}", "label": f"Диктор {k:02d}, запись {r} (f0 около {synthetic.speaker(k)['f0']:.0f} Гц)", "person": f"v{k:02d}"}
                 for k in range(1, synthetic.N_SPEAKERS + 1) for r in range(1, synthetic.N_RECORDINGS + 1)]
        items = [x for x in items if matches(q, x["label"], x["id"])]
        if limit:
            items = items[:limit]
        return [{"id": u, "label": f"Записано или загружено: {u}", "person": None} for u in self._uploads.ids()] + items

    def info(self) -> dict[str, Any]:
        return {"sources": [{"id": SYNTHETIC, "title": SYNTHETIC_TITLE, "real": False, "images": synthetic.N_SPEAKERS * synthetic.N_RECORDINGS,
                             "note": "Голоса синтетические: реальные записи людей не используются. Свой голос можно записать с микрофона или загрузить WAV; "
                                     "запись хранится только в памяти сервера и видна только вам."}],
                "default_source": SYNTHETIC, "audio": True, "hint": ""}

    def _raw(self, sample_id: str) -> np.ndarray:
        up = self._uploads.get(sample_id)
        if up is not None:
            return up.copy()
        m = _ID.fullmatch(sample_id)
        if not m:
            raise KeyError(f"Неизвестный образец: {sample_id}")
        return synthetic.recording(int(m[1]), int(m[2])).copy()

    def _audio(self, sample_id: str, p: VoiceParams) -> np.ndarray:
        x = self._raw(sample_id)
        if p.extra_noise > 0:
            seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
            x = np.clip(x + np.random.default_rng(seed).standard_normal(len(x)) * p.extra_noise / 100 * np.sqrt((x ** 2).mean()), -1, 1).astype(np.float32)
        return x

    def get_sample_preview(self, sample_id: str, params: dict) -> bytes:
        x = self._audio(sample_id, VoiceParams(**params))
        logp = 10 * np.log10(audio.power_spec(x) + 1e-10)
        return cv2.imencode(".png", render.heat(logp, len(x) / audio.SR, 140, 320, cv2.COLORMAP_MAGMA))[1].tobytes()

    def get_sample_audio(self, sample_id: str, params: dict) -> bytes:
        return audio.write_wav(self._audio(sample_id, VoiceParams(**params)))

    def upload(self, data: bytes) -> str:
        x = audio.read_wav(data)
        dur = len(x) / audio.SR
        if dur < 0.8:
            raise ValueError("Запись слишком короткая (меньше 0,8 с).")
        if dur > audio.MAX_SEC:
            raise ValueError(f"Запись слишком длинная (больше {audio.MAX_SEC:.0f} с).")
        return self._uploads.add((x / (np.abs(x).max() + 1e-9) * 0.8).astype(np.float32))

    # ---------- обработка ----------
    def _analysis(self, sample_id: str, p: VoiceParams) -> dict:
        key = f"{sample_id}|{p.vad}|{p.vad_level}|{p.n_mfcc}|{p.extra_noise}"
        a = self._cache.get(key)
        if a is None:
            a = audio.analyze(self._audio(sample_id, p), p.vad, p.vad_level, p.n_mfcc)
            self._cache.put(key, a)
        return a

    def _template(self, sample_id: str, p: VoiceParams, method: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        method = method or p.method
        v = features.template(self._analysis(sample_id, p), method)
        return v, features.zvec(v, method, p.vad, p.vad_level, p.n_mfcc)

    @staticmethod
    def _f0(a: dict) -> float | None:
        v = a["f0"][a["mask"] & (a["f0"] > 0)]
        return float(np.median(v)) if len(v) else None

    def process(self, sample_id: str, params: dict) -> dict[str, Any]:
        p = VoiceParams(**params)
        a = self._analysis(sample_id, p)
        v, z = self._template(sample_id, p)
        x, dur, mask = a["x"], a["dur"], a["mask"]
        f0 = self._f0(a)
        logp = 10 * np.log10(a["P"] + 1e-10)
        segs = int(np.sum(np.diff(np.r_[0, mask.astype(int)]) == 1))
        pitch_text = (f"Автокорреляция в каждом кадре находит период колебаний голосовых складок: это основной тон (f0) диктора. У мужчин он обычно 85-180 Гц, "
                      f"у женщин 165-255 Гц. Медиана этой записи: {f0:.0f} Гц.") if f0 else "Высота тона не найдена: в записи нет озвученных кадров."
        steps = [
            ("original", "1. Исходный сигнал", f"Запись {dur:.1f} с, частота дискретизации 16 кГц. Показана форма волны: громкость в каждый момент времени. "
             "Пока это просто колебания воздуха, записанные микрофоном: на таком сигнале голоса сравнивать нельзя.", render.waveform(x)),
            ("vad", "2. Выделение речи (VAD)", f"Способ: {VAD_TITLES[p.vad].lower()}. Зелёным отмечены кадры по 25 мс, где есть речь: {a['speech_sec']:.1f} из {dur:.1f} с "
             f"({segs} фрагм.). Паузы и шум отбрасываются, иначе тишина попадёт в шаблон и «размоет» голос.", render.waveform(x, mask)),
            ("spectrum", "3. Спектрограмма", "Сигнал нарезается на кадры по 25 мс, для каждого считается спектр (какие частоты звучат). По горизонтали время, по вертикали частота "
             "0-8 кГц, яркость - энергия. Горизонтальные полосы - гармоники голоса, тёмные пятна между ними - форманты речевого тракта.",
             render.heat(logp, dur, cmap=cv2.COLORMAP_MAGMA, ylabels=(("8 kHz", 0.0), ("0", 0.93)))),
            ("mel", "4. Мел-спектрограмма", "Частоты пересчитываются в мел-шкалу: слух различает низкие частоты точнее высоких, поэтому 40 фильтров идут чаще внизу и реже вверху. "
             "Получается компактное описание огибающей спектра.", render.heat(a["logmel"], dur, ylabels=(("40", 0.0), ("1", 0.93)))),
            ("mfcc", "5. MFCC", f"Дискретное косинусное преобразование мел-спектра даёт {p.n_mfcc} коэффициентов MFCC в каждом кадре. Они описывают тембр: форму речевого тракта "
             "диктора. Первый коэффициент (громкость) отбрасывается.", render.heat(a["mfcc"], dur, cmap=cv2.COLORMAP_JET, ylabels=((f"c{p.n_mfcc}", 0.0), ("c1", 0.93)))),
            ("pitch", "6. Высота тона (f0)", pitch_text, render.pitch_plot(a["f0"], mask)),
            ("template", "7. Шаблон голоса", f"Метод: {features.METHODS[p.method]}. По всей речи записи считаются средние и разброс признаков, получается вектор из {len(v)} чисел. "
             "Столбики показывают его в z-оценках (отклонение от «среднего диктора» в долях разброса): это и есть шаблон, который сравнивается с другими.",
             render.bars([z], title=f"{p.method}: {len(z)} numbers")),
        ]
        facts = [["Метод", features.METHODS[p.method]], ["Длительность", f"{dur:.1f} с"], ["Речь (VAD)", f"{a['speech_sec']:.1f} с, {a['speech_sec'] / dur * 100:.0f} % записи"],
                 ["Высота тона f0", f"{f0:.0f} Гц" if f0 else "не найдена"], ["Размерность шаблона", len(v)],
                 ["Шум в записи", f"добавлено {p.extra_noise:g} % от громкости" if p.extra_noise else "добавочного шума нет"]]
        return {"sample_id": sample_id, "found": True, "facts": facts,
                "steps": [{"id": i, "title": t, "description": d, "desc": d, "image": to_data_url(im)} for i, t, d, im in steps]}

    # ---------- сравнение ----------
    def compare(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        p = VoiceParams(**params)
        aa, ab = self._analysis(sample_a, p), self._analysis(sample_b, p)
        (va, za), (_, zb) = self._template(sample_a, p), self._template(sample_b, p)
        sim = features.similarity(za, zb)
        d2 = float(np.mean((za - zb) ** 2))

        def spec(a, label):
            return render.mark(render.heat(10 * np.log10(a["P"] + 1e-10), a["dur"], 150, 320, cv2.COLORMAP_MAGMA), label)
        top = np.hstack([spec(aa, "A"), spec(ab, "B")])
        bottom = render.bars([za, zb], h=170, title=f"A (orange) and B (blue): d2 = {d2:.2f}, similarity = {sim:.2f}")
        f0a, f0b = self._f0(aa), self._f0(ab)
        details = [["Метод", features.METHODS[p.method]], ["Сходство", f"{sim:.3f}"], ["Среднее расстояние² между шаблонами", f"{d2:.2f}"],
                   ["Речь A / B", f"{aa['speech_sec']:.1f} с / {ab['speech_sec']:.1f} с"],
                   ["Высота тона A / B", f"{f0a:.0f} Гц / {f0b:.0f} Гц" if f0a and f0b else "не найдена"], ["Размерность шаблона", len(va)]]
        return {"similarity": round(sim, 4), "n_a": 0, "n_b": 0, "matched": 0, "details": details, "overlay": to_data_url(np.vstack([top, bottom])),
                "context": {"modality": "voice", "method": p.method, "extra_noise": p.extra_noise, "f0_a": round(f0a, 1) if f0a else None,
                            "f0_b": round(f0b, 1) if f0b else None, "speech_a": round(aa["speech_sec"], 2), "speech_b": round(ab["speech_sec"], 2), "hd": None}}

    # ---------- метрики ----------
    def _scores(self, p: VoiceParams, n_subjects: int, method: str) -> tuple[list, list]:
        n = max(3, min(n_subjects, synthetic.N_SPEAKERS))
        spk = list(range(1, n + 1))
        rec = range(1, synthetic.N_RECORDINGS + 1)
        z = {(k, r): self._template(f"v{k:02d}_r{r}", p, method)[1] for k in spk for r in rec}
        genuine = [features.similarity(z[(k, a)], z[(k, b)]) for k in spk for a, b in itertools.combinations(rec, 2)]
        impostor = [features.similarity(z[(k, r)], z[(j, r)]) for k, j in itertools.combinations(spk, 2) for r in (1, 2)]
        return genuine, impostor

    def evaluate(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        p = VoiceParams(**params)
        genuine, impostor = self._scores(p, n_subjects, p.method)
        out = error_curves(genuine, impostor)
        out.update(source=p.source, source_title=SYNTHETIC_TITLE, real=False, skipped_images=0)
        return out

    # ---------- сравнение методов ----------
    def compare_detectors(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Три способа отделить речь от тишины на одной записи: доля речи, число фрагментов, время."""
        p = VoiceParams(**params)
        x = self._audio(sample_id, p)
        rows, strips = [], []
        for k, (name, title) in enumerate(VAD_TITLES.items()):
            t = time.perf_counter()
            m = audio.vad(x, name, p.vad_level)
            ms = (time.perf_counter() - t) * 1000
            strips.append(render.mark(render.waveform(x, m, 90, 640), f"{k + 1}. {name}"))
            rows.append({"n": k + 1, "name": name, "title": title, "available": True, "found": bool(m.any()), "time_ms": round(ms, 2),
                         "speech": f"{m.sum() * audio.HOP / audio.SR:.1f} с ({m.mean() * 100:.0f} %)", "segments": int(np.sum(np.diff(np.r_[0, m.astype(int)]) == 1)),
                         "color": VAD_COLORS[name]})
        columns = [["n", "№"], ["title", "Способ выделения речи"], ["speech", "Найдено речи"], ["segments", "Фрагментов"], ["time_ms", "Время, мс"]]
        return {"sample_id": sample_id, "rows": rows, "columns": columns, "overlay": to_data_url(np.vstack(strips))}

    def compare_embedders(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        """Одна и та же пара записей, разные методы шаблона: сходство, решение при рекомендуемом пороге, время."""
        p = VoiceParams(**params)
        aa, ab = self._analysis(sample_a, p), self._analysis(sample_b, p)
        rows = []
        for name, title in features.METHODS.items():
            t = time.perf_counter()
            try:
                za, zb = self._template(sample_a, p, name)[1], self._template(sample_b, p, name)[1]
            except ValueError as e:
                rows.append({"name": name, "title": title, "available": False, "note": str(e)})
                continue
            sim, thr = features.similarity(za, zb), features.DEFAULT_THRESHOLDS[name]
            rows.append({"name": name, "title": title, "available": True, "dim": len(za), "similarity": round(sim, 4), "threshold": thr,
                         "decision": "MATCH" if sim >= thr else "NO MATCH", "time_ms": round((time.perf_counter() - t) * 1000, 1)})
        columns = [["title", "Метод шаблона"], ["dim", "Длина шаблона"], ["similarity", "Сходство"], ["threshold", "Рекомендуемый порог"],
                   ["decision", "Решение при нём"], ["time_ms", "Время, мс"]]

        def up(a):
            return to_data_url(render.heat(10 * np.log10(a["P"] + 1e-10), a["dur"], 120, 384, cv2.COLORMAP_MAGMA))
        return {"rows": rows, "columns": columns, "detector": VAD_TITLES[p.vad], "aligned_a": up(aa), "aligned_b": up(ab)}

    def evaluate_embedders(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        p = VoiceParams(**params)
        rows = []
        for name, title in features.METHODS.items():
            t = time.perf_counter()
            genuine, impostor = self._scores(p, n_subjects, name)
            ev = error_curves(genuine, impostor)
            g, im = np.array(genuine), np.array(impostor)
            rows.append({"name": name, "title": title, "available": True, "eer": ev["eer"], "eer_threshold": ev["eer_threshold"], "mean_genuine": round(float(g.mean()), 3),
                         "mean_impostor": round(float(im.mean()), 3), "margin": round(float(g.min() - im.max()), 3), "n_genuine": ev["n_genuine"], "n_impostor": ev["n_impostor"],
                         "time_s": round(time.perf_counter() - t, 1)})
        return {"rows": rows, "detector": VAD_TITLES[p.vad]}
