import base64
import hashlib
import itertools
import time
from typing import Any

import cv2
import numpy as np

from app.core.cache import LRUCache
from app.core.imaging import to_data_url, to_gray
from app.core.realdata import matches
from app.modules.base import BiometricModule
from app.modules.fingerprint import real, synthetic
from app.modules.fingerprint import matchers
from app.modules.fingerprint.minutiae import draw_minutiae
from app.modules.fingerprint.params import FingerprintParams
from app.modules.fingerprint.pipeline import run_pipeline
from app.modules.uploads import UploadStore

N_FINGERS = 12
N_IMPRESSIONS = 3
SYNTHETIC = "synthetic"
SYNTHETIC_TITLE = "Синтетика лаборатории (12 пальцев × 3 оттиска)"


class FingerprintModule(BiometricModule):
    name = "fingerprint"
    title = "Распознавание отпечатков пальцев"

    def __init__(self):
        self._uploads = UploadStore()
        self._cache = LRUCache(400)

    # ---------- образцы ----------
    def list_samples(self, source: str | None = None, q: str = "", limit: int = 0) -> list[dict[str, Any]]:
        source = source or SYNTHETIC
        if source == SYNTHETIC:
            items = [{"id": f"f{f:02d}_i{i}", "label": f"Палец {f:02d}, оттиск {i}", "person": f"f{f:02d}", "finger": f, "impression": i}
                     for f in range(1, N_FINGERS + 1) for i in range(1, N_IMPRESSIONS + 1)]
        else:
            items = real.list_samples(source)
        items = [x for x in items if matches(q, x["label"], x["id"])]
        if limit:
            items = items[:limit]
        mine = [{"id": uid, "label": f"Загружено: {uid}", "person": None, "finger": None, "impression": None}
                for uid in self._uploads.ids()]
        return mine + items

    def info(self) -> dict[str, Any]:
        sources = [{"id": SYNTHETIC, "title": SYNTHETIC_TITLE, "real": False, "images": N_FINGERS * N_IMPRESSIONS,
                    "note": "Датасет синтетический: реальные биометрические данные людей не используются."}] + real.sources_info()
        return {"sources": sources, "default_source": SYNTHETIC, "hint": ""}

    def _raw(self, sample_id: str) -> np.ndarray:
        img = self._uploads.get(sample_id)
        if img is not None:
            return img.copy()
        if real.is_real_id(sample_id):
            return real.load(sample_id)
        try:
            f, i = sample_id.split("_")
            return synthetic.impression(int(f[1:]), int(i[1:])).copy()
        except (ValueError, KeyError):
            raise KeyError(f"Неизвестный образец: {sample_id}")

    def _gray(self, sample_id: str, p: FingerprintParams) -> np.ndarray:
        img = self._raw(sample_id)
        if p.extra_noise > 0:
            seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
            noise = np.random.RandomState(seed).randn(*img.shape) * p.extra_noise
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def get_sample_preview(self, sample_id: str, params: dict) -> bytes:
        ok, buf = cv2.imencode(".png", self._gray(sample_id, FingerprintParams(**params)))
        return buf.tobytes()

    def upload(self, data: bytes) -> str:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        img = to_gray(img[:, :, :3] if img.ndim == 3 else img)
        scale = 256 / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = np.full((256, 256), 240, np.uint8)
        y0, x0 = (256 - img.shape[0]) // 2, (256 - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        return self._uploads.add(canvas)

    # ---------- обработка ----------
    def _run(self, sample_id: str, p: FingerprintParams, with_images=True):
        key = f"{sample_id}|{p.model_dump_json(exclude={'dist_tol', 'angle_tol_deg', 'max_rotation_deg', 'matcher', 'source'})}|{with_images}"
        res = self._cache.get(key)
        if res is None:
            res = run_pipeline(self._gray(sample_id, p), p, with_images)
            self._cache.put(key, res)
        return res

    def process(self, sample_id: str, params: dict) -> dict[str, Any]:
        p = FingerprintParams(**params)
        res = self._run(sample_id, p)
        steps = [dict(s, image=to_data_url(s["image"])) for s in res.steps]
        n_e = sum(m.type == "ending" for m in res.minutiae)
        return {"sample_id": sample_id, "steps": steps,
                "minutiae": [m.to_dict() for m in res.minutiae],
                "n_minutiae": len(res.minutiae), "n_ending": n_e,
                "n_bifurcation": len(res.minutiae) - n_e}

    def compare(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        p = FingerprintParams(**params)
        ra, rb = self._run(sample_a, p), self._run(sample_b, p)
        m = matchers.compare(p.matcher, ra, rb, p)
        details = [["Метод сопоставления", matchers.MATCHERS[p.matcher]]] + m.details
        return {"similarity": round(m.score, 4), "n_a": m.n_a, "n_b": m.n_b, "matched": m.matched, "rotation_deg": round(m.rotation_deg, 1),
                "shift": [round(v, 1) for v in m.shift], "overlay": to_data_url(m.overlay), "details": details,
                "context": {"modality": "fingerprint", "extra_noise": p.extra_noise, "matcher": p.matcher}}

    # ---------- метрики ----------
    def _eval_layout(self, p: FingerprintParams, n_subjects: int):
        """Какие образцы участвуют в оценке: (пальцы, число оттисков, идентификаторы образцов, описание набора)."""
        if p.source == SYNTHETIC:
            n = max(3, min(n_subjects, N_FINGERS))
            ids = {(f, i): f"f{f:02d}_i{i}" for f in range(1, n + 1) for i in range(1, N_IMPRESSIONS + 1)}
            return list(range(1, n + 1)), N_IMPRESSIONS, ids, f"синтетический набор: {n} пальцев × {N_IMPRESSIONS} оттиска"
        if p.source == real.SOCO:
            raise ValueError("В SOCOFing по одному снимку на палец: «свои» пары были бы оригиналом и его искажённой копией, а не независимыми "
                             "оттисками, поэтому EER не считается. Используйте «Стресс-тест искажений» или набор FVC.")
        if not real.is_real_source(p.source):
            raise ValueError(f"Неизвестный набор данных «{p.source}».")
        fingers = real.fvc_fingers(p.source)
        keys = sorted(f for f, v in fingers.items() if len(v) >= 2)[:max(3, min(n_subjects, 10))]
        if len(keys) < 3:
            raise ValueError("В наборе меньше трёх пальцев с двумя и более оттисками.")
        n_imp = min(len(fingers[k]) for k in keys)
        ids = {(k, i + 1): fingers[k][i] for k in keys for i in range(n_imp)}
        return keys, n_imp, ids, f"{real.title(p.source)}: {len(keys)} пальцев × {n_imp} оттисков"

    def evaluate(self, params: dict, n_subjects: int = 10) -> dict[str, Any]:
        p = FingerprintParams(**params)
        fingers, n_imp, ids, desc = self._eval_layout(p, n_subjects)
        imps = range(1, n_imp + 1)
        note = None
        if p.matcher == "cnn":
            # сеть видела часть снимков при обучении, поэтому считаем только пары, где есть снимок, оставленный для проверки
            from app.training import registry
            mdl = registry.require("fingerprint", "отпечатков пальцев", p.source)
            h = mdl.holdout
            emb = {k: registry.embed("fingerprint", self._gray(v, p)) for k, v in ids.items()}
            others = [i for i in imps if i != h]
            sc = lambda a, b: max(0.0, float(emb[a] @ emb[b]))
            genuine = [sc((f, h), (f, i)) for f in fingers for i in others]
            impostor = [sc((f, h), (g, i)) for f in fingers for g in fingers if g != f for i in others]
            note = f"Оценка по парам, где участвует снимок {h}: сеть не видела его при обучении."
        else:
            runs = {k: self._run(v, p, with_images=False) for k, v in ids.items()}

            def score(k1, k2):
                return matchers.compare(p.matcher, runs[k1], runs[k2], p).score

            genuine = [score((f, a), (f, b)) for f in fingers for a, b in itertools.combinations(imps, 2)]
            impostor = [score((f, i), (g, i)) for f, g in itertools.combinations(fingers, 2) for i in imps]
        th = np.round(np.arange(0, 1.0001, 0.01), 2)
        g, im = np.array(genuine), np.array(impostor)
        frr = [float((g < t).mean()) for t in th]
        far = [float((im >= t).mean()) for t in th]
        k = int(np.argmin(np.abs(np.array(far) - np.array(frr))))
        return {"thresholds": th.tolist(), "far": far, "frr": frr,
                "eer": round((far[k] + frr[k]) / 2, 4), "eer_threshold": float(th[k]),
                "genuine_scores": [round(float(s), 4) for s in genuine],
                "impostor_scores": [round(float(s), 4) for s in impostor],
                "n_genuine": len(genuine), "n_impostor": len(impostor), "note": note,
                "source": p.source, "source_title": desc, "real": p.source != SYNTHETIC}

    # ---------- стресс-тест искажений (SOCOFing) ----------
    def stress(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        """Оригинал против его искажённых копий (стирание, центральный поворот, Z-разрез; три уровня).
        Показывает, как падает сходство с ростом искажения. Это не EER: «свои» пары здесь - копии одного снимка."""
        p = FingerprintParams(**{**params, "source": real.SOCO})
        samples = real.soco_stress_samples(max(3, min(n_subjects, 40)))
        if len(samples) < 3:
            raise ValueError("В SOCOFing не нашлось трёх человек с нужным пальцем и искажёнными копиями.")
        cells: dict[tuple[str, str], list[float]] = {}
        for s in samples:
            ra = self._run(s["real"], p, with_images=False)
            for key, sid in s["altered"].items():
                cells.setdefault(key, []).append(matchers.compare(p.matcher, ra, self._run(sid, p, with_images=False), p).score)
        imp = []  # чужие: оригинал против оригиналов других людей
        runs = [self._run(s["real"], p, with_images=False) for s in samples]
        for i, ra in enumerate(runs):
            for j in range(len(runs)):
                if i != j and (i + j) % 3 == 0:
                    imp.append(matchers.compare(p.matcher, ra, runs[j], p).score)
        # порог для «узнано»: допускаем около 5 % ложных принятий чужих (95-й процентиль «чужих» оценок), чтобы число не зависело от шкалы метода
        thr = round(float(np.percentile(imp, 95)), 3) if imp else matchers.recommended_threshold(p.matcher)
        rows = []
        for level in real.LEVELS:
            by_kind, allv = {}, []
            for kind in real.KINDS:
                v = cells.get((level, kind), [])
                by_kind[kind] = round(float(np.mean(v)), 3) if v else None
                allv += v
            rows.append({"level": level, "mean": round(float(np.mean(allv)), 3) if allv else None, "by_kind": by_kind,
                         "match_rate": round(float(np.mean([x >= thr for x in allv])), 3) if allv else None, "n": len(allv)})
        return {"rows": rows, "kinds": real.KINDS, "threshold": thr, "n_people": len(samples), "matcher": p.matcher,
                "impostor_mean": round(float(np.mean(imp)), 3) if imp else None, "impostor_max": round(float(np.max(imp)), 3) if imp else None,
                "note": "Оригинал сравнивается со своей искажённой копией (а не с независимым оттиском), поэтому «своё» сходство здесь завышено "
                        "относительно реальной жизни. Смотрите, как оно падает с уровнем искажения и как приближается к «чужим»."}

    # ---------- 3D ----------
    def relief(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Карта высот: гребни (усиленное Gabor-изображение) поднимаются над фоном."""
        p = FingerprintParams(**params)
        res = self._run(sample_id, p)
        imgs = {s["id"]: s["image"] for s in res.steps}
        n = 160
        enh = imgs["enhanced"].astype(np.float32)
        h = cv2.GaussianBlur((255 - enh) / 255.0, (0, 0), 1.0)
        soft = cv2.GaussianBlur(res.mask.astype(np.float32), (0, 0), 4)
        h = np.clip(h * soft, 0, 1)
        h_small = cv2.resize(h, (n, n), interpolation=cv2.INTER_AREA)
        m_small = cv2.resize(res.mask.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
        heights = np.clip(h_small * 255, 0, 255).astype(np.uint8)
        mask = np.clip(m_small * 255, 0, 255).astype(np.uint8)

        ys, xs = np.where(res.mask > 0)
        size = res.gray.shape[0]
        if len(xs):
            shape = {"cx": float((xs.min() + xs.max()) / 2 / size), "cy": float((ys.min() + ys.max()) / 2 / size),
                     "rx": float((xs.max() - xs.min()) / 2 / size) + 0.02, "ry": float((ys.max() - ys.min()) / 2 / size) + 0.02}
        else:
            shape = {"cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.45}

        pts = []
        for m in res.minutiae:
            gx, gy = min(n - 1, int(m.x / size * n)), min(n - 1, int(m.y / size * n))
            pts.append({"x": m.x, "y": m.y, "type": m.type, "h": float(heights[gy, gx]) / 255})
        return {
            "sample_id": sample_id, "n": n, "image_size": size,
            "heights": base64.b64encode(heights.tobytes()).decode(),
            "mask": base64.b64encode(mask.tobytes()).decode(),
            "shape": shape, "minutiae": pts,
            "textures": {k: to_data_url(v) for k, v in imgs.items() if k in ("original", "binary", "skeleton", "enhanced")},
        }

    # ---------- сравнение методов ----------
    def compare_detectors(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Три способа бинаризации на одном образце: скелет, найденные минуции, время."""
        from app.modules.fingerprint.pipeline import run_pipeline
        p = FingerprintParams(**params)
        titles = {"adaptive": "Адаптивный порог (по локальной окрестности)", "otsu": "Метод Оцу (один порог по гистограмме)",
                  "fixed": "Фиксированный порог (ползунок)"}
        rows, panels = [], []
        for k, (name, title) in enumerate(titles.items()):
            pp = p.model_copy(update={"bin_method": name})
            t = time.perf_counter()
            res = run_pipeline(self._gray(sample_id, pp), pp, True)
            ms = (time.perf_counter() - t) * 1000
            n_e = sum(m.type == "ending" for m in res.minutiae)
            q = float(np.mean([m.quality for m in res.minutiae])) if res.minutiae else 0.0
            img = next(s["image"] for s in res.steps if s["id"] == "minutiae").copy()
            cv2.putText(img, f"{k + 1}. {name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{k + 1}. {name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 255), 1, cv2.LINE_AA)
            panels.append(img)
            rows.append({"n": k + 1, "name": name, "title": title, "available": True, "found": True, "time_ms": round(ms, 1),
                         "minutiae": f"{len(res.minutiae)} (окончаний {n_e}, разветвлений {len(res.minutiae) - n_e})", "quality": round(q, 2),
                         "color": ["#e67e22", "#2980b9", "#8e44ad"][k]})
        columns = [["n", "№"], ["title", "Бинаризация"], ["minutiae", "Найдено минуций"], ["quality", "Среднее качество точек"], ["time_ms", "Время, мс"]]
        return {"sample_id": sample_id, "rows": rows, "columns": columns, "overlay": to_data_url(np.hstack(panels))}

    def compare_embedders(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        """Одна и та же пара отпечатков, разные методы сопоставления: сходство, время и рекомендуемый порог."""
        p = FingerprintParams(**params)
        ra, rb = self._run(sample_a, p), self._run(sample_b, p)
        rows = []
        for name, title in matchers.MATCHERS.items():
            t = time.perf_counter()
            try:
                m = matchers.compare(name, ra, rb, p)
            except ValueError as e:  # сеть ещё не обучена
                rows.append({"name": name, "title": title, "available": False, "note": str(e)})
                continue
            thr = matchers.recommended_threshold(name, p.source)
            rows.append({"name": name, "title": title, "available": True, "similarity": round(m.score, 4), "threshold": thr,
                         "decision": "MATCH" if m.score >= thr else "NO MATCH", "rotation": f"{m.rotation_deg:+.0f}°",
                         "time_ms": round((time.perf_counter() - t) * 1000, 1)})
        columns = [["title", "Метод сопоставления"], ["similarity", "Сходство"], ["threshold", "Рекомендуемый порог"], ["decision", "Решение при нём"],
                   ["rotation", "Найденный поворот"], ["time_ms", "Время, мс"]]
        return {"rows": rows, "columns": columns, "detector": "минуции извлечены текущими настройками конвейера",
                "aligned_a": to_data_url(draw_minutiae(ra.gray, ra.minutiae)), "aligned_b": to_data_url(draw_minutiae(rb.gray, rb.minutiae))}

    def evaluate_embedders(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        rows = []
        for name, title in matchers.MATCHERS.items():
            t = time.perf_counter()
            try:
                ev = self.evaluate({**params, "matcher": name}, n_subjects)
            except ValueError as e:  # сеть ещё не обучена
                if name != "cnn":
                    raise
                rows.append({"name": name, "title": title, "available": False, "note": str(e)})
                continue
            g, im = np.array(ev["genuine_scores"]), np.array(ev["impostor_scores"])
            rows.append({"name": name, "title": title + (" (оценка по снимкам, которых сеть не видела)" if name == "cnn" else ""), "available": True, "eer": ev["eer"], "eer_threshold": ev["eer_threshold"],
                         "mean_genuine": round(float(g.mean()), 3), "mean_impostor": round(float(im.mean()), 3),
                         "margin": round(float(g.min() - im.max()), 3), "n_genuine": ev["n_genuine"], "n_impostor": ev["n_impostor"],
                         "time_s": round(time.perf_counter() - t, 1)})
        return {"rows": rows, "detector": SYNTHETIC_TITLE if FingerprintParams(**params).source == SYNTHETIC else real.title(FingerprintParams(**params).source)}
