import hashlib
import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.cache import LRUCache
from app.core.imaging import to_data_url
from app.core.realdata import matches
from app.modules.base import BiometricModule
from app.modules.face.align import SIZE, align_face, crop_face
from app.modules.face.detect import (DETECTORS, Detection, available_detectors, detect_face, detector_available,
                                     resolve_detector)
from app.modules.face.embed import (CLASSICAL, EMBEDDERS, available_embedders, eigen_basis, eigen_embedding, embed, embedder_available,
                                    embedder_title, resolve_embedder)
from app.modules.face.params import FaceParams
from app.modules.uploads import UploadStore

GALLERY = Path(__file__).resolve().parents[3] / "datasets" / "faces"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 640
SCAN_TTL = 60  # секунд: список файлов галереи не перечитывается при каждом запросе (в LFW это около секунды)
FLAT = "gallery"  # источник для папок людей прямо в datasets/faces/<человек>/ (старая раскладка)
LM_NAMES = ["левый глаз", "правый глаз", "нос", "левый угол рта", "правый угол рта"]
LM_COLORS = [(0, 220, 0), (0, 220, 0), (0, 0, 255), (255, 120, 0), (255, 120, 0)]
SRC_TEXT = {"yunet": "ключевые точки найдены нейросетью YuNet (5 точек)",
            "eyes-cascade": "глаза найдены каскадом Хаара, нос и рот оценены по пропорциям рамки",
            "proportional": "точки оценены по пропорциям рамки лица (глаза не найдены каскадом)"}


@dataclass
class FaceResult:
    ok: bool
    img: np.ndarray
    detector: str
    embedder: str
    det: Detection | None = None
    aligned: np.ndarray | None = None
    emb: np.ndarray | None = None
    steps: list = field(default_factory=list)
    facts: list = field(default_factory=list)


def _imread(path: Path) -> np.ndarray:
    # np.fromfile + imdecode: работает и с кириллицей в пути на Windows
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение {path.name}")
    return img


def _limit(img: np.ndarray) -> np.ndarray:
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = size / max(img.shape[:2])
    r = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 40, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def _draw_box(img, det: Detection):
    vis = img.copy()
    cv2.rectangle(vis, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 220, 0), 2)
    return vis


def _draw_landmarks(img, det: Detection):
    vis = _draw_box(img, det)
    r = max(2, det.w // 40)
    for (px, py), c in zip(det.landmarks, LM_COLORS):
        cv2.circle(vis, (int(px), int(py)), r, c, -1, cv2.LINE_AA)
    return vis


def _heatmap(vec: np.ndarray) -> np.ndarray:
    v = vec / (np.abs(vec).max() + 1e-9)
    cols = 32
    pad = (-len(v)) % cols
    m = np.pad(v, (0, pad)).reshape(-1, cols)
    u8 = ((m * 0.5 + 0.5) * 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(u8, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)


class FaceModule(BiometricModule):
    name = "face"
    title = "Распознавание лица"
    default_threshold = 0.40

    def __init__(self):
        self._uploads = UploadStore()
        self._cache = LRUCache(300)
        self._paths: dict[str, Path] = {}
        self._means: dict[str, np.ndarray] = {}
        self._scan_lock = threading.Lock()
        self._scan_cache: dict | None = None
        self._scan_time = 0.0
        self._scan_root: Path | None = None

    # ---------- галерея и образцы ----------
    def _scan_all(self, force: bool = False) -> dict[str, dict[str, list[tuple[str, Path]]]]:
        """Источник -> человек -> [(идентификатор, путь)]. Поддерживаются обе раскладки: datasets/faces/<источник>/<человек>/<фото>
        (например, LFW) и datasets/faces/<человек>/<фото>. Результат кэшируется на SCAN_TTL секунд."""
        with self._scan_lock:  # запросы приходят параллельно: сканируем в локальные словари
            if (not force and self._scan_cache is not None and self._scan_root == GALLERY
                    and time.monotonic() - self._scan_time < SCAN_TTL):
                return self._scan_cache
            sources: dict[str, dict[str, list[tuple[str, Path]]]] = {}
            paths: dict[str, Path] = {}

            def add(src: str, person: str, folder: str, files: list[str], prefix: str) -> None:
                for name in files:
                    sid = f"{prefix}{person}__{Path(name).stem}"
                    if sid not in paths:
                        paths[sid] = Path(folder) / name
                        sources.setdefault(src, {}).setdefault(person, []).append((sid, paths[sid]))

            def images(folder: str) -> list[str]:
                return sorted(e.name for e in os.scandir(folder) if e.is_file() and Path(e.name).suffix.lower() in EXT)

            if GALLERY.exists():
                for d in sorted((e for e in os.scandir(GALLERY) if e.is_dir()), key=lambda e: e.name):
                    own = images(d.path)
                    if own:  # папка человека прямо в галерее
                        add(FLAT, d.name, d.path, own, "")
                        continue
                    for pd in sorted((e for e in os.scandir(d.path) if e.is_dir()), key=lambda e: e.name):  # папка источника
                        add(d.name, pd.name, pd.path, images(pd.path), f"{d.name}::")
            self._paths, self._scan_cache, self._scan_time, self._scan_root = paths, sources, time.monotonic(), GALLERY
            return sources

    def _scan(self, source: str | None = None) -> dict[str, list[tuple[str, Path]]]:
        """Люди выбранного источника (None или «all» - всех источников вместе)."""
        allsrc = self._scan_all()
        if not source or source == "all":
            people: dict[str, list[tuple[str, Path]]] = {}
            for src in allsrc.values():
                for person, lst in src.items():
                    people.setdefault(person, []).extend(lst)
            return people
        if source not in allsrc:
            raise KeyError(f"Источник лиц «{source}» не найден в datasets/faces/")
        return allsrc[source]

    def list_samples(self, source: str | None = None, q: str = "", limit: int = 0) -> list[dict[str, Any]]:
        people = self._scan(source)
        items = []
        for person, lst in sorted(people.items(), key=lambda kv: len(kv[1]) < 2):  # сначала люди, у которых есть пара фото для «своих» сравнений
            for sid, f in lst:
                label = f"{person} / {f.name}"
                if matches(q, label):
                    items.append({"id": sid, "label": label, "person": person})
        if limit:
            items = items[:limit]
        return [{"id": uid, "label": f"Загружено: {uid}", "person": None} for uid in self._uploads.ids()] + items

    def info(self) -> dict[str, Any]:
        allsrc = self._scan_all()
        people = self._scan()
        det, emb = available_detectors(), available_embedders()
        missing = [DETECTORS[k] for k in ("yunet", "ssd", "ultraface") if not det[k]] + [EMBEDDERS[k] for k in ("sface", "arcface") if not emb[k]]
        neural_emb = emb["sface"] or emb["arcface"]
        sources = [{"id": "all", "title": "Все источники галереи", "real": True, "images": sum(len(v) for v in people.values()),
                    "note": "Реальные фотографии людей (LFW и ваши). Для сравнения методов нужны люди минимум с двумя фото."}]
        for src, ppl in allsrc.items():
            sources.append({"id": src, "title": "Ваши фото (datasets/faces/<имя>/)" if src == FLAT else f"Галерея «{src}» (datasets/faces/{src}/<имя>/)",
                            "real": True, "images": sum(len(v) for v in ppl.values()), "persons": len(ppl),
                            "note": "Реальные фотографии людей. Метрики зависят от детектора и признаков; сравните методы на одном наборе."})
        return {
            "detectors": det, "embedders": emb, "sources": sources, "default_source": "all",
            "persons": len(people), "images": sum(len(v) for v in people.values()),
            "hint": ("" if neural_emb or len(people) >= 3 else
                     "Для классических признаков (LBP, HOG, Габор, DCT, пиксели) нужна галерея минимум из 3 разных людей "
                     "(иначе сходство у всех лиц будет завышено). ")
                    + ("" if not missing else
                       "Не скачаны нейросетевые модели: " + "; ".join(missing) + ". Работают классические методы. "
                       "Для полной версии выполните: python -m scripts.download_models и перезапустите сервер. ")
                    + ("" if people else "Галерея пуста: положите фото в datasets/faces/<имя_человека>/ (или datasets/faces/<набор>/<имя_человека>/) или загрузите свои изображения."),
        }

    def _load(self, sample_id: str, p: FaceParams) -> np.ndarray:
        up = self._uploads.get(sample_id)
        if up is not None:
            img = up.copy()
        else:
            if sample_id not in self._paths:
                self._scan_all(force=True)
            if sample_id not in self._paths:
                raise KeyError(f"Неизвестный образец: {sample_id}")
            img = _limit(_imread(self._paths[sample_id]))
        if p.extra_noise > 0:
            seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
            noise = np.random.RandomState(seed).randn(*img.shape) * p.extra_noise
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def get_sample_preview(self, sample_id: str, params: dict) -> bytes:
        ok, buf = cv2.imencode(".png", self._load(sample_id, FaceParams(**params)))
        return buf.tobytes()

    def upload(self, data: bytes) -> str:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        return self._uploads.add(_limit(img))

    # ---------- конвейер ----------
    def _run(self, sample_id: str, p: FaceParams, with_images: bool = True) -> FaceResult:
        key = f"{sample_id}|{p.model_dump_json(exclude={'source'})}|{with_images}"
        res = self._cache.get(key)
        if res is None:
            res = self._pipeline(sample_id, p, with_images)
            self._cache.put(key, res)
        return res

    def _pipeline(self, sample_id: str, p: FaceParams, with_images: bool) -> FaceResult:
        img = self._load(sample_id, p)
        detector, embedder = resolve_detector(p), resolve_embedder(p)
        res = FaceResult(False, img, detector, embedder)
        h, w = img.shape[:2]

        def add(i, t, d, im):
            if with_images:
                res.steps.append(dict(id=i, title=t, description=d, image=im))

        add("original", "1. Исходное изображение", f"Входная фотография {w}×{h} пикселей. Для скорости крупные кадры уменьшаются до {MAX_SIDE} px по большей стороне.", img)
        det = detect_face(img, p, detector)
        dname = DETECTORS[detector]
        if det is None:
            add("detection", "2. Обнаружение лица", f"Детектор: {dname}. Лицо не найдено. Попробуйте снизить min neighbors, "
                "уменьшить минимальный размер лица или выбрать другой детектор.", img)
            res.facts = [["Детектор", dname], ["Результат", "лицо не найдено"]]
            return res
        res.det = det
        conf_note = " (для Хаара это условная оценка по весу каскада)" if detector == "haar" else ""
        add("detection", "2. Обнаружение лица",
            f"Детектор: {dname}. Рамка: x={det.x}, y={det.y}, width={det.w}, height={det.h}. Confidence = {det.confidence:.2f}{conf_note}.",
            _draw_box(img, det))
        add("landmarks", "3. Ключевые точки лица",
            f"Зелёные - глаза, красная - нос, оранжевые - углы рта; {SRC_TEXT[det.lm_source]}.", _draw_landmarks(img, det))

        crop = crop_face(img, (det.x, det.y, det.w, det.h))
        if p.align:
            # по найденным глазам (Хаар) выравниваем по 2 точкам, по YuNet - по всем 5
            res.aligned = align_face(img, det.landmarks, use_all=det.lm_source != "eyes-cascade")
            tilt = float(np.degrees(np.arctan2(det.landmarks[1][1] - det.landmarks[0][1], det.landmarks[1][0] - det.landmarks[0][0])))
            desc = f"Лицо поворачивается, масштабируется и сдвигается так, чтобы глаза и рот встали в стандартные позиции {SIZE}×{SIZE}. Наклон линии глаз до выравнивания: {tilt:.1f}°. Слева - простое вырезание, справа - после выравнивания."
        else:
            res.aligned = crop
            desc = "Выравнивание отключено: лицо только вырезано по рамке. Слева и справа одно и то же. Включите выравнивание и сравните similarity."
        add("aligned", "4. Выравнивание (alignment)", desc, np.hstack([cv2.resize(crop, (224, 224)), cv2.resize(res.aligned, (224, 224))]))

        norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cv2.cvtColor(res.aligned, cv2.COLOR_BGR2GRAY))
        add("normalized", "5. Нормализация освещения (CLAHE)",
            "Локальное выравнивание контраста уменьшает влияние освещения. Классические признаки (LBP, HOG, Габор, DCT, пиксели) строятся по этому серому изображению; нейросети (SFace, ArcFace) принимают выровненное цветное лицо.",
            cv2.resize(norm, (224, 224), interpolation=cv2.INTER_NEAREST))

        res.emb = self._embed(res.aligned, embedder, p)
        add("embedding", "6. Face embedding",
            f"{embedder_title(embedder)}. Размерность {len(res.emb)}, норма {np.linalg.norm(res.emb):.3f}. Каждая клетка - одно число вектора признаков: лицо превращено в набор чисел, которые и сравниваются.",
            _heatmap(res.emb))
        res.facts = [["Детектор", dname], ["Рамка (x, y, w, h)", f"{det.x}, {det.y}, {det.w}, {det.h}"],
                     ["Confidence", f"{det.confidence:.2f}"], ["Ключевые точки", SRC_TEXT[det.lm_source]],
                     ["Выравнивание", "включено" if p.align else "выключено"], ["Признаки", embedder_title(embedder)],
                     ["Размерность embedding", len(res.emb)], ["Норма вектора", f"{np.linalg.norm(res.emb):.3f}"]]
        res.ok = True
        return res

    def process(self, sample_id: str, params: dict) -> dict[str, Any]:
        r = self._run(sample_id, FaceParams(**params))
        steps = [dict(s, image=to_data_url(s["image"])) for s in r.steps]
        return {"sample_id": sample_id, "steps": steps, "facts": r.facts, "found": r.ok}

    # ---------- сравнение ----------
    def _embed(self, aligned: np.ndarray, name: str, p: FaceParams) -> np.ndarray:
        if name == "cnn":
            from app.training import registry
            return registry.embed("face", cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        return eigen_embedding(aligned, self._eigen(p)) if name == "eigenfaces" else embed(aligned, name)

    def _eigen(self, p: FaceParams):
        """Базис Eigenfaces по выровненным лицам галереи (кэшируется до изменения галереи или детектора)."""
        p0 = p.model_copy(update={"extra_noise": 0.0, "embedder": "lbp"})
        key = "eigen|" + p0.model_dump_json() + str(len(self._paths))
        if key not in self._means:
            faces = []
            for lst in self._scan(p.source).values():
                for sid, _ in lst[:4]:
                    r = self._run(sid, p0, with_images=False)
                    if r.ok:
                        faces.append(r.aligned)
                if len(faces) >= 160:
                    break
            self._means[key] = eigen_basis(faces)
        return self._means[key]

    def _gallery_mean(self, p: FaceParams, embedder: str) -> np.ndarray | None:
        """Классические признаки похожи у всех лиц, поэтому вычитаем средний вектор галереи («среднее лицо»).
        Среднее имеет смысл только при 3+ разных людях, иначе оно вырождается и возвращается None."""
        p0 = p.model_copy(update={"extra_noise": 0.0, "embedder": embedder})
        key = p0.model_dump_json() + str(len(self._paths))
        if key not in self._means:
            vecs, persons = [], set()
            for person, lst in self._scan(p.source).items():
                for sid, _ in lst[:3]:
                    r = self._run(sid, p0, with_images=False)
                    if r.ok:
                        vecs.append(r.emb)
                        persons.add(person)
                if len(vecs) >= 40:
                    break
            self._means[key] = np.mean(vecs, 0) if len(persons) >= 3 else None
        return self._means[key]

    def _similarity(self, a: FaceResult, b: FaceResult, p: FaceParams, mean=None) -> float:
        va, vb = a.emb, b.emb
        if a.embedder in CLASSICAL and mean is not None:
            va, vb = va - mean, vb - mean
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    def compare(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        p = FaceParams(**params)
        ra, rb = self._run(sample_a, p), self._run(sample_b, p)
        for lab, r in (("A", ra), ("B", rb)):
            if not r.ok:
                raise ValueError(f"На образце {lab} лицо не найдено. Измените параметры детекции или выберите другой образец.")
        mean = self._gallery_mean(p, ra.embedder) if ra.embedder in CLASSICAL else None
        sim = self._similarity(ra, rb, p, mean)

        def tile(r):
            top = _letterbox(_draw_landmarks(r.img, r.det), 256)
            return np.vstack([top, _letterbox(r.aligned, 256)])
        canvas = np.hstack([tile(ra), tile(rb)])
        cv2.putText(canvas, "A", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "B", (264, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"cos = {sim:.2f}", (180, 506), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
        details = [["Детектор", DETECTORS[ra.detector]], ["Признаки", embedder_title(ra.embedder)],
                   ["Размерность", len(ra.emb)], ["Метрика", "cosine similarity"],
                   ["Выравнивание", "да" if p.align else "нет"]]
        return {"similarity": round(sim, 4), "n_a": 0, "n_b": 0, "matched": 0, "details": details,
                "context": {"modality": "face", "detector": ra.detector, "embedder": ra.embedder,
                            "align": p.align, "extra_noise": p.extra_noise},
                "overlay": to_data_url(canvas)}

    # ---------- метрики ----------
    def evaluate(self, params: dict, n_subjects: int = 10) -> dict[str, Any]:
        p = FaceParams(**params)
        people = {k: v for k, v in self._scan(p.source).items() if len(v) >= 2}
        names = list(people)[:max(2, n_subjects)]
        if len(names) < 2:
            raise ValueError("Для оценки метрик нужны минимум 2 человека, у каждого минимум 2 фото в папке datasets/faces/<имя>/.")
        res: dict[str, list[FaceResult]] = {}
        skipped = 0
        for n in names:
            ok = []
            for sid, _ in people[n][:6]:
                r = self._run(sid, p, with_images=False)
                ok.append(r) if r.ok else None
                skipped += 0 if r.ok else 1
            if len(ok) >= 1:
                res[n] = ok
        embedder = resolve_embedder(p)
        mean = self._gallery_mean(p, embedder) if embedder in CLASSICAL else None
        genuine = [self._similarity(a, b, p, mean) for lst in res.values() for a, b in itertools.combinations(lst, 2)]
        impostor = [self._similarity(a, b, p, mean) for (n1, l1), (n2, l2) in itertools.combinations(res.items(), 2)
                    for a in l1[:2] for b in l2[:2]]
        if not genuine or not impostor:
            raise ValueError("Недостаточно пар: на слишком многих фото лицо не найдено. Смените детектор или параметры.")
        th = np.round(np.arange(0, 1.0001, 0.01), 2)
        g, im = np.array(genuine), np.array(impostor)
        frr = [float((g < t).mean()) for t in th]
        far = [float((im >= t).mean()) for t in th]
        k = int(np.argmin(np.abs(np.array(far) - np.array(frr))))
        return {"thresholds": th.tolist(), "far": far, "frr": frr, "eer": round((far[k] + frr[k]) / 2, 4),
                "eer_threshold": float(th[k]), "genuine_scores": [round(float(s), 4) for s in genuine],
                "impostor_scores": [round(float(s), 4) for s in impostor],
                "n_genuine": len(genuine), "n_impostor": len(impostor), "skipped_images": skipped}

    # ---------- сравнение методов ----------
    def compare_detectors(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Запускает все детекторы на одном изображении: рамки разных цветов, время и уверенность."""
        p = FaceParams(**params)
        img = self._load(sample_id, p)
        palette = [(0, 200, 0), (0, 140, 255), (200, 0, 200), (255, 200, 0), (0, 0, 255), (255, 80, 0)]  # BGR
        vis, rows = img.copy(), []
        for k, (name, title) in enumerate(DETECTORS.items()):
            c = palette[k % len(palette)]
            row = {"n": k + 1, "name": name, "title": title, "available": detector_available(name),
                   "color": "#%02x%02x%02x" % (c[2], c[1], c[0])}
            if not row["available"]:
                row["note"] = "Сеть для лиц ещё не обучена: вкладка «Обучение распознаванию», данные «Лица»." if name == "cnn" else "модель не скачана"
                rows.append(row)
                continue
            t = time.perf_counter()
            det = detect_face(img, p, name)
            row.update(time_ms=round((time.perf_counter() - t) * 1000, 1), found=det is not None)
            if det is not None:
                row.update(box=[det.x, det.y, det.w, det.h], confidence=round(det.confidence, 3), lm_source=SRC_TEXT[det.lm_source])
                cv2.rectangle(vis, (det.x, det.y), (det.x + det.w, det.y + det.h), c, 2)
                cv2.putText(vis, str(k + 1), (det.x + 4 + k * 16, det.y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
            rows.append(row)
        return {"sample_id": sample_id, "rows": rows, "overlay": to_data_url(vis)}

    def compare_embedders(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        """Одна и та же пара выровненных лиц, но разные признаки: сходство, размерность и время."""
        p = FaceParams(**params)
        pa = p.model_copy(update={"embedder": "lbp"})  # для выравнивания нужен только детектор
        ra, rb = self._run(sample_a, pa), self._run(sample_b, pa)
        for lab, r in (("A", ra), ("B", rb)):
            if not r.ok:
                raise ValueError(f"На образце {lab} лицо не найдено. Измените параметры детекции или выберите другой образец.")
        rows = []
        for name, title in EMBEDDERS.items():
            row = {"name": name, "title": title, "available": embedder_available(name)}
            if not row["available"]:
                row["note"] = "Сеть для лиц ещё не обучена: вкладка «Обучение распознаванию», данные «Лица»." if name == "cnn" else "модель не скачана"
                rows.append(row)
                continue
            t = time.perf_counter()
            try:
                ea, eb = self._embed(ra.aligned, name, p), self._embed(rb.aligned, name, p)
            except ValueError as e:  # например, для Eigenfaces мала галерея
                row.update(available=False, note=str(e))
                rows.append(row)
                continue
            ms = (time.perf_counter() - t) * 1000 / 2
            mean = self._gallery_mean(p, name) if name in CLASSICAL else None
            if mean is not None:
                ea, eb = ea - mean, eb - mean
            sim = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
            row.update(dim=int(len(ea)), similarity=round(sim, 4), time_ms=round(ms, 1), centered=mean is not None)
            rows.append(row)
        up = lambda im: to_data_url(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC))
        return {"rows": rows, "detector": DETECTORS[ra.detector], "aligned_a": up(ra.aligned), "aligned_b": up(rb.aligned)}

    def evaluate_embedders(self, params: dict, n_subjects: int = 8) -> dict[str, Any]:
        """EER каждого признака на галерее: какой метод лучше различает людей (выбранным детектором)."""
        rows = []
        for name, title in EMBEDDERS.items():
            row = {"name": name, "title": title, "available": embedder_available(name)}
            if not row["available"]:
                row["note"] = "Сеть для лиц ещё не обучена: вкладка «Обучение распознаванию», данные «Лица»." if name == "cnn" else "модель не скачана"
                rows.append(row)
                continue
            t = time.perf_counter()
            try:
                ev = self.evaluate({**params, "embedder": name}, n_subjects)
            except ValueError as e:
                if name != "eigenfaces":
                    raise
                row.update(available=False, note=str(e))
                rows.append(row)
                continue
            g, im = np.array(ev["genuine_scores"]), np.array(ev["impostor_scores"])
            if name == "cnn":
                row["title"] += " (галерея входила в обучение: оценка завышена)"
            row.update(eer=ev["eer"], eer_threshold=ev["eer_threshold"], mean_genuine=round(float(g.mean()), 3),
                       mean_impostor=round(float(im.mean()), 3), n_genuine=ev["n_genuine"], n_impostor=ev["n_impostor"],
                       time_s=round(time.perf_counter() - t, 1))
            rows.append(row)
        return {"rows": rows, "detector": DETECTORS[resolve_detector(FaceParams(**params))]}
