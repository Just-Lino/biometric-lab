import hashlib
import itertools
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.imaging import to_data_url
from app.modules.base import BiometricModule
from app.modules.face import models
from app.modules.face.align import SIZE, align_face, crop_face
from app.modules.face.detect import Detection, detect_face, resolve_detector
from app.modules.face.embed import embed, embedder_title, resolve_embedder
from app.modules.face.params import FaceParams

GALLERY = Path(__file__).resolve().parents[3] / "datasets" / "faces"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 640
MAX_UPLOADS = 20
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
        self._uploads: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache: OrderedDict[str, FaceResult] = OrderedDict()
        self._paths: dict[str, Path] = {}
        self._means: dict[str, np.ndarray] = {}
        self._scan_lock = threading.Lock()

    # ---------- галерея и образцы ----------
    def _scan(self) -> dict[str, list[tuple[str, Path]]]:
        with self._scan_lock:  # запросы приходят параллельно: сканируем в локальные словари
            people: dict[str, list[tuple[str, Path]]] = {}
            paths: dict[str, Path] = {}
            if GALLERY.exists():
                for d in sorted(p for p in GALLERY.iterdir() if p.is_dir()):
                    for f in sorted(d.iterdir()):
                        if f.suffix.lower() in EXT:
                            sid = f"{d.name}__{f.stem}"
                            if sid not in paths:
                                paths[sid] = f
                                people.setdefault(d.name, []).append((sid, f))
            self._paths = paths
            return people

    def list_samples(self) -> list[dict[str, Any]]:
        items = [{"id": sid, "label": f"{person} / {f.name}", "person": person}
                 for person, lst in self._scan().items() for sid, f in lst]
        items += [{"id": uid, "label": f"Загружено: {uid}", "person": None} for uid in self._uploads]
        return items

    def info(self) -> dict[str, Any]:
        people = self._scan()
        have_y, have_s = models.model_path("yunet") is not None, models.model_path("sface") is not None
        return {
            "detectors": {"haar": True, "yunet": have_y}, "embedders": {"lbp": True, "sface": have_s},
            "persons": len(people), "images": sum(len(v) for v in people.values()),
            "hint": ("" if have_s or len(people) >= 3 else
                     "Для классического LBP нужна галерея минимум из 3 разных людей (иначе сходство у всех лиц будет завышено). ")
                    + ("" if have_y and have_s else
                     "Нейросетевые модели (YuNet, SFace) не найдены, работают Haar + LBP. "
                     "Для полной версии выполните: python -m scripts.download_models и перезапустите сервер. ")
                    + ("" if people else "Галерея пуста: положите фото в datasets/faces/<имя_человека>/ или загрузите свои изображения."),
        }

    def _load(self, sample_id: str, p: FaceParams) -> np.ndarray:
        if sample_id in self._uploads:
            img = self._uploads[sample_id].copy()
        else:
            if sample_id not in self._paths:
                self._scan()
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
        uid = "up_" + uuid.uuid4().hex[:6]
        self._uploads[uid] = _limit(img)
        while len(self._uploads) > MAX_UPLOADS:
            self._uploads.popitem(last=False)
        return uid

    # ---------- конвейер ----------
    def _run(self, sample_id: str, p: FaceParams, with_images: bool = True) -> FaceResult:
        key = f"{sample_id}|{p.model_dump_json()}|{with_images}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        res = self._pipeline(sample_id, p, with_images)
        self._cache[key] = res
        while len(self._cache) > 300:
            self._cache.popitem(last=False)
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
        dname = "YuNet (нейросеть)" if detector == "yunet" else "каскад Хаара (классический метод)"
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
            "Локальное выравнивание контраста уменьшает влияние освещения. Классический LBP использует именно это изображение; SFace принимает выровненное цветное лицо.",
            cv2.resize(norm, (224, 224), interpolation=cv2.INTER_NEAREST))

        res.emb = embed(res.aligned, embedder)
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
    def _lbp_mean(self, p: FaceParams) -> np.ndarray | None:
        """LBP-векторы похожи у всех лиц, поэтому вычитаем средний вектор галереи («среднее лицо»).
        Среднее имеет смысл только при 3+ разных людях, иначе оно вырождается и возвращается None."""
        p0 = p.model_copy(update={"extra_noise": 0.0, "embedder": "lbp"})
        key = p0.model_dump_json() + str(len(self._paths))
        if key not in self._means:
            vecs, persons = [], set()
            for person, lst in self._scan().items():
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
        if a.embedder == "lbp" and mean is not None:
            va, vb = va - mean, vb - mean
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    def compare(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        p = FaceParams(**params)
        ra, rb = self._run(sample_a, p), self._run(sample_b, p)
        for lab, r in (("A", ra), ("B", rb)):
            if not r.ok:
                raise ValueError(f"На образце {lab} лицо не найдено. Измените параметры детекции или выберите другой образец.")
        mean = self._lbp_mean(p) if ra.embedder == "lbp" else None
        sim = self._similarity(ra, rb, p, mean)

        def tile(r):
            top = _letterbox(_draw_landmarks(r.img, r.det), 256)
            return np.vstack([top, _letterbox(r.aligned, 256)])
        canvas = np.hstack([tile(ra), tile(rb)])
        cv2.putText(canvas, "A", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "B", (264, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"cos = {sim:.2f}", (180, 506), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
        details = [["Детектор", ra.detector], ["Признаки", embedder_title(ra.embedder)],
                   ["Размерность", len(ra.emb)], ["Метрика", "cosine similarity"],
                   ["Выравнивание", "да" if p.align else "нет"]]
        return {"similarity": round(sim, 4), "n_a": 0, "n_b": 0, "matched": 0, "details": details,
                "context": {"modality": "face", "detector": ra.detector, "embedder": ra.embedder,
                            "align": p.align, "extra_noise": p.extra_noise},
                "overlay": to_data_url(canvas)}

    # ---------- метрики ----------
    def evaluate(self, params: dict, n_subjects: int = 10) -> dict[str, Any]:
        p = FaceParams(**params)
        people = {k: v for k, v in self._scan().items() if len(v) >= 2}
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
        mean = self._lbp_mean(p) if embedder == "lbp" else None
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
