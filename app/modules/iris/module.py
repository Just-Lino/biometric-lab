import hashlib
import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.cache import LRUCache
from app.core.imaging import to_data_url
from app.core.realdata import matches
from app.modules.base import BiometricModule
from app.modules.iris import real, synthetic
from app.modules.uploads import UploadStore
from app.modules.iris.encode import ANG, ENCODERS, RAD, Template, encode, match, normalize, occlusion_mask
from app.training import registry
from app.training.datasets import iris_crop
from app.modules.iris.localize import LOCALIZERS, Circles, localize, preprocess
from app.modules.iris.params import IrisParams

SYNTHETIC = "synthetic"
SYNTHETIC_TITLE = "Синтетика лаборатории (12 глаз × 3 снимка)"
COLORS = [(0, 0, 255), (0, 170, 0), (255, 120, 0), (200, 0, 200)]  # BGR


@dataclass
class IrisResult:
    ok: bool
    gray: np.ndarray
    method: str = ""
    circles: Circles | None = None
    strip: np.ndarray | None = None
    mask: np.ndarray | None = None
    maps: tuple | None = None
    tpl: Template | None = None
    steps: list = field(default_factory=list)
    facts: list = field(default_factory=list)
    emb: np.ndarray | None = None  # embedding сети (кодировщик «cnn»)


def _bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _draw(img, c: Circles, colors=((0, 0, 255), (0, 200, 0)), th=2):
    for (x, y, r), col in zip((c.pupil, c.iris), colors):
        cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)), col, th, cv2.LINE_AA)
        cv2.drawMarker(img, (int(round(x)), int(round(y))), col, cv2.MARKER_CROSS, 8, 1)
    return img


def _strip_img(strip, mask=None, scale=2):
    v = _bgr(np.clip(strip, 0, 255).astype(np.uint8))
    if mask is not None:
        v[~mask] = (0.45 * v[~mask] + 0.55 * np.array([60, 60, 230])).astype(np.uint8)  # исключённые точки - красным
    return cv2.resize(v, (ANG * scale, RAD * scale), interpolation=cv2.INTER_NEAREST)


def _response_img(r):
    m = np.abs(r).max() + 1e-6
    t = (r / m).clip(-1, 1)
    out = np.zeros(r.shape + (3,), np.uint8)
    out[..., 2] = (255 * np.clip(t, 0, 1)).astype(np.uint8) + (180 * (1 - np.abs(t))).astype(np.uint8)  # красный: +
    out[..., 0] = (255 * np.clip(-t, 0, 1)).astype(np.uint8) + (180 * (1 - np.abs(t))).astype(np.uint8)  # синий: -
    out[..., 1] = (180 * (1 - np.abs(t))).astype(np.uint8)
    return cv2.resize(out, (ANG * 2, RAD * 2), interpolation=cv2.INTER_NEAREST)


def _code_img(t: Template, planes=4):
    rows = []
    for k in range(min(planes, t.code.shape[0])):
        v = np.where(t.code[k], 255, 0).astype(np.uint8)
        v[~t.mask[k]] = 128  # ненадёжные и закрытые биты - серые
        rows.append(cv2.resize(v, (ANG * 2, 48), interpolation=cv2.INTER_NEAREST))
        rows.append(np.full((4, ANG * 2), 230, np.uint8))
    return _bgr(np.vstack(rows[:-1]))


class IrisModule(BiometricModule):
    name = "iris"
    title = "Распознавание радужной оболочки"
    default_threshold = 0.66

    def __init__(self):
        self._uploads = UploadStore()
        self._cache = LRUCache(80)

    # ---------- образцы ----------
    def list_samples(self, source: str | None = None, q: str = "", limit: int = 0) -> list[dict[str, Any]]:
        source = source or SYNTHETIC
        if source == SYNTHETIC:
            items = [{"id": f"e{e:02d}_i{i}", "label": f"Глаз {e:02d}, снимок {i}", "person": f"e{e:02d}"}
                     for e in range(1, synthetic.N_EYES + 1) for i in range(1, synthetic.N_IMPRESSIONS + 1)]
        else:
            items = real.list_samples(source)
        items = [x for x in items if matches(q, x["label"], x["id"])]
        if limit:
            items = items[:limit]
        return [{"id": u, "label": f"Загружено: {u}", "person": None} for u in self._uploads.ids()] + items

    def _truth(self, sample_id: str) -> dict | None:
        if sample_id in self._uploads or real.is_real_id(sample_id):
            return None  # у загрузок и реальных снимков истинных окружностей нет
        e, i = sample_id.split("_")
        return synthetic.impression(int(e[1:]), int(i[1:]))[1]

    def _gray(self, sample_id: str, p: IrisParams) -> np.ndarray:
        up = self._uploads.get(sample_id)
        if up is not None:
            img = up.copy()
        elif real.is_real_id(sample_id):
            img = real.load(sample_id)
        else:
            try:
                e, i = sample_id.split("_")
                img = synthetic.impression(int(e[1:]), int(i[1:]))[0].copy()
            except (ValueError, KeyError):
                raise KeyError(f"Неизвестный образец: {sample_id}")
        if p.extra_noise > 0:
            seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
            img = np.clip(img.astype(np.float32) + np.random.RandomState(seed).randn(*img.shape) * p.extra_noise, 0, 255).astype(np.uint8)
        return img

    def get_sample_preview(self, sample_id: str, params: dict) -> bytes:
        return cv2.imencode(".png", self._gray(sample_id, IrisParams(**params)))[1].tobytes()

    def upload(self, data: bytes) -> str:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        s = min(synthetic.W / img.shape[1], synthetic.H / img.shape[0])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvas = np.full((synthetic.H, synthetic.W), int(np.median(img)), np.uint8)
        y0, x0 = (synthetic.H - img.shape[0]) // 2, (synthetic.W - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        return self._uploads.add(canvas)

    # ---------- обработка ----------
    def _locate(self, pre, p: IrisParams):
        order = [p.localizer] if p.localizer != "auto" else ["daugman", "contour", "hough"]  # авто: по очереди до успеха
        for m in order:
            c = localize(pre, p, m)
            if c is not None:
                return c
        return None

    def _run(self, sample_id: str, p: IrisParams, with_images: bool = True) -> IrisResult:
        key = f"{sample_id}|{p.model_dump_json()}|{with_images}"
        res = self._cache.get(key)
        if res is None:
            res = self._pipeline(sample_id, p, with_images)
            self._cache.put(key, res)
        return res

    def _pipeline(self, sample_id: str, p: IrisParams, with_images: bool) -> IrisResult:
        gray = self._gray(sample_id, p)
        res = IrisResult(ok=False, gray=gray)
        add = (lambda i, t, d, im: res.steps.append({"id": i, "title": t, "description": d, "desc": d, "image": im})) if with_images else (lambda *a: None)
        add("original", "1. Исходное изображение", f"Снимок глаза в оттенках серого (как в ближнем ИК-диапазоне), {gray.shape[1]}×{gray.shape[0]}. "
            "Распознаётся текстура радужной оболочки; зрачок служит ориентиром для её поиска.", _bgr(gray))
        pre = preprocess(gray, p)
        add("preprocess", "2. Предобработка", f"Гауссово сглаживание (σ = {p.blur}) убирает шум и мелкие детали, чтобы границы зрачка и радужки "
            "находились устойчиво.", _bgr(pre))
        c = self._locate(pre, p)
        if c is None:
            add("pupil", "3. Локализация зрачка", "Зрачок или граница радужки не найдены. Смените метод локализации, уменьшите сглаживание "
                "или порог Хафа.", _bgr(gray))
            res.facts = [["Метод локализации", "не удалось найти окружности"]]
            return res
        res.circles, res.method = c, c.method
        (px, py, pr), (ix, iy, ir) = c.pupil, c.iris
        img3 = _bgr(gray)
        cv2.circle(img3, (int(px), int(py)), int(pr), (0, 0, 255), 2, cv2.LINE_AA)
        add("pupil", "3. Локализация зрачка", f"Метод: {LOCALIZERS[c.method]}. Зрачок: центр ({px:.1f}; {py:.1f}), радиус {pr:.1f} px. "
            "Зрачок - самая тёмная круглая область, его граница - резкий переход к радужке.", img3)
        add("iris", "4. Граница радужки", f"Внешняя граница (лимб): центр ({ix:.1f}; {iy:.1f}), радиус {ir:.1f} px. Центры зрачка и радужки "
            f"обычно не совпадают: смещение {np.hypot(px - ix, py - iy):.1f} px. Ищется по боковым дугам, которые не закрыты веками.", _draw(_bgr(gray), c))
        strip, valid, maps = normalize(gray, c)
        mask = occlusion_mask(strip, valid, p.mask_eyelids)
        res.strip, res.mask, res.maps = strip, mask, maps
        if with_images:
            seg = _bgr(gray)
            over = seg.copy()
            mx, my = maps
            for yy, xx, ok in zip(my.ravel(), mx.ravel(), mask.ravel()):
                if 0 <= xx < gray.shape[1] and 0 <= yy < gray.shape[0]:
                    over[int(yy), int(xx)] = (60, 200, 60) if ok else (60, 60, 230)
            seg = cv2.addWeighted(seg, 0.45, cv2.dilate(over, np.ones((2, 2), np.uint8)), 0.55, 0)
            _draw(seg, c, th=1)
            add("segment", "5. Сегментация", f"Кольцо между зрачком и лимбом - область радужки. Зелёные точки участвуют в сравнении, красные исключены "
                f"(веки, ресницы, блики). Открыто {mask.mean() * 100:.0f} % радужки.", seg)
            add("normalize", "6. Нормализация (rubber sheet)", f"Модель Даугмана «резиновый лист»: кольцо разворачивается в прямоугольник {RAD}×{ANG} "
                "(радиус × угол). Так снимки с разным размером зрачка и положением глаза становятся сравнимыми; поворот глаза превращается "
                "в циклический сдвиг по горизонтали. Красным - исключённые точки.", _strip_img(strip, mask))
        if p.encoder == "cnn":
            registry.require("iris", "радужки", p.source)
            crop = iris_crop(gray, c)
            res.emb = registry.embed("iris", crop)
            if with_images:
                add("features", "7. Вход нейросети", "Квадрат вокруг найденной радужки: сеть получает его вместо развёртки (при обучении поворот этой вырезки "
                    "имитировал поворот глаза). Размер 96×96, внутри сети уменьшается до 64×64.", _bgr(cv2.resize(crop, (192, 192), interpolation=cv2.INTER_CUBIC)))
                v = res.emb / (np.abs(res.emb).max() + 1e-9)
                heat = cv2.resize(np.clip(v.reshape(4, 8) * 255, 0, 255).astype(np.uint8), (256, 128), interpolation=cv2.INTER_NEAREST)
                add("code", "8. Embedding сети", f"32 числа (4×8 клетки: светлее - больше значение). Это шаблон радужки для нейросети: два снимка сравниваются "
                    f"косинусным сходством векторов (чем ближе к 1, тем больше похожи).", cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS))
            res.ok = True
            res.facts = [["Метод локализации", LOCALIZERS[c.method]], ["Зрачок (x, y, r)", f"{px:.1f}, {py:.1f}, {pr:.1f}"],
                         ["Радужка (x, y, R)", f"{ix:.1f}, {iy:.1f}, {ir:.1f}"], ["Открыто радужки", f"{mask.mean() * 100:.0f} %"],
                         ["Кодировщик", ENCODERS["cnn"]], ["Embedding", "32 числа, норма 1"]]
            return res
        tpl = encode(strip, mask, p.encoder, p)
        res.tpl = tpl
        bits = int(tpl.code.size)
        good = int(tpl.mask.sum())
        if with_images:
            add("features", "7. Извлечение признаков", f"{ENCODERS[p.encoder]}. Показан отклик первого фильтра: красный - положительный, "
                "синий - отрицательный. Знак (а для Габора - фаза) этого отклика и превращается в биты.", _response_img(tpl.response))
            add("code", "8. Iris Code", f"Бинарный шаблон: {bits} бит в {tpl.code.shape[0]} слоях (показаны до четырёх). Белый - 1, чёрный - 0, "
                f"серый - ненадёжные или закрытые биты, они не сравниваются. Надёжных бит: {good} ({good / bits * 100:.0f} %). "
                "Два шаблона сравниваются по доле несовпавших битов (расстояние Хэмминга).", _code_img(tpl))
        res.ok = True
        res.facts = [["Метод локализации", LOCALIZERS[c.method]], ["Зрачок (x, y, r)", f"{px:.1f}, {py:.1f}, {pr:.1f}"],
                     ["Радужка (x, y, R)", f"{ix:.1f}, {iy:.1f}, {ir:.1f}"], ["Отношение радиусов зрачок / радужка", f"{pr / ir:.2f}"],
                     ["Открыто радужки", f"{mask.mean() * 100:.0f} %"], ["Кодировщик", ENCODERS[p.encoder]],
                     ["Длина кода", f"{bits} бит, надёжных {good / bits * 100:.0f} %"]]
        truth = self._truth(sample_id)
        if truth:
            ep = np.hypot(px - truth["pupil"][0], py - truth["pupil"][1])
            ei = np.hypot(ix - truth["iris"][0], iy - truth["iris"][1])
            res.facts.append(["Ошибка относительно эталона (синтетика)", f"центр зрачка {ep:.1f} px, радиус {abs(pr - truth['pupil'][2]):.1f} px; "
                                                                       f"центр радужки {ei:.1f} px, радиус {abs(ir - truth['iris'][2]):.1f} px"])
        return res

    def process(self, sample_id: str, params: dict) -> dict[str, Any]:
        r = self._run(sample_id, IrisParams(**params))
        return {"sample_id": sample_id, "found": r.ok, "steps": [dict(s, image=to_data_url(s["image"])) for s in r.steps], "facts": r.facts}

    # ---------- сравнение ----------
    def compare(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        p = IrisParams(**params)
        ra, rb = self._run(sample_a, p), self._run(sample_b, p)
        for lab, r in (("A", ra), ("B", rb)):
            if not r.ok:
                raise ValueError(f"На образце {lab} не удалось найти зрачок и радужку. Смените метод локализации или параметры.")
        if p.encoder == "cnn":
            return self._compare_cnn(ra, rb, p)
        m = match(ra.tpl, rb.tpl, p.max_shift)
        sh = int(round(m["shift"] * ANG / ra.tpl.width))
        w = 640
        eyes = np.hstack([cv2.resize(_draw(_bgr(r.gray), r.circles), (w // 2, w * 3 // 8)) for r in (ra, rb)])
        sa = cv2.resize(_strip_img(ra.strip, ra.mask, 1), (w, 70), interpolation=cv2.INTER_NEAREST)
        sb = cv2.resize(_strip_img(np.roll(rb.strip, sh, 1), np.roll(rb.mask, sh, 1), 1), (w, 70), interpolation=cv2.INTER_NEAREST)
        ca, cb = ra.tpl.code[0], np.roll(rb.tpl.code[0], m["shift"], axis=1)
        mm = ra.tpl.mask[0] & np.roll(rb.tpl.mask[0], m["shift"], axis=1)
        x = np.full(ca.shape + (3,), 150, np.uint8)
        x[mm & (ca == cb)] = (255, 255, 255)
        x[mm & (ca != cb)] = (40, 40, 220)
        x = cv2.resize(x, (w, 70), interpolation=cv2.INTER_NEAREST)
        gap = np.full((6, w, 3), 245, np.uint8)
        canvas = np.vstack([eyes, gap, sa, gap, sb, gap, x])
        for txt, y in (("A", 22), ("B", 22)):
            cv2.putText(canvas, txt, (8 if txt == "A" else w // 2 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y0 = eyes.shape[0] + 6
        for i, txt in enumerate(("A: normalized", f"B: shifted {m['shift_deg']:+.1f} deg", "A xor B (red - different bits)")):
            cv2.putText(canvas, txt, (6, y0 + i * 76 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, txt, (6, y0 + i * 76 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        details = [["Расстояние Хэмминга (HD)", f"{m['hd']:.3f}"], ["Сходство = 1 - HD", f"{m['score']:.3f}"],
                   ["Компенсация поворота", f"{m['shift_deg']:+.1f}°"], ["Сравнивалось бит", f"{m['valid'] * 100:.0f} % от кода"],
                   ["Локализация", LOCALIZERS[ra.method]], ["Кодировщик", ENCODERS[p.encoder]]]
        return {"similarity": round(m["score"], 4), "n_a": 0, "n_b": 0, "matched": 0, "details": details, "overlay": to_data_url(canvas),
                "context": {"modality": "iris", "hd": round(m["hd"], 4), "valid": round(m["valid"], 3), "shift_deg": m["shift_deg"],
                            "coverage_a": round(float(ra.mask.mean()), 3), "coverage_b": round(float(rb.mask.mean()), 3),
                            "localizer": ra.method, "encoder": p.encoder, "extra_noise": p.extra_noise}}

    def _compare_cnn(self, ra: IrisResult, rb: IrisResult, p: IrisParams) -> dict[str, Any]:
        m = registry.require("iris", "радужки", p.source)
        cos = float(ra.emb @ rb.emb)
        eyes = np.hstack([cv2.resize(_draw(_bgr(r.gray), r.circles), (320, 240)) for r in (ra, rb)])
        crops = np.hstack([cv2.resize(_bgr(iris_crop(r.gray, r.circles)), (320, 240), interpolation=cv2.INTER_CUBIC) for r in (ra, rb)])
        canvas = np.vstack([eyes, crops])
        cv2.putText(canvas, "A", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "B", (328, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "network input", (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        details = [["Косинусное сходство embedding", f"{cos:.3f}"], ["Размерность embedding", len(ra.emb)], ["Локализация", LOCALIZERS[ra.method]],
                   ["Сеть", f"{m.epochs} эпох, точность на новых снимках {m.test_acc * 100:.0f} %"],
                   ["Рекомендуемый порог", f"{m.threshold:.2f} (по EER на снимках, которых сеть не видела)"]]
        return {"similarity": round(max(0.0, cos), 4), "n_a": 0, "n_b": 0, "matched": 0, "details": details, "overlay": to_data_url(canvas),
                "context": {"modality": "iris", "encoder": "cnn", "hd": None, "valid": 1.0, "shift_deg": 0, "localizer": ra.method,
                            "coverage_a": round(float(ra.mask.mean()), 3), "coverage_b": round(float(rb.mask.mean()), 3), "extra_noise": p.extra_noise}}

    # ---------- метрики ----------
    def _eval_layout(self, p: IrisParams, n_subjects: int):
        """Какие снимки участвуют в оценке: (глаза, идентификаторы (глаз, снимок) -> образец, описание набора)."""
        n_imp = synthetic.N_IMPRESSIONS
        if p.source == SYNTHETIC:
            n = max(3, min(n_subjects, synthetic.N_EYES))
            ids = {(e, i): f"e{e:02d}_i{i}" for e in range(1, n + 1) for i in range(1, n_imp + 1)}
            return list(range(1, n + 1)), ids, f"синтетический набор: {n} глаз × {n_imp} снимка"
        if not real.is_real_source(p.source):
            raise ValueError(f"Неизвестный набор данных «{p.source}».")
        eyes = real.eye_images(n_imp, max(3, min(n_subjects, 40)))
        if len(eyes) < 3:
            raise ValueError("В CASIA меньше трёх глаз с тремя и более снимками.")
        ids = {(e, i + 1): sid for e, sids in eyes.items() for i, sid in enumerate(sids)}
        return list(eyes), ids, f"{real.TITLE}: {len(eyes)} глаз × {n_imp} снимка"

    def _curves(self, genuine, impostor, res, ok, desc, p, note=None) -> dict[str, Any]:
        th = np.round(np.arange(0, 1.0001, 0.01), 2)
        g, im = np.array(genuine), np.array(impostor)
        frr = [float((g < t).mean()) for t in th]
        far = [float((im >= t).mean()) for t in th]
        k = int(np.argmin(np.abs(np.array(far) - np.array(frr))))
        out = {"thresholds": th.tolist(), "far": far, "frr": frr, "eer": round((far[k] + frr[k]) / 2, 4), "eer_threshold": float(th[k]),
               "genuine_scores": [round(float(s), 4) for s in genuine], "impostor_scores": [round(float(s), 4) for s in impostor],
               "n_genuine": len(genuine), "n_impostor": len(impostor), "skipped_images": len(res) - len(ok),
               "source": p.source, "source_title": desc, "real": p.source != SYNTHETIC}
        if note:
            out["note"] = note
        return out

    def evaluate(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        p = IrisParams(**params)
        eyes, ids, desc = self._eval_layout(p, n_subjects)
        if p.encoder == "cnn":
            return self._evaluate_cnn(p, eyes, ids, desc)
        imps = range(1, synthetic.N_IMPRESSIONS + 1)
        res = {k: self._run(v, p, with_images=False) for k, v in ids.items()}
        ok = {k: r for k, r in res.items() if r.ok}
        sc = lambda a, b: match(ok[a].tpl, ok[b].tpl, p.max_shift)["score"]
        genuine = [sc((e, a), (e, b)) for e in eyes for a, b in itertools.combinations(imps, 2) if (e, a) in ok and (e, b) in ok]
        impostor = [sc((e1, k), (e2, k)) for e1, e2 in itertools.combinations(eyes, 2) for k in (1, 2) if (e1, k) in ok and (e2, k) in ok]
        if not genuine or not impostor:
            raise ValueError("Недостаточно пар: на многих снимках радужка не найдена. Смените метод локализации или параметры.")
        return self._curves(genuine, impostor, res, ok, desc, p)

    def _evaluate_cnn(self, p: IrisParams, eyes, ids, desc) -> dict[str, Any]:
        """Сеть видела часть снимков при обучении, поэтому считаются только пары со снимком, оставленным для проверки."""
        mdl = registry.require("iris", "радужки", p.source)
        h = mdl.holdout
        res = {k: self._run(v, p, with_images=False) for k, v in ids.items()}
        ok = {k: r for k, r in res.items() if r.ok}
        others = [i for i in range(1, synthetic.N_IMPRESSIONS + 1) if i != h]
        sc = lambda a, b: max(0.0, float(ok[a].emb @ ok[b].emb))
        genuine = [sc((e, h), (e, i)) for e in eyes for i in others if (e, h) in ok and (e, i) in ok]
        impostor = [sc((e, h), (g, i)) for e in eyes for g in eyes if g != e for i in others if (e, h) in ok and (g, i) in ok]
        if not genuine or not impostor:
            raise ValueError("Недостаточно пар: на многих снимках радужка не найдена.")
        return self._curves(genuine, impostor, res, ok, desc, p, f"Оценка по парам, где участвует снимок {h}: сеть не видела его при обучении.")

    # ---------- сравнение методов ----------
    def compare_detectors(self, sample_id: str, params: dict) -> dict[str, Any]:
        """Все методы локализации на одном снимке: окружности разных цветов, время и ошибка относительно эталона."""
        p = IrisParams(**params)
        gray = self._gray(sample_id, p)
        pre = preprocess(gray, p)
        truth = self._truth(sample_id)
        vis, rows = _bgr(gray), []
        for k, (name, title) in enumerate(LOCALIZERS.items()):
            col = COLORS[k]
            t = time.perf_counter()
            c = localize(pre, p, name)
            row = {"n": k + 1, "name": name, "title": title, "available": True, "color": "#%02x%02x%02x" % (col[2], col[1], col[0]),
                   "time_ms": round((time.perf_counter() - t) * 1000, 1), "found": c is not None}
            if c is not None:
                row["pupil"] = ", ".join(f"{v:.1f}" for v in c.pupil)
                row["iris"] = ", ".join(f"{v:.1f}" for v in c.iris)
                _draw(vis, c, (col, col), 1)
                cv2.putText(vis, str(k + 1), (int(c.iris[0] + c.iris[2] * 0.72) + 2, int(c.iris[1] - c.iris[2] * 0.72) - 4 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                if truth:
                    e1 = np.hypot(c.pupil[0] - truth["pupil"][0], c.pupil[1] - truth["pupil"][1]) + abs(c.pupil[2] - truth["pupil"][2])
                    e2 = np.hypot(c.iris[0] - truth["iris"][0], c.iris[1] - truth["iris"][1]) + abs(c.iris[2] - truth["iris"][2])
                    row["error"] = f"зрачок {e1:.1f} px, радужка {e2:.1f} px"
            rows.append(row)
        columns = [["n", "№"], ["title", "Метод"], ["found", "Найдено"], ["time_ms", "Время, мс"], ["pupil", "Зрачок (x, y, r)"],
                   ["iris", "Радужка (x, y, R)"], ["error", "Ошибка относительно эталона"]]
        return {"sample_id": sample_id, "rows": rows, "columns": columns, "overlay": to_data_url(cv2.resize(vis, None, fx=2, fy=2))}

    def compare_embedders(self, sample_a: str, sample_b: str, params: dict) -> dict[str, Any]:
        """Одна и та же пара развёрток, разные кодировщики: длина кода, сходство, расстояние Хэмминга, сдвиг."""
        p = IrisParams(**params)
        ra, rb = self._run(sample_a, p, False), self._run(sample_b, p, False)
        for lab, r in (("A", ra), ("B", rb)):
            if not r.ok:
                raise ValueError(f"На образце {lab} не удалось найти зрачок и радужку.")
        rows = []
        for name, title in ENCODERS.items():
            t = time.perf_counter()
            if name == "cnn":
                m = registry.get("iris")
                if m is None or m.dataset != p.source:
                    rows.append({"name": name, "title": title, "available": False,
                                 "note": ("Сеть для радужки ещё не обучена: откройте вкладку «Обучение распознаванию»." if m is None else
                                          f"Сеть обучена на наборе «{m.dataset}», а выбран «{p.source}»: обучите её на выбранном наборе.")})
                    continue
                cos = float(registry.embed("iris", iris_crop(ra.gray, ra.circles)) @ registry.embed("iris", iris_crop(rb.gray, rb.circles)))
                rows.append({"name": name, "title": title, "available": True, "dim": 32, "similarity": round(max(0.0, cos), 4), "hd": "-",
                             "shift": "-", "valid": "100 %", "time_ms": round((time.perf_counter() - t) * 1000, 1)})
                continue
            ta, tb = encode(ra.strip, ra.mask, name, p), encode(rb.strip, rb.mask, name, p)
            m = match(ta, tb, p.max_shift)
            rows.append({"name": name, "title": title, "available": True, "dim": int(ta.code.size), "similarity": round(m["score"], 4),
                         "hd": round(m["hd"], 4), "shift": f"{m['shift_deg']:+.1f}°", "valid": f"{m['valid'] * 100:.0f} %",
                         "time_ms": round((time.perf_counter() - t) * 1000, 1)})
        columns = [["title", "Кодировщик"], ["dim", "Длина кода, бит"], ["similarity", "Сходство = 1 - HD"], ["hd", "HD"],
                   ["shift", "Компенсация поворота"], ["valid", "Сравнивалось бит"], ["time_ms", "Время, мс"]]
        up = lambda r: to_data_url(_strip_img(r.strip, r.mask, 1))
        return {"rows": rows, "columns": columns, "detector": LOCALIZERS[ra.method], "aligned_a": up(ra), "aligned_b": up(rb)}

    def evaluate_embedders(self, params: dict, n_subjects: int = 12) -> dict[str, Any]:
        rows = []
        for name, title in ENCODERS.items():
            t = time.perf_counter()
            try:
                ev = self.evaluate({**params, "encoder": name}, n_subjects)
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
        return {"rows": rows, "detector": LOCALIZERS.get(IrisParams(**params).localizer, "авто (Даугман, затем другие)")}

    def info(self) -> dict[str, Any]:
        sources = [{"id": SYNTHETIC, "title": SYNTHETIC_TITLE, "real": False, "images": synthetic.N_EYES * synthetic.N_IMPRESSIONS,
                    "note": "Датасет синтетический: реальные биометрические данные людей не используются."}] + real.sources_info()
        return {"localizers": {k: True for k in LOCALIZERS}, "encoders": {k: True for k in ENCODERS}, "eyes": synthetic.N_EYES, "hint": "",
                "sources": sources, "default_source": SYNTHETIC}
