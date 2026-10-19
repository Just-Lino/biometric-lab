import base64
import hashlib
import itertools
import uuid
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np

from app.core.imaging import to_data_url, to_gray
from app.modules.base import BiometricModule
from app.modules.fingerprint import synthetic
from app.modules.fingerprint.matching import draw_match, match_minutiae
from app.modules.fingerprint.params import FingerprintParams
from app.modules.fingerprint.pipeline import run_pipeline

N_FINGERS = 12
N_IMPRESSIONS = 3
MAX_UPLOADS = 20


class FingerprintModule(BiometricModule):
    name = "fingerprint"
    title = "Распознавание отпечатков пальцев"

    def __init__(self):
        self._uploads: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache: OrderedDict[str, Any] = OrderedDict()

    # ---------- образцы ----------
    def list_samples(self) -> list[dict[str, Any]]:
        items = [{"id": f"f{f:02d}_i{i}", "label": f"Палец {f:02d}, оттиск {i}", "finger": f, "impression": i}
                 for f in range(1, N_FINGERS + 1) for i in range(1, N_IMPRESSIONS + 1)]
        items += [{"id": uid, "label": f"Загружено: {uid}", "finger": None, "impression": None}
                  for uid in self._uploads]
        return items

    def _gray(self, sample_id: str, p: FingerprintParams) -> np.ndarray:
        if sample_id in self._uploads:
            img = self._uploads[sample_id].copy()
        else:
            try:
                f, i = sample_id.split("_")
                img = synthetic.impression(int(f[1:]), int(i[1:])).copy()
            except (ValueError, KeyError):
                raise KeyError(f"Неизвестный образец: {sample_id}")
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
        uid = "up_" + uuid.uuid4().hex[:6]
        self._uploads[uid] = canvas
        while len(self._uploads) > MAX_UPLOADS:
            self._uploads.popitem(last=False)
        return uid

    # ---------- обработка ----------
    def _run(self, sample_id: str, p: FingerprintParams, with_images=True):
        key = f"{sample_id}|{p.model_dump_json(exclude={'dist_tol', 'angle_tol_deg', 'max_rotation_deg'})}|{with_images}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        res = run_pipeline(self._gray(sample_id, p), p, with_images)
        self._cache[key] = res
        while len(self._cache) > 400:
            self._cache.popitem(last=False)
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
        m = match_minutiae(ra.minutiae, rb.minutiae, dist_tol=p.dist_tol,
                           angle_tol_deg=p.angle_tol_deg, max_rotation_deg=p.max_rotation_deg)
        overlay = draw_match(ra.gray, rb.gray, ra.minutiae, rb.minutiae, m)
        return {"similarity": round(m.score, 4), "n_a": len(ra.minutiae), "n_b": len(rb.minutiae),
                "matched": len(m.matched), "rotation_deg": round(m.rotation_deg, 1),
                "shift": [round(v, 1) for v in m.shift], "overlay": to_data_url(overlay),
                "details": [["Minutiae A", len(ra.minutiae)], ["Minutiae B", len(rb.minutiae)], ["Совпало", len(m.matched)],
                            ["Поворот", f"{m.rotation_deg:.1f}°"], ["Сдвиг", f"{m.shift[0]:.1f}, {m.shift[1]:.1f} px"]],
                "context": {"modality": "fingerprint", "extra_noise": p.extra_noise}}

    # ---------- метрики ----------
    def evaluate(self, params: dict, n_subjects: int = 10) -> dict[str, Any]:
        p = FingerprintParams(**params)
        n = max(3, min(n_subjects, N_FINGERS))
        ids = {(f, i): f"f{f:02d}_i{i}" for f in range(1, n + 1) for i in range(1, N_IMPRESSIONS + 1)}
        mins = {k: self._run(v, p, with_images=False).minutiae for k, v in ids.items()}

        def score(k1, k2):
            return match_minutiae(mins[k1], mins[k2], dist_tol=p.dist_tol, angle_tol_deg=p.angle_tol_deg,
                                  max_rotation_deg=p.max_rotation_deg).score

        genuine = [score((f, a), (f, b)) for f in range(1, n + 1)
                   for a, b in itertools.combinations(range(1, N_IMPRESSIONS + 1), 2)]
        impostor = [score((f, i), (g, i)) for f, g in itertools.combinations(range(1, n + 1), 2)
                    for i in range(1, N_IMPRESSIONS + 1)]
        th = np.round(np.arange(0, 1.0001, 0.01), 2)
        g, im = np.array(genuine), np.array(impostor)
        frr = [float((g < t).mean()) for t in th]
        far = [float((im >= t).mean()) for t in th]
        k = int(np.argmin(np.abs(np.array(far) - np.array(frr))))
        return {"thresholds": th.tolist(), "far": far, "frr": frr,
                "eer": round((far[k] + frr[k]) / 2, 4), "eer_threshold": float(th[k]),
                "genuine_scores": [round(float(s), 4) for s in genuine],
                "impostor_scores": [round(float(s), 4) for s in impostor],
                "n_genuine": len(genuine), "n_impostor": len(impostor)}

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
