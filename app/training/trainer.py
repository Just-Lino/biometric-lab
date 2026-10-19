"""Фоновое обучение CNN с записью «снимка» состояния на каждой эпохе:
метрики, фильтры, карты признаков, проекция embeddings, распределения «свои/чужие», предсказания на тесте."""
import base64
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.imaging import to_data_url
from app.training.cnn import CNN
from app.training.datasets import TrainData, augment, load_dataset, prep


@dataclass
class TrainConfig:
    modality: str = "fingerprint"
    epochs: int = 30
    lr: float = 0.003
    batch: int = 32
    aug: float = 1.0
    copies: int = 8
    n_subjects: int = 12


def _eer(gen: np.ndarray, imp: np.ndarray) -> float:
    if len(gen) == 0 or len(imp) == 0:
        return float("nan")
    th = np.linspace(0, 1, 101)
    far = np.array([(imp >= t).mean() for t in th]); frr = np.array([(gen < t).mean() for t in th])
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2)


def _b64_channels(arr: np.ndarray) -> dict:
    """(C,h,w) -> uint8 с нормировкой каждого канала отдельно."""
    C, h, w = arr.shape
    out = np.zeros_like(arr, dtype=np.uint8)
    for c in range(C):
        a = arr[c]; lo, hi = float(a.min()), float(a.max())
        out[c] = ((a - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    return {"c": C, "h": h, "w": w, "data": base64.b64encode(out.tobytes()).decode()}


def _conv_links(Wt, W0, cin):
    """Средняя |вес| по каждой паре (выходной, входной канал) и относительное изменение с начала обучения."""
    a = np.abs(Wt).reshape(Wt.shape[0], cin, -1).mean(2)
    d = np.abs(Wt - W0).reshape(Wt.shape[0], cin, -1).mean(2)
    b = np.abs(W0).reshape(W0.shape[0], cin, -1).mean(2)
    return a, d / (b + 1e-9)


class Trainer:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.reset()

    def reset(self):
        self.state, self.message = "idle", ""
        self.epoch, self.total = 0, 0
        self.history = {"loss": [], "train_acc": [], "test_acc": [], "eer": [],
                        "wchange": {k: [] for k in ("conv1", "conv2", "conv3", "embedding", "classifier")}}
        self.snapshots: list[dict] = []
        self.config: dict = {}
        self.names: list[str] = []
        self.samples: list[dict] = []
        self.preview: str | None = None
        self.stop_flag = False

    # ---------- управление ----------
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg: TrainConfig) -> None:
        if self.running():
            raise ValueError("Обучение уже идёт. Остановите его или дождитесь конца.")
        with self.lock:
            self.reset()
            self.config = cfg.__dict__.copy()
            self.total = cfg.epochs
            self.state = "preparing"
        self.thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag = True

    def wait(self, timeout: float = 120) -> None:
        if self.thread:
            self.thread.join(timeout)

    def status(self) -> dict:
        with self.lock:
            return {"state": self.state, "message": self.message, "epoch": self.epoch, "total": self.total,
                    "config": self.config, "history": self.history, "names": self.names,
                    "samples": self.samples, "preview": self.preview, "snapshots": len(self.snapshots)}

    def snapshot(self, n: int) -> dict | None:
        with self.lock:
            return self.snapshots[n] if 0 <= n < len(self.snapshots) else None

    # ---------- обучение ----------
    def _run(self, cfg: TrainConfig) -> None:
        try:
            data = load_dataset(cfg.modality, cfg.n_subjects)
            S, ncls = data.size, len(data.names)
            Xtr = np.stack([prep(i, S) for i in data.train])[:, None]
            Xte = np.stack([prep(i, S) for i in data.test])[:, None]
            net = CNN(S, ncls, 32, seed=1)
            self._w0 = [layer.W.copy() for layer in net.layers]  # начальные веса: сравниваем с ними
            rng = np.random.RandomState(0)
            vrng = np.random.RandomState(123)
            aug_viz = [(augment(im, vrng, 1.0, data), int(y)) for im, y in zip(data.train, data.ytrain) for _ in range(2)]
            Xviz = np.concatenate([Xtr, Xte, np.stack([prep(a, S) for a, _ in aug_viz])[:, None]])
            yviz = np.concatenate([data.ytrain, data.ytest, np.array([y for _, y in aug_viz])])
            split = np.array([0] * len(Xtr) + [1] * len(Xte) + [2] * len(aug_viz))
            clean = split < 2
            with self.lock:
                self.names = data.names
                self.samples = [{"name": data.names[int(y)], "img": to_data_url(cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA))}
                                for im, y in zip(data.test, data.ytest)]
                self.preview = self.samples[0]["img"] if self.samples else None
                self.state = "training"
            self._snap(0, net, data, Xtr, Xte, Xviz, yviz, split, clean, loss=float("nan"))
            for ep in range(1, cfg.epochs + 1):
                if self.stop_flag:
                    break
                X, y = [], []
                for im, lab in zip(data.train, data.ytrain):
                    for _ in range(cfg.copies):
                        X.append(prep(augment(im, rng, cfg.aug, data), S)); y.append(lab)
                X = np.stack(X)[:, None].astype(np.float32); y = np.array(y)
                idx = rng.permutation(len(X)); losses = []
                for b in range(0, len(X), cfg.batch):
                    if self.stop_flag:
                        break
                    j = idx[b:b + cfg.batch]
                    losses.append(net.step(X[j], y[j], cfg.lr)[0])
                loss = float(np.mean(losses)) if losses else float("nan")
                self._snap(ep, net, data, Xtr, Xte, Xviz, yviz, split, clean, loss)
            with self.lock:
                self.state = "stopped" if self.stop_flag else "finished"
        except Exception as e:  # ошибки данных и т.п. показываем в интерфейсе
            with self.lock:
                self.state, self.message = "error", str(e)

    def _snap(self, ep, net, data: TrainData, Xtr, Xte, Xviz, yviz, split, clean, loss):
        ytr, yte = data.ytrain, data.ytest
        ptr = net.forward(Xtr)[0].argmax(1); tr_acc = float((ptr == ytr).mean())
        lg = net.forward(Xte)[0]
        z = lg - lg.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
        te_pred = pr.argmax(1); te_acc = float((te_pred == yte).mean())
        emb = net.embed(Xviz)
        # проекция embeddings на 2 главные компоненты (знак фиксируем, чтобы картинка не «переворачивалась»)
        c = emb - emb.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        comps = Vt[:2].copy()
        for k in range(len(comps)):
            if comps[k][np.argmax(np.abs(comps[k]))] < 0:
                comps[k] *= -1
        proj = c @ comps.T
        if proj.shape[1] < 2:
            proj = np.pad(proj, ((0, 0), (0, 2 - proj.shape[1])))
        # «свои / чужие» по косинусному сходству чистых образцов
        E = emb[clean]; yc = yviz[clean]
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sim = En @ En.T
        iu = np.triu_indices(len(E), 1)
        same = (yc[:, None] == yc[None, :])[iu]; s = sim[iu]
        gen, imp = s[same], s[~same]
        bins = np.linspace(0, 1, 21)
        hg, _ = np.histogram(np.clip(gen, 0, 1), bins); hi, _ = np.histogram(np.clip(imp, 0, 1), bins)
        eer = _eer(gen, imp)
        lg1, e1 = net.forward(Xte[:1], keep=True)
        z1 = lg1[0] - lg1[0].max(); p1 = np.exp(z1); p1 /= p1.sum()
        maps = {k: _b64_channels(v[0]) for k, v in net.maps.items()}
        # связи между слоями для схемы сети: толщина = средний |вес|, цвет = насколько связь изменилась с начала
        c1, c2, c3, d1, d2 = net.layers
        w0 = self._w0
        l1, ch1 = _conv_links(c1.W, w0[0], 1)
        l2, ch2 = _conv_links(c2.W, w0[1], c1.cout)
        l3, ch3 = _conv_links(c3.W, w0[2], c2.cout)
        ch_n = c3.cout
        emb_l = np.abs(d1.W).reshape(ch_n, -1, d1.W.shape[1]).mean((1, 2))
        emb_c = (np.abs(d1.W - w0[3]).reshape(ch_n, -1, d1.W.shape[1]).mean((1, 2)) /
                 (np.abs(w0[3]).reshape(ch_n, -1, d1.W.shape[1]).mean((1, 2)) + 1e-9))
        cls_l = np.abs(d2.W).mean(0); cls_s = d2.W.mean(0)
        cls_c = np.abs(d2.W - w0[4]).mean(0) / (np.abs(w0[4]).mean(0) + 1e-9)
        node_change = {nm: (np.linalg.norm(l.W - w, axis=1) / (np.linalg.norm(w, axis=1) + 1e-9))
                       for nm, l, w in (("conv1", c1, w0[0]), ("conv2", c2, w0[1]), ("conv3", c3, w0[2]))}
        r3 = lambda a: np.round(np.asarray(a, float), 3).tolist()
        netdata = {"links": {"conv1": r3(l1[:, 0]), "conv2": r3(l2), "conv3": r3(l3), "emb": r3(emb_l), "cls": r3(cls_l), "cls_sign": r3(cls_s)},
                   "link_change": {"conv1": r3(ch1[:, 0]), "conv2": r3(ch2), "conv3": r3(ch3), "emb": r3(emb_c), "cls": r3(cls_c)},
                   "node_change": {k: r3(v) for k, v in node_change.items()}}
        preview = {"label": int(data.ytest[0]), "probs": r3(p1), "emb": r3(e1[0])}
        f = net.c1.W.reshape(net.c1.cout, 3, 3)
        wchange = {nm: float(np.linalg.norm(layer.W - w0) / (np.linalg.norm(w0) + 1e-9))
                   for nm, layer, w0 in zip(("conv1", "conv2", "conv3", "embedding", "classifier"), net.layers, self._w0)}
        snap = {
            "epoch": ep, "loss": None if not np.isfinite(loss) else round(loss, 4), "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4), "eer": None if not np.isfinite(eer) else round(eer, 4),
            "filters": np.round(f, 3).tolist(), "weight_change": {k: round(v, 4) for k, v in wchange.items()}, "maps": maps, "net": netdata, "preview": preview,
            "embedding": [[round(float(p[0]), 3), round(float(p[1]), 3), int(y), int(sp)] for p, y, sp in zip(proj, yviz, split)],
            "hist": {"bins": np.round(bins, 2).tolist(), "genuine": hg.tolist(), "impostor": hi.tolist()},
            "test_preds": [{"pred": int(p), "conf": round(float(pr[i, p]), 3), "ok": bool(p == yte[i])} for i, p in enumerate(te_pred)],
        }
        with self.lock:
            self.snapshots.append(snap)
            self.epoch = ep
            self.history["loss"].append(snap["loss"]); self.history["train_acc"].append(snap["train_acc"])
            self.history["test_acc"].append(snap["test_acc"]); self.history["eer"].append(snap["eer"])
            for k, v in snap["weight_change"].items():
                self.history["wchange"][k].append(v)


trainer = Trainer()
