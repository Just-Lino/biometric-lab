import os
import tempfile
import time

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "train.db"))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.modules.face.module as fm
from app.main import app
from app.training.cnn import CNN
from app.training.datasets import load_dataset
from app.training.trainer import TrainConfig, Trainer


def _loss(net, x, y):
    return net.loss_grad(net.forward(x)[0], y)[0]


def test_cnn_gradients_match_numeric():
    rng = np.random.RandomState(0)
    net = CNN(size=16, n_classes=3, emb=6, filters=(2, 3, 3), seed=2)
    for layer in net.layers:  # float64, чтобы численная проверка была точной
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
    x = rng.randn(4, 1, 16, 16)
    y = np.array([0, 1, 2, 1])
    logits, _ = net.forward(x)
    _, d, _ = net.loss_grad(logits, y)
    net.backward(d)
    checked = 0
    for layer in net.layers:
        for obj, pn, gn in layer.params:
            arr, g = getattr(obj, pn), getattr(obj, gn)
            for _ in range(6):
                idx = tuple(rng.randint(0, s) for s in arr.shape)
                old = arr[idx]
                arr[idx] = old + 1e-5; lp = _loss(net, x, y)
                arr[idx] = old - 1e-5; lm = _loss(net, x, y)
                arr[idx] = old
                num = (lp - lm) / 2e-5
                assert abs(num - g[idx]) <= 1e-3 + 2e-2 * abs(num), (pn, idx, num, g[idx])
                checked += 1
    assert checked >= 50


def test_training_improves_and_snapshots():
    t = Trainer()
    t.start(TrainConfig(epochs=6, n_subjects=4, copies=3))
    t.wait(120)
    s = t.status()
    assert s["state"] == "finished", s["message"]
    assert len(s["history"]["loss"]) == 7 and s["history"]["loss"][0] is None  # эпоха 0 + 6 эпох
    assert s["history"]["loss"][-1] < s["history"]["loss"][1]
    sn = t.snapshot(6)
    assert len(sn["filters"]) == 8 and len(sn["filters"][0]) == 3
    assert set(sn["maps"]) == {"conv1", "conv2", "conv3"} and len(sn["test_preds"]) == 4
    assert sum(sn["hist"]["genuine"]) > 0 and sum(sn["hist"]["impostor"]) > 0
    assert t.snapshot(99) is None
    assert 0 <= sn["eer"] <= 1  # EER эпохи считается по проверочным снимкам против обучающих
    assert t.snapshot(0)["weight_change"]["conv2"] == 0  # в начале веса равны начальным
    assert t.snapshot(6)["weight_change"]["embedding"] > 0
    assert len(s["history"]["wchange"]["conv1"]) == 7
    net0, net6 = t.snapshot(0)["net"], sn["net"]
    assert np.shape(net6["links"]["conv1"]) == (8,) and np.shape(net6["links"]["conv2"]) == (16, 8)
    assert np.shape(net6["links"]["conv3"]) == (16, 16) and np.shape(net6["links"]["emb"]) == (16,)
    assert np.shape(net6["links"]["cls"]) == (4,) and np.shape(net6["node_change"]["conv2"]) == (16,)
    assert np.max(net0["link_change"]["conv2"]) == 0 and np.max(net6["link_change"]["conv2"]) > 0  # на старте связи не менялись
    assert len(sn["preview"]["probs"]) == 4 and abs(sum(sn["preview"]["probs"]) - 1) < 0.02 and len(sn["preview"]["emb"]) == 32


def test_face_needs_two_people(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "GALLERY", tmp_path)
    with pytest.raises(ValueError, match="минимум 2 человека"):
        load_dataset("face", 5)


def test_training_api_flow():
    with TestClient(app) as c:
        r = c.post("/api/v1/training/start", json={"modality": "fingerprint", "epochs": 3, "n_subjects": 3, "copies": 2})
        assert r.status_code == 200
        for _ in range(100):
            st = c.get("/api/v1/training/status").json()
            if st["state"] in ("finished", "error"):
                break
            time.sleep(0.3)
        assert st["state"] == "finished" and st["epoch"] == 3 and len(st["samples"]) == 3
        assert c.get("/api/v1/training/snapshot/3").status_code == 200
        assert c.get("/api/v1/training/snapshot/50").status_code == 404
        assert c.post("/api/v1/training/start", json={"modality": "retina"}).status_code == 422
        # радужка допустима: обучение запускается (останавливаем сразу, полный прогон проверен отдельным тестом)
        r = c.post("/api/v1/training/start", json={"modality": "iris", "epochs": 10})
        assert r.status_code == 200
        c.post("/api/v1/training/stop")
        for _ in range(300):  # дожидаемся остановки: фоновый поток не должен дописывать сеть и журнал во время следующих тестов
            if c.get("/api/v1/training/status?light=true").json()["state"] not in ("preparing", "training"):
                break
            time.sleep(0.2)


def test_iris_dataset_for_training():
    from app.training.datasets import load_dataset
    d = load_dataset("iris", 4)
    assert d.modality == "iris" and len(d.names) == 4 and len(d.train) == 8 and len(d.test) == 4
    assert d.train[0].shape == (96, 96) and d.size == 64
