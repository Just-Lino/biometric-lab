"""Оптимизации (быстрый фильтр Габора, пулинг) и сеть, обученная студентом, как метод распознавания."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.fingerprint.filters import bank_responses, gabor_bank
from app.modules.fingerprint.module import FingerprintModule
from app.modules.iris.module import IrisModule
from app.training import registry
from app.training.cnn import CNN, _pool
from app.training.trainer import Trainer, TrainConfig


@pytest.mark.parametrize("period,sigma,bins", [(9, 4, 8), (9, 4, 16), (6, 2, 8), (14, 8, 8)])
def test_fft_gabor_equals_filter2d(period, sigma, bins):
    img = np.random.RandomState(0).randn(256, 256).astype(np.float32)
    old = np.stack([cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT) for k in gabor_bank(float(period), float(sigma), bins)])
    assert np.abs(old - bank_responses(img, period, sigma, bins)).max() < 1e-5  # результат прежний, но в разы быстрее


def test_pool_index_matches_max():
    x = np.maximum(np.random.RandomState(1).randn(2, 3, 8, 8).astype(np.float32), 0)
    out, idx = _pool(x)
    assert np.array_equal(out, x.reshape(2, 3, 4, 2, 4, 2).max(axis=(3, 5))) and idx.max() <= 3


def test_cnn_gradients_and_both_optimizers():
    rng = np.random.RandomState(0)
    x, y = rng.randn(16, 1, 16, 16).astype(np.float32), rng.randint(0, 3, 16)
    for opt, lr in (("adam", 0.01), ("sgd", 0.02)):
        net = CNN(16, 3, 8, filters=(4, 4, 4), seed=0, optimizer=opt)
        first = net.step(x, y, lr)[0]
        for _ in range(60):
            last = net.step(x, y, lr)[0]
        assert last < first * 0.5, opt  # оба оптимизатора учатся
    # численная проверка градиента последнего слоя
    net = CNN(16, 3, 8, filters=(4, 4, 4), seed=0)
    lg, _ = net.forward(x)
    _, d, _ = net.loss_grad(lg, y)
    net.backward(d)
    g = net.d2.dW[0, 0]
    eps, w = 1e-3, net.d2.W
    w[0, 0] += eps
    lp = net.loss_grad(net.forward(x)[0], y)[0]
    w[0, 0] -= 2 * eps
    lm = net.loss_grad(net.forward(x)[0], y)[0]
    assert abs((lp - lm) / (2 * eps) - g) < 1e-2


def test_state_roundtrip():
    net = CNN(16, 3, 8, filters=(4, 4, 4), seed=1)
    x = np.random.RandomState(2).randn(4, 1, 16, 16).astype(np.float32)
    net2 = CNN(16, 3, 8, filters=(4, 4, 4), seed=9)
    net2.load_state(net.state())
    assert np.allclose(net.forward(x)[0], net2.forward(x)[0]) and net.n_params() == net2.n_params()


def test_cnn_methods_require_training():
    for mod, params in ((FingerprintModule(), {"matcher": "cnn"}), (IrisModule(), {"encoder": "cnn"})):
        a, b = ("f01_i1", "f01_i2") if "matcher" in params else ("e01_i1", "e01_i2")
        with pytest.raises(ValueError, match="не обучена"):
            mod.compare(a, b, params)
    assert registry.get("fingerprint") is None


def test_trained_cnn_as_fingerprint_and_iris_method():
    from app.core.db import init_db
    init_db()
    for modality, cfg in (("fingerprint", {"epochs": 10}), ("iris", {"epochs": 10, "arch": "medium"})):
        t = Trainer()
        t.start(TrainConfig(modality=modality, **cfg))
        t.wait(120)
        assert t.status()["state"] == "finished"
    info = registry.get("fingerprint").info()
    assert info["available"] and info["epochs"] == 10 and 0 <= info["threshold"] <= 1
    registry.reset_cache()  # после перезапуска сервера модель читается с диска
    assert registry.get("fingerprint").epochs == 10 and registry.get("iris").config["arch"] == "medium"
    fp, ir = FingerprintModule(), IrisModule()
    same, other = fp.compare("f02_i1", "f02_i3", {"matcher": "cnn"}), fp.compare("f02_i1", "f05_i3", {"matcher": "cnn"})
    assert same["similarity"] > other["similarity"] and same["details"][1][0].startswith("Косинусное")
    ev = fp.evaluate({"matcher": "cnn"})
    assert ev["note"] and "не видела" in ev["note"] and ev["n_genuine"] == 20  # только пары со снимком, оставленным для проверки
    same, other = ir.compare("e02_i1", "e02_i3", {"encoder": "cnn"}), ir.compare("e02_i1", "e05_i3", {"encoder": "cnn"})
    assert same["similarity"] > other["similarity"]
    steps = [s["title"] for s in ir.process("e01_i1", {"encoder": "cnn"})["steps"]]
    assert steps[-2:] == ["7. Вход нейросети", "8. Embedding сети"]
    rows = {x["name"]: x for x in fp.compare_embedders("f01_i1", "f01_i3", {})["rows"]}
    assert rows["cnn"]["available"] and rows["cnn"]["threshold"] == info["threshold"]


def test_training_options_journal_and_model_api():
    with TestClient(app) as c:
        assert c.post("/api/v1/training/start", json={"arch": "huge"}).status_code == 422
        assert c.post("/api/v1/training/start", json={"optimizer": "rmsprop"}).status_code == 422
        assert c.post("/api/v1/training/start", json={"holdout": 4}).status_code == 422
        c.delete("/api/v1/training/runs")  # журнал общий для всех тестов сессии
        assert c.get("/api/v1/training/model/fingerprint").json() == {"available": False}
        assert c.get("/api/v1/training/model/retina").status_code == 404
        r = c.post("/api/v1/training/start", json={"epochs": 4, "arch": "large", "optimizer": "sgd", "lr": 0.01, "holdout": 2})
        assert r.status_code == 200
        from app.training.trainer import trainer
        trainer.wait(120)
        st = c.get("/api/v1/training/status").json()
        assert st["state"] == "finished" and st["config"]["arch"] == "large" and st["config"]["holdout"] == 2
        runs = c.get("/api/v1/training/runs").json()
        assert len(runs) == 1 and runs[0]["config"]["optimizer"] == "sgd" and len(runs[0]["curves"]["loss"]) == 5
        assert runs[0]["summary"]["params"] == 80012 and runs[0]["summary"]["seconds"] > 0
        m = c.get("/api/v1/training/model/fingerprint").json()
        assert m["available"] and m["holdout"] == 2 and m["arch"] == "large"
        assert c.delete("/api/v1/training/runs").status_code == 200 and c.get("/api/v1/training/runs").json() == []


def test_light_status_omits_static_images():
    t = Trainer()
    t.start(TrainConfig(modality="fingerprint", epochs=2))
    t.wait(60)
    full, light = t.status(), t.status(light=True)
    assert full["samples"] and not light["samples"] and light["preview"] is None
    assert light["epoch"] == full["epoch"] == 2 and light["history"] == full["history"]
    import json
    assert len(json.dumps(light)) < len(json.dumps(full)) / 3
