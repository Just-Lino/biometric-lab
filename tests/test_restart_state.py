"""Состояние в памяти процесса: после перезапуска сервера студент получает понятное сообщение, а не ошибку 500."""
import io
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.modules.registry import get_module  # noqa: E402
from app.modules.uploads import UPLOAD_ID, UploadStore  # noqa: E402
from app.modules.voice import audio, synthetic  # noqa: E402
from app.training import registry  # noqa: E402

BASE = "/api/v1/biometrics"


def png_bytes() -> bytes:
    rng = np.random.RandomState(1)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(160, 160) * 255).astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def upload(client, modality):
    if modality == "voice":
        data, name, mime = audio.write_wav(synthetic.recording(1, 1)), "v.wav", "audio/wav"
    else:
        data, name, mime = png_bytes(), "x.png", "image/png"
    r = client.post(f"{BASE}/{modality}/upload", files={"file": (name, data, mime)})
    assert r.status_code == 200, r.text
    return r.json()["sample_id"]


def restart(modality):
    """Имитация перезапуска: хранилище загрузок модуля создаётся пустым, как в новом процессе."""
    get_module(modality)._uploads = UploadStore()


def lost_message(r):
    assert r.status_code == 404, (r.status_code, r.text[:200])
    d = r.json()["detail"]
    assert "перезапущен" in d and "Загрузите файл заново" in d
    return d


@pytest.mark.parametrize("modality", ["fingerprint", "face", "iris", "voice"])
def test_lost_upload_gets_clear_message_everywhere(modality):
    with TestClient(app) as c:
        uid = upload(c, modality)
        assert UPLOAD_ID.fullmatch(uid)
        assert c.get(f"{BASE}/{modality}/sample/{uid}.png").status_code == 200       # до перезапуска всё работает
        restart(modality)
        other = "f01_i1" if modality == "fingerprint" else ("e01_i1" if modality == "iris" else ("v01_r1" if modality == "voice" else uid))
        calls = [c.post(f"{BASE}/{modality}/process", json={"sample_id": uid}),
                 c.post(f"{BASE}/{modality}/compare", json={"sample_a": uid, "sample_b": other}),
                 c.post(f"{BASE}/{modality}/compare", json={"sample_a": other, "sample_b": uid}),
                 c.get(f"{BASE}/{modality}/sample/{uid}.png")]
        for r in calls:
            lost_message(r)
        assert uid not in [s["id"] for s in c.get(f"{BASE}/{modality}/samples").json()]   # из списка образцов загрузка тоже исчезла


def test_upload_of_another_session_is_not_visible_and_gives_same_message():
    with TestClient(app) as a, TestClient(app) as b:                     # разные cookie lab_sid
        uid = upload(a, "fingerprint")
        r = b.post(f"{BASE}/fingerprint/process", json={"sample_id": uid})
        assert lost_message(r) == lost_message(a.post(f"{BASE}/fingerprint/process", json={"sample_id": "up_ffffff"}))
        assert a.post(f"{BASE}/fingerprint/process", json={"sample_id": uid}).status_code == 200   # владельцу доступна


def test_unknown_gallery_sample_has_clean_message_without_quotes():
    with TestClient(app) as c:
        r = c.post(f"{BASE}/face/process", json={"sample_id": "nobody__none_0001"})
        assert r.status_code == 404
        d = r.json()["detail"]
        assert "Неизвестный образец" in d and not d.startswith(("'", '"'))


def test_lost_trained_network_is_reported_clearly_not_as_500():
    registry.reset_cache()
    with TestClient(app) as c:
        r = c.post(f"{BASE}/fingerprint/compare", json={"sample_a": "f01_i1", "sample_b": "f01_i2", "params": {"matcher": "cnn"}})
        assert r.status_code == 400
        d = r.json()["detail"]
        assert "не обучена" in d and "перезапуске" in d and "Обучение распознаванию" in d
        assert c.get("/api/v1/training/model/fingerprint").json()["available"] is False


def test_corrupted_network_file_counts_as_missing():
    registry.reset_cache()
    d = registry._dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "fingerprint.npz").write_bytes(b"not an npz archive")
    with TestClient(app) as c:
        r = c.post(f"{BASE}/fingerprint/compare", json={"sample_a": "f01_i1", "sample_b": "f01_i2", "params": {"matcher": "cnn"}})
        assert r.status_code == 400 and "не обучена" in r.json()["detail"]
        assert c.get("/api/v1/training/model/fingerprint").json()["available"] is False


def test_training_state_after_restart_is_idle_and_old_snapshot_has_clear_message():
    with TestClient(app) as c:
        st = c.get("/api/v1/training/status").json()
        assert st["state"] in ("idle", "finished", "stopped", "error")  # процесс не падает, состояние читается
        r = c.get("/api/v1/training/snapshot/999")
        assert r.status_code == 404 and "перезапущен" in r.json()["detail"] and "Запустите обучение заново" in r.json()["detail"]
        assert c.post("/api/v1/training/stop").json() == {"ok": True}  # остановка без запущенного обучения безвредна


def test_upload_ids_generated_by_stores_match_the_pattern():
    s = UploadStore()
    assert all(UPLOAD_ID.fullmatch(s.add(np.zeros((2, 2), np.uint8))) for _ in range(20))
    assert settings.db_path  # настройки доступны, тест не зависит от файла .env
