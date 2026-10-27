import shutil

from fastapi.testclient import TestClient

import app.main as main
from app.main import CDN_LIBS, app


def test_page_uses_cdn_until_vendor_files_exist(tmp_path, monkeypatch):
    web = tmp_path / "web"  # временная копия страницы: настоящая папка app/web/vendor (офлайн-библиотеки) не трогается
    web.mkdir()
    shutil.copy(main.WEB / "index.html", web / "index.html")
    monkeypatch.setattr(main, "WEB", web)
    with TestClient(app) as c:
        html = c.get("/").text
        assert all(url in html for url in CDN_LIBS) and "defer" in html  # библиотеки не блокируют показ страницы
        (web / "vendor").mkdir()
        (web / "vendor" / "plotly.min.js").write_text("//test")
        html = c.get("/").text
        assert "/vendor/plotly.min.js" in html and "cdn.plot.ly" not in html and "three.min.js" in html


def test_large_json_is_gzipped():
    with TestClient(app) as c:
        r = c.post("/api/v1/biometrics/fingerprint/process", json={"sample_id": "f01_i1"}, headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"
        assert r.json()["steps"]  # клиент распаковывает прозрачно
