"""Cookie сессии (флаг Secure), защитные заголовки и границы загрузок."""
import io
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.api.v1 import biometrics  # noqa: E402


def cookie(r):
    return r.headers.get("set-cookie", "")


def test_cookie_without_secure_by_default(monkeypatch):
    monkeypatch.setattr(settings, "cookie_secure", False)
    with TestClient(app) as c:
        h = cookie(c.get("/health"))
    assert "lab_sid=" in h and "HttpOnly" in h and "SameSite=Lax" in h and "Secure" not in h


def test_cookie_secure_and_hsts_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "cookie_secure", True)
    with TestClient(app) as c:
        r = c.get("/health")
    assert "; Secure" in cookie(r)
    assert "max-age" in r.headers["strict-transport-security"]


def test_security_headers_and_no_iframe_block(monkeypatch):
    monkeypatch.setattr(settings, "cookie_secure", False)
    with TestClient(app) as c:
        r = c.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "microphone=(self)" in r.headers["permissions-policy"]
    assert "strict-transport-security" not in r.headers
    assert "x-frame-options" not in r.headers and "content-security-policy" not in r.headers


def test_upload_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(biometrics, "MAX_UPLOAD_BYTES", 1000)
    with TestClient(app) as c:
        r = c.post("/api/v1/biometrics/fingerprint/upload", files={"file": ("a.png", b"x" * 2000, "image/png")})
        assert r.status_code == 413
    monkeypatch.setattr(biometrics, "MAX_UPLOAD_BYTES", 5_000_000)
    monkeypatch.setattr(biometrics, "MAX_UPLOAD_PIXELS", 100)
    buf = io.BytesIO()
    Image.new("L", (50, 50)).save(buf, "PNG")
    with TestClient(app) as c:
        r = c.post("/api/v1/biometrics/fingerprint/upload", files={"file": ("a.png", buf.getvalue(), "image/png")})
        assert r.status_code == 413


def test_tutor_output_is_escaped_in_page():
    html = open("app/web/index.html", encoding="utf-8").read()
    assert "esc(r.answer)" in html and "esc(err.message)" in html
