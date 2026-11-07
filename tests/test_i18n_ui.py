"""Переключатель языка в настоящем браузере (Playwright + Edge/Chrome): страница на каждом языке открывается без ошибок JS,
панель помощника создаётся, английский интерфейс без русских слов, выбор языка запоминается.

Пропускается, если нет Playwright или браузера."""
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CYR = re.compile(r"[А-Яа-яЁё]")
NODES = """() => { const out=[], w=document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
  while((n=w.nextNode())){ const p=n.parentNode.nodeName; if(p==='SCRIPT'||p==='STYLE') continue; const v=n.nodeValue.trim(); if(v) out.push(v); } return out; }"""
SWITCHER_NAMES = {"Русский язык", "Қазақ тілі", "ҚАЗ", "English", "РУС", "ENG"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = dict(os.environ, DB_PATH=os.path.join(tempfile.mkdtemp(), "i18n_ui.db"), TUTOR_PROVIDER="rule", AUTO_DOWNLOAD_MODELS="false", PYTHONUTF8="1")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        p.kill()
        pytest.skip("сервер не запустился")
    yield f"http://127.0.0.1:{port}"
    p.terminate()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(channel="msedge", headless=True)
        except Exception:
            try:
                b = pw.chromium.launch(channel="chrome", headless=True)
            except Exception:
                pytest.skip("нет Edge/Chrome для Playwright")
        yield b
        b.close()


def open_page(browser, server, lang, path="/"):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_cookies([{"name": "lab_lang", "value": lang, "url": server}])
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)[:300]))
    page.goto(server + path, wait_until="load", timeout=90000)
    page.wait_for_timeout(1500)
    return ctx, page, errors


@pytest.mark.parametrize("lang", ["ru", "kk", "en"])
def test_main_page_works_in_every_language(browser, server, lang):
    ctx, page, errors = open_page(browser, server, lang)
    try:
        assert page.evaluate("document.documentElement.lang") == lang
        assert page.evaluate("window.LAB_LANG") == lang
        page.click(".tab[data-mod='fingerprint']")
        page.click("#as-fab") if page.query_selector("#as-fab") else None
        page.wait_for_selector("#as-send", state="attached", timeout=10000)      # панель помощника построена на любом языке
        page.click(".tab[data-mod='voice']")
        page.wait_for_timeout(500)
        assert not errors, errors
    finally:
        ctx.close()


def test_english_interface_has_no_russian_words(browser, server):
    ctx, page, errors = open_page(browser, server, "en")
    try:
        for mod in ("fingerprint", "face", "iris", "voice", "code", "train", "nn"):
            page.click(f".tab[data-mod='{mod}']")
            page.wait_for_timeout(1500)
            if mod in ("fingerprint", "face", "iris", "voice"):
                page.click("#bproc")
                page.wait_for_timeout(3000)
        left = [t for t in page.evaluate(NODES) if CYR.search(t) and t not in SWITCHER_NAMES]
        assert not left, left[:8]
        assert not errors, errors
    finally:
        ctx.close()


def test_choosing_language_sets_cookie_and_reloads(browser, server):
    ctx, page, errors = open_page(browser, server, "ru")
    try:
        page.evaluate("LABI18N.setLang('en')")
        page.wait_for_load_state("load")
        page.wait_for_timeout(1500)
        assert page.evaluate("window.LAB_LANG") == "en"
        assert any(c["name"] == "lab_lang" and c["value"] == "en" for c in ctx.cookies())
        assert page.evaluate("document.title") != "Виртуальная лаборатория биометрии"
    finally:
        ctx.close()


@pytest.mark.parametrize("path", ["/labplus/voice", "/labplus/metrics", "/labplus/robustness", "/labplus/standards"])
def test_labplus_pages_in_english(browser, server, path):
    ctx, page, errors = open_page(browser, server, "en", path)
    try:
        left = [t for t in page.evaluate(NODES) if CYR.search(t) and t not in SWITCHER_NAMES]
        assert not left, left[:8]
        assert not errors, errors
    finally:
        ctx.close()
