"""Практикум в настоящем браузере (Playwright + Edge/Chrome): пример выполняется в Pyodide и совпадает с серверным /run,
бесконечный цикл останавливается завершением воркера (страница остаётся живой), без Pyodide работает серверный запасной путь.

Тесты пропускаются, если нет Playwright, браузера или скачанных файлов Pyodide (python -m scripts.download_pyodide)."""
import json
import os
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
pytestmark = pytest.mark.skipif(not (ROOT / "app" / "web" / "vendor" / "pyodide" / "pyodide.asm.wasm").exists(),
                                reason="Pyodide не скачан: python -m scripts.download_pyodide")

from app.sandbox.executor import execute  # noqa: E402
from app.sandbox.tasks import get_task  # noqa: E402

LOOP = "def binarize(gray, threshold):\n    while True:\n        pass\n"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = dict(os.environ, DB_PATH=os.path.join(tempfile.mkdtemp(), "ui.db"), TUTOR_PROVIDER="rule", AUTO_DOWNLOAD_MODELS="false", PYTHONUTF8="1")
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


def open_code_tab(browser, server, task="binarize", block_pyodide=False, delay_pyodide_ms=0):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    if delay_pyodide_ms:   # проверка состояния «загрузка»: начало загрузки среды искусственно откладывается
        page.add_init_script("const _f2=window.fetch;window.fetch=function(u,o){const a=arguments;if(String(u).includes('/vendor/pyodide/pyodide.mjs')&&o&&o.method==='HEAD')"
                             f"return new Promise(r=>setTimeout(()=>r(_f2.apply(this,a)),{delay_pyodide_ms}));return _f2.apply(this,a)}}")
    page.add_init_script("window.__reqs=[];const _f=window.fetch;window.fetch=function(u,o){window.__reqs.push(String(u));return _f.apply(this,arguments)}")
    if block_pyodide:
        page.route("**/vendor/pyodide/**", lambda r: r.fulfill(status=404, body="no"))
    page.goto(server + "/", wait_until="domcontentloaded", timeout=90000)
    page.add_style_tag(content="#as-fab{display:none!important}")
    page.click('.tab[data-mod="code"]')
    page.wait_for_selector(f'.cdtask[data-id="{task}"]', state="attached", timeout=30000)
    page.click(f'.cdtask[data-id="{task}"]')
    page.wait_for_function(f"CD.cur && CD.cur.id === '{task}'", timeout=30000)   # задание открыто, заготовка уже в редакторе
    return page


def wait_state(page, state, timeout=180000):
    page.wait_for_function(f"typeof CDPY !== 'undefined' && CDPY.state === '{state}'", timeout=timeout)


def run_in_page(page, code, timeout=30000):
    page.evaluate("code => cdSet(code)", code)
    page.evaluate("() => { $('cdFeedback').textContent = ''; $('cdOut').textContent = ''; CD.last = null; }")
    page.click("#cdRun")
    page.wait_for_function("CD.last !== null && !$('cdRun').disabled", timeout=timeout)
    return {"feedback": page.evaluate("CD.last"), "out": page.evaluate("$('cdOut').textContent")}


def server_view(task_id, code):
    """Что показал бы серверный /run: те же тексты, которые строит cdOut."""
    t = get_task(task_id)
    res = execute(t["spec"], code, "example")
    from app.services import code_tutor
    e = res["example"]
    out = ["Пример:\n  вход:      " + e["input"] + "\n  ожидается: " + e["expected"] + "\n  результат: " +
           ("ошибка " + e["error"]["type"] + ": " + e["error"]["message"] if e["error"] else ("совпадает ✓" if e["ok"] else "не совпадает ✗ " + e["message"]))]
    return code_tutor.feedback(t, res), "\n\n".join(out)


def test_example_runs_in_the_browser_and_matches_the_server(browser, server):
    page = open_code_tab(browser, server, delay_pyodide_ms=8000)
    page.wait_for_selector("#cdRun", state="attached")
    assert page.evaluate("$('cdRun').disabled") is True and "Загрузка среды Python" in page.evaluate("$('cdRun').textContent")  # пока грузится
    wait_state(page, "ready")
    assert page.evaluate("$('cdRun').textContent") == "▶ Запустить на примере" and page.evaluate("$('cdRun').disabled") is False
    t = get_task("binarize")
    for code in (t["starter_code"], t["solution"], "import numpy as np\ndef binarize(g, t):\n    return np.where(g <= t, 0, 255).astype(np.uint8)\n"):
        page.evaluate("() => { window.__reqs.length = 0 }")
        got = run_in_page(page, code)
        want_fb, want_out = server_view("binarize", code)
        assert got["feedback"] == want_fb
        assert got["out"].strip() == want_out.strip()
        reqs = page.evaluate("window.__reqs")
        assert any("/api/v1/code/feedback" in r for r in reqs) and not any(r.endswith("/api/v1/code/run") for r in reqs)  # запуск не уходил на сервер
    page.close()


def test_other_tasks_match_the_server_in_the_browser(browser, server):
    page = open_code_tab(browser, server)
    wait_state(page, "ready")
    for tid in ("lbp", "eer", "orientation"):
        page.click(f'.cdtask[data-id="{tid}"]')
        page.wait_for_function(f"CD.cur && CD.cur.id === '{tid}'")
        t = get_task(tid)
        for code in (t["solution"], t["starter_code"]):
            got = run_in_page(page, code)
            want_fb, want_out = server_view(tid, code)
            assert got["feedback"] == want_fb and got["out"].strip() == want_out.strip(), tid
    page.close()


def test_infinite_loop_terminates_the_worker_and_the_page_stays_alive(browser, server):
    page = open_code_tab(browser, server)
    wait_state(page, "ready")
    t0 = time.time()
    got = run_in_page(page, LOOP, timeout=20000)
    assert 2.5 < time.time() - t0 < 15
    assert "тест выполнялся дольше 3 с" in got["out"] and "слишком долго" in got["feedback"]
    assert page.evaluate("1 + 1") == 2                                   # страница не зависла
    assert page.evaluate("!$('cdSubmit').disabled") is True
    wait_state(page, "ready", timeout=120000)                           # воркер создан заново и снова готов
    good = run_in_page(page, get_task("binarize")["solution"])
    assert "Пример выполняется верно" in good["feedback"]
    page.close()


def test_without_pyodide_the_server_is_used_as_a_fallback(browser, server):
    page = open_code_tab(browser, server, block_pyodide=True)
    wait_state(page, "failed", timeout=30000)
    assert page.evaluate("$('cdRun').disabled") is False and page.evaluate("$('cdRun').textContent") == "▶ Запустить на примере"
    assert "недоступна" in page.evaluate("document.getElementById('toasts').textContent")
    page.evaluate("() => { window.__reqs.length = 0 }")
    t = get_task("binarize")
    got = run_in_page(page, t["solution"])
    assert "Пример выполняется верно" in got["feedback"]
    assert any(r.endswith("/api/v1/code/run") for r in page.evaluate("window.__reqs"))
    assert "запускается на сервере" in page.evaluate("$('cdEnvNote').textContent")
    page.close()
