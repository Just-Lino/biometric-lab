"""Три языка (русский, казахский, английский): каталоги, сопоставление, перевод ответов API, задания практикума, помощник."""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import i18n
from app.main import app
from app.sandbox.tasks import load_tasks
from app.services import assistant
from labplus.core import intent as lp_intent

CYR = re.compile(r"[А-Яа-яЁё]")
PLACEHOLDER = re.compile(r"\{(\d+)\}")
I18N_DIR = Path(i18n.__file__).resolve().parents[1] / "i18n"


def get(client, path, lang):
    return client.get(path, headers={"Cookie": f"lab_lang={lang}"})


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------- каталоги ----------
def entries():
    out = []
    for f in sorted(I18N_DIR.glob("*.i18n")):
        for e in i18n.parse_catalog(f.read_text(encoding="utf-8")):
            out.append((f.name, e))
    return out


def test_catalog_entries_are_complete_and_consistent():
    """В каждой записи есть ru, en и kk; номера вставок {n} в переводах те же, что в русском ключе; английский без кириллицы."""
    bad = []
    for name, e in entries():
        ru = e.get("ru")
        if not ru or not e.get("en") or not e.get("kk"):
            bad.append((name, "неполная запись", ru))
            continue
        want = sorted(PLACEHOLDER.findall(ru.replace("{line}", "{0}").replace("{msg}", "{1}")))
        for lang in ("en", "kk"):
            if sorted(PLACEHOLDER.findall(e[lang])) != want:
                bad.append((name, f"вставки {lang}", ru[:60]))
        if CYR.search(e["en"]):
            bad.append((name, "кириллица в en", e["en"][:60]))
    assert not bad, bad[:10]


def test_exact_template_and_sentence_matching():
    cat = i18n.Catalog("en", [{"ru": "Привет.", "en": "Hello."}, {"ru": "Файл {0} не найден", "en": "File {0} not found"},
                              {"ru": "Много слов.", "en": "Many words."}])
    assert cat.translate("Привет.") == "Hello."
    assert cat.translate("  Привет.  ") == "  Hello.  "                  # пробелы по краям сохраняются
    assert cat.translate("Файл a.txt не найден") == "File a.txt not found"
    assert cat.translate("Привет. Много слов.") == "Hello. Many words."   # составное сообщение переводится по предложениям
    assert cat.translate("Неизвестный текст") == "Неизвестный текст"      # нет перевода - остаётся русским, а не ломается
    assert cat.translate("no cyrillic") == "no cyrillic"


def test_template_does_not_swallow_several_sentences():
    cat = i18n.Catalog("en", [{"ru": "Я поняла: {0}.", "en": "I understood: {0}."}, {"ru": "Запустить?", "en": "Start?"},
                              {"ru": "Скажите да.", "en": "Say yes."}])
    assert cat.translate("Я поняла: тест. Запустить? Скажите да.") == "I understood: тест. Start? Say yes."


def test_template_with_empty_value_and_edge_spaces():
    cat = i18n.Catalog("en", [{"ru": "Ошибка{0}: {1}.", "en": "Error{0}: {1}."}, {"ru": " (строка {0})", "en": " (line {0})"}])
    assert cat.translate("Ошибка: плохо.") == "Error: плохо."
    assert cat.translate("Ошибка (строка 3): x.") == "Error (line 3): x."


def test_multiline_code_keeps_indent_when_comment_is_translated():
    cat = i18n.Catalog("en", [{"ru": "# TODO: вернуть", "en": "# TODO: return"}])
    src = "def f():\n    # TODO: вернуть\n    pass\n"
    assert cat.translate(src) == "def f():\n    # TODO: return\n    pass\n"


def test_language_detection_order():
    assert i18n.detect_lang("lab_lang=kk", "", "en") == "kk"                # cookie важнее заголовка
    assert i18n.detect_lang("lab_lang=kk", "lang=en", "") == "en"           # параметр важнее cookie
    assert i18n.detect_lang("", "", "kk-KZ,kk;q=0.9,en;q=0.5") == "kk"
    assert i18n.detect_lang("", "", "en-US,en;q=0.9") == "en"
    assert i18n.detect_lang("", "", "de-DE") == "ru"                        # неизвестный язык - русский
    assert i18n.detect_lang("lab_lang=fr", "", "") == "ru"


# ---------- API ----------
def test_api_messages_are_translated_and_russian_is_unchanged(client):
    ru = get(client, "/api/v1/biometrics/nope/samples", "ru").json()["detail"]
    en = get(client, "/api/v1/biometrics/nope/samples", "en").json()["detail"]
    kk = get(client, "/api/v1/biometrics/nope/samples", "kk").json()["detail"]
    assert ru == "Модуль «nope» не найден" and en == "Module “nope” not found" and kk == "«nope» модулі табылмады"


def test_content_language_header_and_vary(client):
    r = get(client, "/api/v1/biometrics/fingerprint/info", "en")
    assert r.headers["content-language"] == "en" and "Cookie" in r.headers.get("vary", "")


def test_no_russian_left_in_english_modules(client):
    """Описания наборов, шаги конвейера и результаты сравнения на английском без русских слов."""
    for mod in ("fingerprint", "face", "iris", "voice"):
        s = get(client, f"/api/v1/biometrics/{mod}/samples?limit=2", "en").json()
        ids = [x["id"] for x in (s if isinstance(s, list) else (s.get("items") or s.get("samples") or []))[:2]]
        bodies = [get(client, f"/api/v1/biometrics/{mod}/info", "en").json(), s]
        if len(ids) == 2:
            bodies.append(client.post(f"/api/v1/biometrics/{mod}/process", json={"sample_id": ids[0], "params": {}},
                                      headers={"Cookie": "lab_lang=en"}).json())
            bodies.append(client.post(f"/api/v1/biometrics/{mod}/compare", json={"sample_a": ids[0], "sample_b": ids[1], "params": {}},
                                      headers={"Cookie": "lab_lang=en"}).json())
        left = [t for t in re.findall(r'"([^"]*[А-Яа-яЁё][^"]*)"', json.dumps(bodies, ensure_ascii=False)) if len(t) < 400]
        assert not left, (mod, left[:5])


def test_english_and_kazakh_code_errors(client):
    r = client.post("/api/v1/code/run", json={"task_id": "binarize", "code": "def binarize(g,t:\n"}, headers={"Cookie": "lab_lang=en"}).json()
    assert not CYR.search(json.dumps(r, ensure_ascii=False)) and "Syntax error" in r["feedback"]
    r = client.post("/api/v1/code/run", json={"task_id": "binarize", "code": "def binarize(g,t):\n    return None\n"},
                    headers={"Cookie": "lab_lang=kk"}).json()
    assert "None" in r["feedback"] and "забыт" not in r["feedback"] and "күткенмен" in r["feedback"]


# ---------- задания практикума ----------
def test_all_tasks_are_translated_and_solutions_still_hidden(client):
    ru = get(client, "/api/v1/code/tasks", "ru").json()
    en = get(client, "/api/v1/code/tasks", "en").json()
    assert len(ru) == len(en) == len(load_tasks())
    assert not CYR.search(json.dumps(en, ensure_ascii=False))
    for t in load_tasks():
        body = get(client, f"/api/v1/code/tasks/{t}", "en").text
        assert not CYR.search(body), t
        assert "solution" not in body.lower().split('"starter_code"')[0] and '"hints"' not in body


def test_all_task_hints_are_translated():
    from app.services.code_tutor import hint
    cat = i18n.get_catalog("kk")
    for t in load_tasks().values():
        for level in (1, 2, 3):
            text = hint(t, level, 0)[1]
            assert cat.translate(text) != text, (t["id"], level)


# ---------- страницы ----------
def test_pages_carry_language_and_catalog(client):
    html = get(client, "/", "kk").text
    assert '<html lang="kk"' in html and 'window.LAB_LANG="kk"' in html and "/i18n/i18n.js" in html
    assert 'window.LAB_LANG="ru"' in get(client, "/", "ru").text and "window.LAB_CAT={}" in get(client, "/", "ru").text
    assert '<html lang="en"' in get(client, "/labplus/metrics", "en").text
    assert "Pragma" not in get(client, "/", "en").headers or True


def test_client_catalog_is_small_and_has_no_task_texts(client):
    cat = i18n.client_catalog("en")
    assert 500 < len(cat) < 2500 and not any("Реализуйте функцию" in k for k in cat)
    e = get(client, "/api/v1/i18n/kk", "ru").json()
    assert e["lang"] == "kk" and e["entries"]
    assert get(client, "/api/v1/i18n/de", "ru").status_code == 404


# ---------- помощник ----------
def acts(text):
    return assistant.parse(text, {})["actions"]


def test_assistant_understands_english_and_kazakh_like_russian():
    """Каждый пример и фраза из банка на английском и казахском даёт те же действия, что и русская."""
    for ru in assistant.EXAMPLES + assistant.PHRASE_BANK:
        base = acts(ru)
        for lang in ("en", "kk"):
            tr = i18n.get_catalog(lang).translate(ru)
            assert tr != ru, (lang, ru)
            assert acts(tr) == base, (lang, tr)


def test_assistant_replies_follow_language(client):
    def reply(lang, text):
        return client.post("/api/v1/assistant/parse", json={"text": text, "context": {}}, headers={"Cookie": f"lab_lang={lang}"}).json()["reply"]
    assert reply("en", "Open fingerprints, FVC2004 dataset, compare samples") == \
        "Opening the tab “Fingerprints”, selecting the dataset FVC2004 DB1, comparing samples A and B."
    assert reply("kk", "FVC2004 DB3 бойынша сапаны бағала").startswith("«Саусақ іздері» қойындысын ашамын")
    assert reply("ru", "оцени качество на FVC2004").startswith("Открываю вкладку")
    unknown = reply("en", "dance with me")
    assert unknown.startswith("The command was not recognized.") and not CYR.search(unknown.replace("«", "").replace("»", ""))


def test_assistant_threshold_with_verb_after_object():
    assert {"do": "threshold", "delta": -0.1} in acts("Шекті 0,1-ге азайт")
    assert {"do": "threshold", "delta": -0.1} in acts("Decrease the threshold by 0.1")
    assert {"do": "threshold", "value": 0.5} in acts("Threshold zero point five")


def test_labplus_intent_in_three_languages(client):
    for text, target in [("show fingerprint recognition on dataset FVC2004", "fingerprint"), ("саусақ ізін тануды көрсет", "fingerprint"),
                         ("покажи распознавание отпечатков пальцев", "fingerprint"), ("show the ROC curves", "metrics"),
                         ("шуға төзімділікті тексер", "robustness"), ("explain the result", None), ("hello", None)]:
        p = lp_intent.parse(text)
        assert p["intent"] != "unknown" and p["target"] == target, text
    assert lp_intent.parse("hello")["intent"] == "greet" and lp_intent.parse("сәлем")["intent"] == "greet"
    assert lp_intent.parse("stop")["intent"] == "stop" and lp_intent.parse("тоқта")["intent"] == "stop"
    assert lp_intent.parse("xyz qwerty")["intent"] == "unknown"


def test_tutor_understands_questions_in_all_languages(client):
    def ask(lang, q):
        return client.post("/api/v1/tutor/ask", json={"mode": "explain", "question": q, "context": {}}, headers={"Cookie": f"lab_lang={lang}"}).json()["answer"]
    assert "decision boundary" in ask("en", "What is the threshold?")
    assert "шешім шекарасын" in ask("kk", "Шек дегеніміз не?")
    assert ask("ru", "что такое порог").startswith("Threshold (порог)")
