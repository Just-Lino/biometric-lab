"""Помощник лаборатории: превращает фразу («покажи, как изображение превращается в решение», «сравни методы на FVC2004»)
в список действий, которые выполняет интерфейс (открыть вкладку, выбрать набор и метод, нажать кнопки, провести экскурсию).

Разбор построен на правилах (ключевые слова и шаблоны), без внешних сервисов: работает офлайн и предсказуемо.
Распознавание речи и озвучку делает браузер; здесь только смысл команды. Если фраза не распознана, помощник говорит об этом
и подсказывает примеры, а не угадывает."""
import re

MOD_TITLE = {"fingerprint": "Отпечатки пальцев", "face": "Лицо", "iris": "Радужка", "train": "Обучение распознаванию",
             "code": "Практикум: код", "nn": "Площадка нейросети"}
SOURCE_TITLE = {"socofing": "SOCOFing", "casia_interval": "CASIA-Iris-Interval", "LFW": "LFW", "synthetic": "синтетика"}
LAB_PAGES = {"voice": ("/labplus/voice", "голос"), "metrics": ("/labplus/metrics", "метрики ROC и DET"),
             "robustness": ("/labplus/robustness", "устойчивость к искажениям"), "standards": ("/labplus/standards", "стандарты ISO/IEC")}

MODALITY_RE = [("fingerprint", r"отпечат|пальц|дактил|fingerprint"), ("face", r"\bлиц|face"), ("iris", r"радужк|\biris\b|глаз")]
METHODS = {  # метод (значение списка в интерфейсе) -> шаблон; свой набор для каждой биометрии
    "fingerprint": [("local", r"локальн\w* структур|локальн\w* метод|\blocal\b"), ("poc", r"\bpoc\b|пок\b|фазов\w* корреляц"), ("sift", r"\bsift\b|сифт"),
                    ("cnn", r"обученн\w* (вами|сет)|нейросет\w* обученн|нейросеть вами"), ("hough", r"голосован|hough|\bхаф")],
    "iris": [("loggabor", r"лог[ -]?габор|loggabor"), ("gabor", r"габор"), ("ordinal", r"ordinal|порядков"), ("dct", r"\bdct\b|дискретн\w* косинус"),
             ("haar", r"\bhaar\b|хаар|вейвлет"), ("cnn", r"обученн\w* (вами|сет)|нейросет\w* обученн")],
    "face": [("arcface", r"arcface|арк ?фейс"), ("sface", r"sface|эс ?фейс"), ("lbp", r"\blbp\b"), ("hog", r"\bhog\b|хог"), ("eigenfaces", r"eigen|собственн\w* лиц"),
             ("gabor", r"габор"), ("dct", r"\bdct\b"), ("pixels", r"пиксел"), ("cnn", r"обученн\w* (вами|сет)|нейросет\w* обученн")],
}
METHOD_TITLE = {"hough": "голосование", "local": "локальные структуры", "poc": "фазовая корреляция", "sift": "SIFT", "cnn": "обученная вами нейросеть",
                "gabor": "Габор", "loggabor": "лог-Габор", "ordinal": "ordinal", "dct": "DCT", "haar": "вейвлет Хаара", "arcface": "ArcFace",
                "sface": "SFace", "lbp": "LBP", "hog": "HOG", "eigenfaces": "Eigenfaces", "pixels": "пиксели"}

EXAMPLES = ["Покажи, как изображение превращается в решение", "Открой отпечатки, набор FVC2004, сравни образцы", "Оцени качество на FVC2004 DB3",
            "Покажи 3D-конвейер", "Сравни методы", "Радужка CASIA, покажи этапы", "Стресс-тест SOCOFing", "Объясни результат", "Покажи кривые ROC"]
HELP = ("Я управляю лабораторией: открываю вкладки (отпечатки, лицо, радужка, обучение), выбираю набор данных (FVC2002, FVC2004, SOCOFing, CASIA, LFW, синтетика) "
        "и метод, показываю этапы, сравниваю образцы, оцениваю качество (FAR, FRR, EER), сравниваю методы, строю 3D-рельеф и 3D-конвейер, "
        "запускаю стресс-тест искажений и обучение сети. Фраза «покажи, как изображение превращается в решение» запускает экскурсию с объяснением каждого этапа. "
        "Слова «объясни результат» разберут последнее сравнение. Страницы LAB+: голос, кривые ROC и DET, устойчивость, стандарты.")


def _norm(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"(3 ?d|3 ?д|три ?д|трех ?мерн\w*|3-d|3-д)", " 3d ", t)  # распознавание речи пишет «3D» по-разному
    t = re.sub(r"[^\w\s.,%+-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _has(t: str, pat: str) -> bool:
    return re.search(pat, t) is not None


def _dataset(t: str):
    """(значение набора, биометрия) или (None, None). Значение - идентификатор набора в лаборатории или его часть."""
    if _has(t, r"soc[o0]|сокофинг|сокоф|соко\b|соко финг"):
        return "socofing", "fingerprint"
    if _has(t, r"casia|кас+ия|кася|касиа"):
        return "casia_interval", "iris"
    if _has(t, r"\blfw\b|лфв"):
        return "LFW", "face"
    if _has(t, r"fvc|фвц|фвс|ф в ц"):
        y = re.search(r"(2002|2004)", t)
        d = re.search(r"(?:\bdb|дб|баз[аыуе]?\s*)\s*([1-4])", t)
        return f"fvc{y.group(1) if y else 2004}_db{d.group(1) if d else 1}", "fingerprint"
    if _has(t, r"синтетик|синтетическ"):
        return "synthetic", None
    return None, None


def parse(text: str, ctx: dict | None = None) -> dict:
    ctx = ctx or {}
    t = _norm(text)
    cur = ctx.get("mod") if ctx.get("mod") in MOD_TITLE else None
    out: list[dict] = []
    if not t:
        return {"understood": False, "actions": [], "reply": "Команда пустая. Скажите или напишите, что сделать.", "examples": EXAMPLES[:4]}

    if _has(t, r"что (ты )?умеешь|какие команды|что можно|помощь|\bhelp\b|справк"):
        return {"understood": True, "actions": [{"do": "help"}], "reply": HELP, "examples": EXAMPLES}

    # страницы дополнения LAB+
    page = None
    if _has(t, r"\broc\b|\bdet\b|fmr|fnmr|доверительн|правил\w* трех"):
        page = "metrics"
    elif _has(t, r"стандарт|\biso\b|исо\b|гост"):
        page = "standards"
    elif _has(t, r"(регистрац|шаблон|проверк|распознаван|биометри)\w* голос|голос\w* (биометри|профил|шаблон)|открой голос|голосов\w* биометри"):
        page = "voice"
    elif _has(t, r"устойчив|помех|robust") and not _has(t, r"стресс|искажени\w* soc|soc\w* искажени"):
        page = "robustness"
    if page:
        url, title = LAB_PAGES[page]
        return {"understood": True, "actions": [{"do": "page", "url": url}], "reply": f"Открываю страницу LAB+: {title}.", "examples": []}

    # вкладка
    mod = next((m for m, pat in MODALITY_RE if _has(t, pat)), None)
    ds, ds_mod = _dataset(t)
    other = None
    if _has(t, r"обуч|тренир|нейросет\w* учит"):
        other = "train"
    elif _has(t, r"практикум|задани\w* (по )?код|напиши\w* код"):
        other = "code"
    elif _has(t, r"площадк\w* нейросет|playground"):
        other = "nn"

    # что сделать
    acts: list[str] = []
    methods_phrase = _has(t, r"(сравни\w*|покажи|запусти)\w* (все )?(методы|способы|признаки|кодировщики|сопоставител\w*|детекторы|локализатор\w*)|сравнени\w* (методов|признаков)")
    if _has(t, r"как изображени\w* превращает\w* в решени|от изображени\w* до решени|экскурс|весь путь|весь конвейер|проведи (меня )?по этапам|расскаж\w* про (весь )?конвейер"):
        acts.append("tour")
    else:
        if _has(t, r"этап|шаги обработк|как обрабатыва|покажи обработк|пошагов|минуци"):
            acts.append("stages")
        if _has(t, r"\bсравни\w*|сравнен\w*|проверь пару|покажи решение|match") and not methods_phrase:
            acts.append("compare")
    if methods_phrase:
        if _has(t, r"оцен|eer|лучш|точност"):
            acts.append("methods_eer")
        elif _has(t, r"детектор|локализатор|бинаризац"):
            acts.append("detectors")
        else:
            acts.append("methods")
    elif _has(t, r"оцен\w* (все )?(методы|признаки|кодировщики)|лучш\w* метод"):
        acts.append("methods_eer")
    if "methods_eer" not in acts and _has(t, r"оцен\w* качеств|качеств\w* (распознаван|систем)|\beer\b|\bfar\b|\bfrr\b|посчитай (ошибк|метрик)|оцени систему"):
        acts.append("evaluate")
    elif "methods_eer" not in acts and _has(t, r"\bоцени\b") and "methods" not in acts:
        acts.append("evaluate")
    if _has(t, r"рельеф|поверхност"):
        acts.append("relief3d")
    if _has(t, r"3d|облет|облет") and "relief3d" not in acts:
        acts.append("pipeline3d")
    if _has(t, r"облет|облет|оборот|вокруг"):
        if "pipeline3d" not in acts:
            acts.append("pipeline3d")
        acts.append("orbit")
    if _has(t, r"стресс") or (ds == "socofing" and _has(t, r"искаж")):
        acts.append("stress")
    if other == "train" and _has(t, r"(останов|стоп|прекрат)"):
        acts.append("train_stop")
    elif other == "train" and _has(t, r"(начн|начни|запуст|включ|обучи|старт)"):
        acts.append("train_start")
    if _has(t, r"объясни|поясни|почему|что (это )?значит|разбери|как понять|что получилось"):
        acts.append("explain")

    thr = re.search(r"порог\w*\s*(?:равн\w*|на|в|=)?\s*(\d+(?:[.,]\d+)?)", t)
    method = None
    mm = mod or ds_mod or (cur if cur in METHODS else None)
    if mm in METHODS:
        method = next((m for m, pat in METHODS[mm] if _has(t, pat)), None)

    # вкладка для выполнения
    target = None
    if other and not (mod and acts and "train_start" not in acts and "train_stop" not in acts):
        target = other
    elif mod or ds_mod:
        target = mod or ds_mod
    elif (acts or ds or method or thr) and cur not in ("fingerprint", "face", "iris"):
        target = "fingerprint"
    if "stress" in acts:
        target = "fingerprint"
        ds = ds or "socofing"

    tab = target or cur
    bio = tab in ("fingerprint", "face", "iris")
    if target and (target != cur or other or not acts):
        out.append({"do": "open", "mod": target})
    if tab == "train" or target == "train":
        if mod or ds_mod:
            out.append({"do": "train_mod", "mod": mod or ds_mod})
        if ds:
            out.append({"do": "train_data", "value": ds})
    if bio and ds:
        out.append({"do": "source", "value": ds})
    if bio and method and mod in (None, tab):
        out.append({"do": "method", "value": method})
    if bio and thr:
        out.append({"do": "threshold", "value": round(min(0.95, max(0.01, float(thr.group(1).replace(",", ".")))), 2)})
    order = ["tour", "stages", "compare", "methods", "detectors", "methods_eer", "evaluate", "relief3d", "pipeline3d", "orbit", "stress",
             "train_start", "train_stop", "explain"]
    for a in order:
        if a in acts:
            if a in ("tour", "stages", "compare", "methods", "detectors", "methods_eer", "evaluate", "relief3d", "pipeline3d", "orbit") and not bio:
                continue
            out.append({"do": a})

    if not out:
        return {"understood": False, "actions": [], "reply": "Команда не распознана. Я работаю по ключевым словам, попробуйте сформулировать иначе, например: "
                + "; ".join(f"«{e}»" for e in EXAMPLES[:3]) + ". Скажите «что ты умеешь», чтобы услышать весь список.", "examples": EXAMPLES}
    return {"understood": True, "actions": out, "reply": _reply(out), "examples": []}


def _reply(actions: list[dict]) -> str:
    parts = []
    for a in actions:
        d = a["do"]
        if d == "open":
            parts.append(f"открываю вкладку «{MOD_TITLE.get(a['mod'], a['mod'])}»")
        elif d == "source":
            v = a["value"]
            parts.append("выбираю набор " + SOURCE_TITLE.get(v, v.replace("_db", " DB").upper() if v.startswith("fvc") else v))
        elif d == "train_mod":
            parts.append(f"обучаю на данных: {MOD_TITLE.get(a['mod'], a['mod']).lower()}")
        elif d == "train_data":
            v = a["value"]
            parts.append("набор для обучения " + SOURCE_TITLE.get(v, v.replace("_db", " DB").upper() if v.startswith("fvc") else v))
        elif d == "method":
            parts.append(f"метод: {METHOD_TITLE.get(a['value'], a['value'])}")
        elif d == "threshold":
            parts.append(f"порог {a['value']}".replace(".", ","))
        else:
            parts.append({"tour": "провожу экскурсию от изображения до решения", "stages": "показываю этапы обработки", "compare": "сравниваю образцы A и B",
                          "methods": "сравниваю методы на выбранной паре", "detectors": "сравниваю способы получения признаков", "methods_eer": "оцениваю методы по EER",
                          "evaluate": "считаю FAR, FRR и EER", "relief3d": "строю 3D-рельеф", "pipeline3d": "строю 3D-конвейер", "orbit": "включаю облёт камеры",
                          "stress": "запускаю стресс-тест искажений", "train_start": "готовлю запуск обучения", "train_stop": "останавливаю обучение",
                          "explain": "разбираю последний результат"}.get(d, d))
    s = ", ".join(parts)
    return s[:1].upper() + s[1:] + "."
