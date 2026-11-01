"""Помощник лаборатории: превращает фразу («покажи, как изображение превращается в решение», «сравни методы на FVC2004»)
в список действий, которые выполняет интерфейс (открыть вкладку, выбрать набор, метод и образцы, нажать кнопки, провести экскурсию).

Разбор построен на правилах (ключевые слова и шаблоны), без внешних сервисов: работает офлайн и предсказуемо.
Распознавание речи и озвучку делает браузер; здесь только смысл команды. Он умеет:
- числа словами («порог ноль пять», «палец сто один оттиск три»);
- несколько вариантов распознавания от браузера (берётся первый, который удалось понять);
- подсказки «возможно, вы имели в виду», если фраза не распознана, вместо угадывания."""
import difflib
import re

MOD_TITLE = {"fingerprint": "Отпечатки пальцев", "face": "Лицо", "iris": "Радужка", "voice": "Голос", "train": "Обучение распознаванию",
             "code": "Практикум: код", "nn": "Площадка нейросети"}
SOURCE_TITLE = {"socofing": "SOCOFing", "casia_interval": "CASIA-Iris-Interval", "LFW": "LFW", "synthetic": "синтетика"}
LAB_PAGES = {"voice": ("/labplus/voice", "голос"), "metrics": ("/labplus/metrics", "метрики ROC и DET"),
             "robustness": ("/labplus/robustness", "устойчивость к искажениям"), "standards": ("/labplus/standards", "стандарты ISO/IEC")}
TASK_TITLE = {"binarize": "бинаризация", "normalize": "нормализация", "crossing_number": "число пересечений", "verify": "верификация", "far_frr": "FAR и FRR"}
TASK_ORDER = ["binarize", "normalize", "crossing_number", "verify", "far_frr"]
TASK_RE = [("binarize", r"бинаризац"), ("normalize", r"нормализац"), ("crossing_number", r"пересечен|crossing|минуци"), ("verify", r"верификац"),
           ("far_frr", r"\bfar\b|\bfrr\b|фар\b|фрр")]

MODALITY_RE = [("fingerprint", r"отпечат|пальц|дактил|fingerprint"), ("face", r"\bлиц|face"), ("iris", r"радужк|\biris\b|глаз"),
                ("voice", r"\bголос(?:а|у|ом|е|ов)?\b|голосов(?!ан)\w*|диктор\w*|спикер|говорящ")]
METHODS = {  # метод (значение списка в интерфейсе) -> шаблон; свой набор для каждой биометрии
    "fingerprint": [("local", r"локальн\w* структур|локальн\w* метод|\blocal\b"), ("poc", r"\bpoc\b|пок\b|фазов\w* корреляц"), ("sift", r"\bsift\b|сифт"),
                    ("cnn", r"обученн\w* (вами|сет)|нейросет\w* обученн|нейросеть вами"), ("hough", r"голосован|hough|\bхаф")],
    "iris": [("loggabor", r"лог[ -]?габор|loggabor"), ("gabor", r"габор"), ("ordinal", r"ordinal|порядков"), ("dct", r"\bdct\b|дискретн\w* косинус"),
             ("haar", r"\bhaar\b|хаар|вейвлет"), ("cnn", r"обученн\w* (вами|сет)|нейросет\w* обученн")],
    "voice": [("combo", r"комбинир|\bcombo\b|вместе с высотой"), ("mfcc_delta", r"дельт|динамик"), ("pitch", r"высот\w* тона|\bpitch\b|основн\w* тон"),
              ("mfcc", r"\bmfcc\b|мфкк|мел-?частот")],
    "face": [("arcface", r"arcface|арк ?фейс"), ("sface", r"sface|эс ?фейс"), ("lbp", r"\blbp\b"), ("hog", r"\bhog\b|хог"), ("eigenfaces", r"eigen|собственн\w* лиц"),
             ("gabor", r"габор"), ("dct", r"\bdct\b"), ("pixels", r"пиксел"), ("cnn", r"обученн\w* (вами|сет)|нейросет\w* обученн")],
}
METHOD_TITLE = {"mfcc": "MFCC", "mfcc_delta": "MFCC с динамикой", "pitch": "высота тона", "combo": "MFCC и высота тона", "hough": "голосование", "local": "локальные структуры", "poc": "фазовая корреляция", "sift": "SIFT", "cnn": "обученная вами нейросеть",
                "gabor": "Габор", "loggabor": "лог-Габор", "ordinal": "ordinal", "dct": "DCT", "haar": "вейвлет Хаара", "arcface": "ArcFace",
                "sface": "SFace", "lbp": "LBP", "hog": "HOG", "eigenfaces": "Eigenfaces", "pixels": "пиксели"}

EXAMPLES = ["Покажи, как изображение превращается в решение", "Открой отпечатки, набор FVC2004, сравни образцы", "Оцени качество на FVC2004 DB3",
            "Покажи 3D-конвейер", "Сравни методы", "Радужка CASIA, покажи этапы", "Стресс-тест SOCOFing", "Объясни результат", "Покажи кривые ROC"]
# фразы для подсказки «возможно, вы имели в виду» (по близости к тому, что не удалось разобрать)
PHRASE_BANK = EXAMPLES + [
    "Сравни палец 101 оттиск 1 и палец 102 оттиск 1", "Выбери палец 103 оттиск 2 как образец B", "Следующий этап", "Предыдущий этап", "Перейди на этап 4",
    "Увеличь порог", "Уменьши порог на 0,1", "Порог ноль пять", "Покажи рельеф", "Покажи 3D-конвейер и облёт", "Оцени методы по EER", "Сравни методом SIFT",
    "Открой голос и сравни записи", "Запиши мой голос в образец А", "Прослушай образец Б", "Сравни дикторов методом MFCC", "Открой практикум", "Задание бинаризация", "Запусти код на примере", "Дай подсказку", "Проверь решение", "Начни обучение на CASIA", "Останови обучение",
    "Открой площадку нейросети и запусти обучение", "Повтори", "Что ты умеешь", "Покажи стандарты ISO", "Проверь устойчивость к шуму",
]
HELP = ("Я управляю лабораторией: открываю вкладки (отпечатки, лицо, радужка, голос, обучение, практикум, площадка), выбираю набор данных (FVC2002, FVC2004, SOCOFing, CASIA, LFW, синтетика), "
        "метод, порог и образцы A и B («выбери палец сто один оттиск три», «сравни палец сто один и палец сто два»), показываю этапы и перехожу между ними («следующий этап», «этап четыре»), "
        "сравниваю образцы, оцениваю качество (FAR, FRR, EER), сравниваю методы, строю 3D-рельеф и 3D-конвейер, запускаю стресс-тест искажений и обучение сети. "
        "Фраза «покажи, как изображение превращается в решение» запускает экскурсию с объяснением каждого этапа. "
        "В практикуме: «открой задание бинаризация», «запусти код», «дай подсказку два», «проверь решение». «Повтори» повторяет прошлую команду, «стоп» останавливает. "
        "Слова «объясни результат» разберут последнее сравнение. Страницы LAB+: голос, кривые ROC и DET, устойчивость, стандарты.")

_UNITS = {"ноль": 0, "нуль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9}
_TEENS = {"десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
          "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19}
_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90}
_ORD = {"первый": 1, "первая": 1, "второй": 2, "третий": 3, "четвертый": 4, "пятый": 5, "шестой": 6, "седьмой": 7, "восьмой": 8}


def _numwords(t: str) -> str:
    """Числа словами в цифры: «сто один» -> 101, «ноль целых пять десятых» и «ноль пять» после слова «порог» -> 0,5, «пятьдесят процентов» -> 0,5."""
    toks, out, i = t.split(" "), [], 0
    while i < len(toks):
        w, val, used = toks[i], None, 0
        if w == "сто":
            val, used, j = 100, 1, i + 1
            if j < len(toks) and toks[j] in _TENS:
                val += _TENS[toks[j]]; used += 1; j += 1
            elif j < len(toks) and toks[j] in _TEENS:
                val += _TEENS[toks[j]]; used += 1; j += 1
            if j < len(toks) and toks[j] in _UNITS and _UNITS[toks[j]] > 0 and (used == 1 or toks[j - 1] in _TENS):
                val += _UNITS[toks[j]]; used += 1
        elif w in _TENS:
            val, used, j = _TENS[w], 1, i + 1
            if j < len(toks) and toks[j] in _UNITS and _UNITS[toks[j]] > 0:
                val += _UNITS[toks[j]]; used += 1
        elif w in _TEENS:
            val, used = _TEENS[w], 1
        elif w in _UNITS:
            val, used = _UNITS[w], 1
        elif w in _ORD and ((i + 1 < len(toks) and toks[i + 1] in ("этап", "шаг")) or (i > 0 and toks[i - 1] in ("этап", "шаг"))):
            val, used = _ORD[w], 1
        if val is None:
            out.append(w); i += 1
        else:
            out.append(str(val)); i += used
    s = " ".join(out)

    def dec(n: str, kind: str) -> str:
        k = int(n)
        return f"0,{k}" if kind != "сотых" else (f"0,0{k}" if k < 10 else f"0,{k}")

    s = re.sub(r"\b0\s*(?:целых|целое|запятая|точка)?\s*(\d{1,2})\s*(десятых|сотых)\b", lambda m: dec(m.group(1), m.group(2)), s)
    s = re.sub(r"\b0\s*(?:целых|целое|запятая|точка)\s*(\d{1,2})\b", lambda m: dec(m.group(1), ""), s)
    s = re.sub(r"(порог\w*\s*(?:равн\w*|на|в)?\s*)0\s+(\d{1,2})\b", lambda m: m.group(1) + dec(m.group(2), ""), s)
    s = re.sub(r"\b(\d)\s*десятых\b", lambda m: dec(m.group(1), "десятых"), s)
    s = re.sub(r"\b(\d{1,3})\s*процент\w*", lambda m: str(round(int(m.group(1)) / 100, 2)).replace(".", ","), s)
    return s


def _norm(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"(3 ?d|3 ?д|три ?д|трех ?мерн\w*|3-d|3-д)", " 3d ", t)  # распознавание речи пишет «3D» по-разному
    t = re.sub(r"[^\w\s.,%+-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _numwords(t)


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


_SAMPLE = re.compile(r"(?:палец|пальца|глаз|глаза|диктор\w*)\s*(?:номер\s*)?(\d{1,3})(?:\s*(?:оттиск\w*|снимок|снимка|запис\w*)\s*(?:номер\s*)?(\d{1,2}))?")


def _samples(t: str) -> list[dict]:
    """«Палец 101 оттиск 3» (один или два раза) -> действия выбора образцов A и B."""
    found = [(int(m.group(1)), int(m.group(2) or 1)) for m in _SAMPLE.finditer(t)]
    if not found:
        return []
    kind = "диктор" if _has(t, r"диктор") else ("глаз" if _has(t, r"глаз") and not _has(t, r"палец|пальц") else "палец")
    if len(found) >= 2:
        slots = ["a", "b"]
    else:
        slots = ["b" if _has(t, r"\b(b|б|бэ|би|проверяем\w*)\b") else "a"]
    return [{"do": "sample", "slot": s, "kind": kind, "n": n, "m": m} for s, (n, m) in zip(slots, found[:2])]


def _parse_one(text: str, ctx: dict) -> dict:
    t = _norm(text)
    cur = ctx.get("mod") if ctx.get("mod") in MOD_TITLE else None
    out: list[dict] = []
    if not t:
        return {"understood": False, "actions": [], "reply": "Команда пустая. Скажите или напишите, что сделать.", "examples": EXAMPLES[:4], "score": 0}

    if _has(t, r"что (ты )?умеешь|какие команды|что можно|помощь|\bhelp\b|справк"):
        return {"understood": True, "actions": [{"do": "help"}], "reply": HELP, "examples": EXAMPLES, "score": 1}
    if _has(t, r"^(повтори|еще раз|ещё раз|снова|повтор)( команду)?$"):
        return {"understood": True, "actions": [{"do": "repeat"}], "reply": "Повторяю последнюю команду.", "examples": [], "score": 1}
    if _has(t, r"^(стоп|хватит|отмена|отмени|остановись|прекрати)$"):
        return {"understood": True, "actions": [{"do": "cancel"}], "reply": "Останавливаю.", "examples": [], "score": 1}

    # страницы дополнения LAB+
    page = None
    if _has(t, r"\broc\b|\bdet\b|fmr|fnmr|доверительн|правил\w* трех"):
        page = "metrics"
    elif _has(t, r"стандарт|\biso\b|исо\b|гост"):
        page = "standards"
    elif _has(t, r"lab ?\+|лаб ?плюс|страниц\w* голос|регистрац\w* голос|профил\w* голос"):  # страница дополнения LAB+; вкладка «Голос» лаборатории открывается словом «голос»
        page = "voice"
    elif _has(t, r"устойчив|помех|robust") and not _has(t, r"стресс|искажени\w* soc|soc\w* искажени"):
        page = "robustness"
    if page:
        url, title = LAB_PAGES[page]
        return {"understood": True, "actions": [{"do": "page", "url": url}], "reply": f"Открываю страницу LAB+: {title}.", "examples": [], "score": 1}

    # вкладка
    mod = next((m for m, pat in MODALITY_RE if _has(t, pat)), None)
    ds, ds_mod = _dataset(t)
    other = None
    if _has(t, r"площадк\w*|playground"):
        other = "nn"
    elif _has(t, r"практикум|задани\w*|\bкод\b|кода\b|подсказк\w*|решени\w* (задани|кода)|напиши\w* код") and not _has(t, r"изображени\w* превращает|до решени"):
        other = "code"
    elif _has(t, r"обуч|тренир|нейросет\w* учит"):
        other = "train"
    if cur == "nn" and other == "train" and not (mod or ds_mod or ds):
        other = None  # на площадке «запусти обучение» относится к ней, а не к вкладке «Обучение распознаванию»
    # команды практикума и площадки понимаются там, где они открыты, только если в фразе есть их слова; иначе («покажи этапы») это биометрическая команда
    in_code = other == "code" or (cur == "code" and other is None and _has(t, r"запусти|запуск|выполни|\bкод\b|проверь|отправь|сдай|решени|подсказ|подскаж|помоги|задани|начальн|объясни|поясни|разбери|почему"))
    in_nn = other == "nn" or (cur == "nn" and other is None and _has(t, r"запусти|начни|обуч|включи|старт|останов|стоп|пауза|шаг|сброс|заново|с нуля|эпох"))

    acts: list[str] = []
    extra: list[dict] = []  # действия с параметрами (порог, этап, задание, подсказка)

    # --- практикум и площадка нейросети
    if in_code:
        tk = next((i for i, pat in TASK_RE if _has(t, pat)), None)
        mn = re.search(r"задани\w*\s*(?:номер\s*)?(\d)\b", t)
        if tk or mn:
            idx = TASK_ORDER[int(mn.group(1)) - 1] if mn and 1 <= int(mn.group(1)) <= len(TASK_ORDER) else tk
            if idx:
                extra.append({"do": "task", "id": idx})
        if _has(t, r"запусти|запуск|выполни|прогони") and _has(t, r"\bкод\b|кода|пример|решени"):
            acts.append("code_run")
        if _has(t, r"проверь|отправь|сдай|оцени") and _has(t, r"решени|код|задани"):
            acts.append("code_submit")
        if _has(t, r"начальн\w* код|сброс\w* код|верни\w* (начальн|шаблон)"):
            acts.append("code_reset")
        if _has(t, r"подсказк|подскаж|помоги|полное решение|покажи решение"):
            lv = re.search(r"подсказк\w*\s*(?:уровня\s*)?(\d)\b", t) or re.search(r"(\d)\s*подсказк", t)
            level = int(lv.group(1)) if lv else (4 if _has(t, r"полное решение|покажи решение") else 1)
            extra.append({"do": "hint", "level": max(1, min(4, level))})
        if _has(t, r"объясни|поясни|разбери|почему") and not extra:
            acts.append("code_explain")
    if in_nn:
        if _has(t, r"запусти|начни|обуч|включи|старт") and not _has(t, r"останов|стоп"):
            acts.append("nn_play")
        elif _has(t, r"останов|стоп|пауза|прекрати"):
            acts.append("nn_play")  # та же кнопка: пауза
        if _has(t, r"\bшаг\b|одну эпоху|одна эпоха"):
            acts.append("nn_step")
        if _has(t, r"сброс|заново|с нуля"):
            acts.append("nn_reset")

    # --- биометрические вкладки
    methods_phrase = _has(t, r"(сравни\w*|покажи|запусти)\w* (все )?(методы|способы|признаки|кодировщики|сопоставител\w*|детекторы|локализатор\w*)|сравнени\w* (методов|признаков)")
    step = None
    if _has(t, r"следующ\w* (этап|шаг)|^дальше$|^вперед$|дальше по этапам"):
        step = {"do": "step", "delta": 1}
    elif _has(t, r"предыдущ\w* (этап|шаг)|^назад$|вернись на этап"):
        step = {"do": "step", "delta": -1}
    elif _has(t, r"последн\w* (этап|шаг)"):
        step = {"do": "step", "n": -1}
    else:
        sn = re.search(r"(?:этап|шаг)\s*(?:номер\s*)?(\d)\b", t) or re.search(r"\b(\d)\s*(?:-?й\s*)?(?:этап|шаг)\b", t)
        if sn:
            step = {"do": "step", "n": int(sn.group(1))}
    if not (in_code or in_nn):
        if _has(t, r"как изображени\w* превращает\w* в решени|от изображени\w* до решени|экскурс|весь путь|весь конвейер|проведи (меня )?по этапам|расскаж\w* про (весь )?конвейер"):
            acts.append("tour")
        else:
            if step is None and _has(t, r"этап|шаги обработк|как обрабатыва|покажи обработк|пошагов|минуци"):
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
        if _has(t, r"3d|облет") and "relief3d" not in acts:
            acts.append("pipeline3d")
        if _has(t, r"облет|оборот|вокруг"):
            if "pipeline3d" not in acts:
                acts.append("pipeline3d")
            acts.append("orbit")
        if _has(t, r"стресс") or (ds == "socofing" and _has(t, r"искаж")):
            acts.append("stress")
    if other == "train" and _has(t, r"(останов|стоп|прекрат)"):
        acts.append("train_stop")
    elif other == "train" and _has(t, r"(начн|начни|запуст|включ|обучи|старт)"):
        acts.append("train_start")
    if _has(t, r"объясни|поясни|почему|что (это )?значит|разбери|как понять|что получилось") and not in_code:
        acts.append("explain")

    relative = _has(t, r"(увелич|повыс|подним|больше|выше|уменьш|пониз|опуст|меньше|ниже)\w*.*порог|порог.*(больше|выше|меньше|ниже)")
    thr = None if relative else re.search(r"порог\w*\s*(?:равн\w*|на|в|=)?\s*(\d+(?:[.,]\d+)?)", t)
    delta = None
    if not thr and _has(t, r"порог"):
        dd = re.search(r"на\s*(\d+(?:[.,]\d+)?)", t)
        step_v = float(dd.group(1).replace(",", ".")) if dd and float(dd.group(1).replace(",", ".")) <= 0.5 else 0.05
        if _has(t, r"увелич|повыс|подним|больше|выше"):
            delta = step_v
        elif _has(t, r"уменьш|пониз|опуст|меньше|ниже"):
            delta = -step_v
    method = None
    mm = mod or ds_mod or (cur if cur in METHODS else None)
    if mm in METHODS:
        method = next((m for m, pat in METHODS[mm] if _has(t, pat)), None)
    samples = _samples(t) if not (in_code or in_nn) else []

    # вкладка для выполнения
    target = None
    if other and not (mod and acts and "train_start" not in acts and "train_stop" not in acts and other == "train"):
        target = other
    elif mod or ds_mod:
        target = mod or ds_mod
    elif (acts or ds or method or thr or delta or step or samples) and cur not in ("fingerprint", "face", "iris", "voice") and not (in_code or in_nn):
        target = "fingerprint"
    if "stress" in acts:
        target = "fingerprint"
        ds = ds or "socofing"

    tab = target or cur
    voice_act = None  # запись и прослушивание образцов на вкладке «Голос»
    if tab == "voice":
        slot = "b" if _has(t, r"\b(b|б|бэ|би)\b") else "a"
        if _has(t, r"запиш\w+|запись с микрофона|продиктуй\w*"):
            voice_act = {"do": "record", "slot": slot}
        elif _has(t, r"прослуша\w+|проиграй|воспроизведи"):
            voice_act = {"do": "play", "slot": slot}
    bio = tab in ("fingerprint", "face", "iris", "voice")
    if target and (target != cur or other or not (acts or extra or samples or step or voice_act)):
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
    elif bio and delta:
        out.append({"do": "threshold", "delta": delta})
    if bio:
        out.extend(samples)
    if voice_act:
        out.append(voice_act)
    order = ["tour", "stages", "compare", "methods", "detectors", "methods_eer", "evaluate", "relief3d", "pipeline3d", "orbit", "stress",
             "train_start", "train_stop", "explain", "code_run", "code_submit", "code_reset", "code_explain", "nn_play", "nn_step", "nn_reset"]
    bio_only = {"tour", "stages", "compare", "methods", "detectors", "methods_eer", "evaluate", "relief3d", "pipeline3d", "orbit"}
    if in_code:
        out.extend(extra)
    if bio and step:
        out.append(step)
    for a in order:
        if a in acts:
            if a in bio_only and not bio:
                continue
            out.append({"do": a})

    if not out:
        return {"understood": False, "actions": [], "reply": "", "examples": EXAMPLES, "score": 0}
    return {"understood": True, "actions": out, "reply": _reply(out), "examples": [], "score": len(out)}


def parse(text: str, ctx: dict | None = None, alternatives: list[str] | None = None) -> dict:
    """Разбор команды. alternatives - другие варианты распознавания речи: берётся первый вариант, который удалось понять (с большим числом действий)."""
    ctx = ctx or {}
    cands = [text] + [a for a in (alternatives or []) if a and a != text]
    best, heard = None, text
    for c in cands[:5]:
        r = _parse_one(c, ctx)
        if r["understood"] and (best is None or r["score"] > best["score"]):
            best, heard = r, c
    if best is None:
        r = _parse_one(text, ctx)
        norm = _norm(text)
        bank = {_norm(p): p for p in PHRASE_BANK}
        close = [bank[k] for k in difflib.get_close_matches(norm, list(bank), n=3, cutoff=0.4)] if norm else []
        r["reply"] = r["reply"] or ("Команда не распознана. " + (("Возможно, вы имели в виду: " + "; ".join(f"«{c}»" for c in close) + ". ") if close else
                                    ("Попробуйте, например: " + "; ".join(f"«{e}»" for e in EXAMPLES[:3]) + ". ")) + "Скажите «что ты умеешь», чтобы услышать весь список.")
        r["examples"] = close or EXAMPLES
        r["heard"] = text
        r.pop("score", None)
        return r
    best.pop("score", None)
    best["heard"] = heard
    return best


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
            parts.append((f"порог {a['value']}" if "value" in a else ("повышаю порог" if a["delta"] > 0 else "понижаю порог") + f" на {abs(a['delta'])}").replace(".", ","))
        elif d == "record":
            parts.append(f"записываю голос в образец {a['slot'].upper()} (4 секунды)")
        elif d == "play":
            parts.append(f"проигрываю образец {a['slot'].upper()}")
        elif d == "sample":
            parts.append(f"выбираю образец {a['slot'].upper()}: {a['kind']} {a['n']}, снимок {a['m']}")
        elif d == "step":
            parts.append("следующий этап" if a.get("delta") == 1 else "предыдущий этап" if a.get("delta") == -1 else
                         "последний этап" if a.get("n") == -1 else f"перехожу на этап {a.get('n')}")
        elif d == "task":
            parts.append(f"открываю задание «{TASK_TITLE.get(a['id'], a['id'])}»")
        elif d == "hint":
            parts.append(f"показываю подсказку {a['level']}")
        else:
            parts.append({"tour": "провожу экскурсию от изображения до решения", "stages": "показываю этапы обработки", "compare": "сравниваю образцы A и B",
                          "methods": "сравниваю методы на выбранной паре", "detectors": "сравниваю способы получения признаков", "methods_eer": "оцениваю методы по EER",
                          "evaluate": "считаю FAR, FRR и EER", "relief3d": "строю 3D-рельеф", "pipeline3d": "строю 3D-конвейер", "orbit": "включаю облёт камеры",
                          "stress": "запускаю стресс-тест искажений", "train_start": "готовлю запуск обучения", "train_stop": "останавливаю обучение",
                          "explain": "разбираю последний результат", "code_run": "запускаю код на примере", "code_submit": "проверяю решение",
                          "code_reset": "возвращаю начальный код", "code_explain": "объясняю последний результат", "nn_play": "переключаю обучение на площадке",
                          "nn_step": "делаю один шаг обучения", "nn_reset": "сбрасываю площадку", "repeat": "повторяю последнюю команду"}.get(d, d))
    s = ", ".join(parts)
    return s[:1].upper() + s[1:] + "."
