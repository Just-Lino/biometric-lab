"""Разбор голосовой команды (правила + ключевые слова, без внешних ключей).
Возвращает действие, цель, параметры и тексты ответа для озвучивания."""
from __future__ import annotations

import re

TARGETS = {
    "fingerprint": ["отпеч", "палец", "пальц", "минуци", "fingerprint"],
    "face": ["лиц", "face", "эмбеддинг"],
    "iris": ["радуж", "ирис", "iris", "глаз"],
    "voice": ["голос", "речь", "диктор"],
    "metrics": ["roc", "рок", "det", "eer", "far", "frr", "кривую", "кривые", "метрик", "порог", "ошибок"],
    "robustness": ["атак", "устойчив", "шум", "искажен", "помех", "робаст"],
    "standards": ["стандарт", "iso", "исо", "нормат", "нист", "nist"],
}
TARGET_RU = {
    "fingerprint": "распознавание отпечатков пальцев", "face": "распознавание лица",
    "iris": "распознавание радужки", "voice": "голосовую биометрию",
    "metrics": "метрики и кривые ошибок", "robustness": "проверку устойчивости к искажениям",
    "standards": "раздел о международных стандартах",
}
HOW_IT_WORKS = {
    "fingerprint": "Отпечаток обрабатывается так: улучшение изображения, бинаризация, скелетизация линий, поиск окончаний и разветвлений, затем сопоставление шаблонов и решение по порогу.",
    "face": "Лицо обрабатывается так: поиск лица, выравнивание по глазам, получение вектора признаков, сравнение косинусной близостью и решение по порогу.",
    "iris": "Радужка: находим зрачок и границу радужки, разворачиваем кольцо в прямоугольник, фильтры Габора дают код, коды сравниваются расстоянием Хэмминга.",
    "voice": "Голос: выделяем речь, считаем мел-кепстральные коэффициенты и высоту тона, строим шаблон по нескольким фразам и сравниваем новую запись с ним.",
    "metrics": "Здесь показаны распределения оценок своих и чужих, ошибки FMR и FNMR при разных порогах, кривые ROC и DET и равная ошибка EER.",
    "robustness": "Изображение или звук искажаются всё сильнее, и мы смотрим, при каком уровне искажения оценка сходства падает ниже порога.",
    "standards": "В этом разделе собраны международные стандарты по тестированию, качеству, защите шаблонов и атакам на биометрию, с привязкой к функциям лаборатории.",
}
ACTIONS = {
    "run": ["покаж", "запуст", "начн", "выполн", "сравн", "распозна", "провер", "открой", "включ", "постро", "сдела", "проведи", "зайд"],
    "explain": ["объясн", "поясн", "почему", "что получилос", "что вышло", "как работает", "расскаж", "что значит", "что это"],
    "help": ["помощ", "что ты умеешь", "что можешь", "какие команды", "справк"],
    "stop": ["стоп", "отмена", "хватит", "остановись"],
    "yes": ["да", "давай", "запускай", "согласен", "конечно"],
    "no": ["нет", "не надо", "не нужно"],
    "greet": ["привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро"],
}

WHOLE_WORD = {"roc", "рок", "det", "eer", "far", "frr", "iso", "исо", "да", "нет"}
_DS = re.compile(r"(?:датасет\w*|набор\w*(?:\s+данных)?|баз\w+)\s+([\w\-\.]+)", re.I)
_PEOPLE = re.compile(r"(\d+)\s*(?:человек\w*|люд\w*|субъект\w*|пальц\w*|пар\w*)", re.I)


def normalize(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    return re.sub(r"[^\w\s\-\.]", " ", t, flags=re.UNICODE)


def _has(t: str, stems: list[str]) -> list[str]:
    words = t.split()
    hits = []
    for s in stems:
        if s in WHOLE_WORD:                      # короткие слова — только целым словом
            if s in words:
                hits.append(s)
        elif s in t:
            hits.append(s)
    return hits


def parse(text: str) -> dict:
    t = normalize(text)
    action, matched = None, []
    for a in ("stop", "help", "explain", "run", "greet", "yes", "no"):
        h = _has(t, ACTIONS[a])
        if h:
            action, matched = a, h
            break
    target, thits = None, []
    best = 0
    for tg, stems in TARGETS.items():
        h = _has(t, stems)
        if len(h) > best:
            target, thits, best = tg, h, len(h)
    ds = _DS.search(t)
    pe = _PEOPLE.search(t)
    slots = {}
    if ds:
        slots["dataset"] = ds.group(1)
    if pe:
        slots["people"] = int(pe.group(1))

    if action is None and target:
        action = "run"                           # «отпечатки пальцев» → показать
    if action in ("run", "explain") and target is None and action == "run":
        action = None
    conf = 0.0
    if action and target:
        conf = 0.9
    elif action in ("stop", "help", "greet", "yes", "no") or (action == "explain" and target is None):
        conf = 0.8
    intent = action if conf > 0 else "unknown"
    return {"intent": intent, "target": target, "slots": slots, "confidence": conf,
            "matched": matched + thits, "text": text, **_replies(intent, target, slots)}


def _slot_phrase(slots: dict) -> str:
    bits = []
    if "dataset" in slots:
        bits.append(f"по датасету {slots['dataset']}")
    if "people" in slots:
        bits.append(f"для {slots['people']} человек")
    return (" " + ", ".join(bits)) if bits else ""


def _replies(intent: str, target: str | None, slots: dict) -> dict:
    tr = TARGET_RU.get(target or "", "")
    how = HOW_IT_WORKS.get(target or "", "")
    sp = _slot_phrase(slots)
    if intent == "run":
        return {"reply_run": f"Да, понял. Запускаю {tr}{sp}. {how}",
                "reply_hint": f"Да, понял. {how} Откройте раздел и нажмите кнопку запуска.",
                "reply_confirm": f"Я поняла: {tr}{sp}. Запустить сейчас? Скажите да или нет."}
    if intent == "explain":
        if target:
            return {"reply_run": how, "reply_hint": how, "reply_confirm": how}
        return {"reply_run": "Объясняю последний результат.", "reply_hint": "Объясняю последний результат.",
                "reply_confirm": "Объясняю последний результат."}
    if intent == "help":
        t = ("Я понимаю команды вроде: покажи распознавание отпечатков пальцев по датасету Ф-1; "
             "запусти распознавание лица; покажи кривые ROC; проверь устойчивость к шуму; "
             "объясни результат. Скажите, что нужно.")
        return {"reply_run": t, "reply_hint": t, "reply_confirm": t}
    if intent == "greet":
        t = "Здравствуйте! Я голосовой помощник лаборатории. Скажите, например: покажи распознавание отпечатков пальцев."
        return {"reply_run": t, "reply_hint": t, "reply_confirm": t}
    if intent == "stop":
        return {"reply_run": "Хорошо, остановилась.", "reply_hint": "Хорошо, остановилась.", "reply_confirm": "Хорошо, остановилась."}
    if intent in ("yes", "no"):
        return {"reply_run": "", "reply_hint": "", "reply_confirm": ""}
    t = "Не поняла команду. Попробуйте: покажи распознавание отпечатков пальцев, или скажите «помощь»."
    return {"reply_run": t, "reply_hint": t, "reply_confirm": t}
