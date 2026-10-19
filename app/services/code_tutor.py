"""ИИ-поддержка при выполнении кода: объяснение ошибок и многоуровневые подсказки.
Сейчас работает на правилах и подсказках из JSON задания; провайдер LLM можно подключить в tutor.py так же, как для лаборатории."""

LEVELS = {1: "направление", 2: "объяснение", 3: "фрагмент кода", 4: "полное решение"}
MIN_ATTEMPTS_FOR_SOLUTION = 3

ERR_TEXT = {
    "SyntaxError": "Синтаксическая ошибка{line}: {msg}. Проверьте скобки, двоеточия после if/for/def и отступы.",
    "NameError": "Имя не определено{line}: {msg}. Возможно, опечатка в названии переменной или забыт импорт (например, import numpy as np).",
    "TypeError": "Несовместимые типы или неверное число аргументов{line}: {msg}. Проверьте, что возвращают ваши выражения: массив, число или None.",
    "IndexError": "Индекс за пределами массива{line}: {msg}. Проверьте границы, особенно у краёв изображения.",
    "ValueError": "Неверное значение или форма данных{line}: {msg}. Проверьте формы массивов и аргументы функций.",
    "ZeroDivisionError": "Деление на ноль{line}. Добавьте проверку знаменателя перед делением.",
    "AttributeError": "Нет такого атрибута или метода{line}: {msg}. Проверьте тип объекта (список, массив numpy, число).",
    "NoFunction": "Функция не найдена. Проверьте, что вы не изменили имя и параметры функции из шаблона.",
    "Timeout": "Код выполняется слишком долго: возможно, бесконечный цикл или медленный перебор пикселей в цикле. Попробуйте векторизацию numpy.",
    "Blocked": "Код заблокирован защитой песочницы: {msg}",
    "OverflowError": "Переполнение числа{line}: {msg}. Проверьте типы данных (uint8 переполняется при вычитании).",
}


def explain_error(err: dict) -> str:
    line = f" (строка {err['line']})" if err.get("line") else ""
    tpl = ERR_TEXT.get(err["type"], "Ошибка " + err["type"] + "{line}: {msg}")
    return tpl.format(line=line, msg=err.get("message", ""))


def feedback(task: dict, result: dict, grading: dict | None = None) -> str:
    """Текстовый разбор результата для студента (без выдачи готового решения)."""
    if result.get("error"):
        return explain_error(result["error"])
    ex = result.get("example")
    if ex:  # режим «запуск на примере»
        if ex.get("error"):
            return explain_error(ex["error"])
        if ex["ok"]:
            return "Пример выполняется верно. Нажмите «Проверить решение»: проверка включает ещё и граничные случаи."
        return "Результат на примере не совпал с ожидаемым. " + ex.get("message", "")
    tests = result.get("tests", [])
    if not tests:
        return ""
    bad_basic = [t for t in tests if t["kind"] == "basic" and not t["ok"]]
    bad_edge = [t for t in tests if t["kind"] == "edge" and not t["ok"]]
    errs = [t for t in tests if t.get("error")]
    notes = []
    if errs:
        notes.append("Во время тестов возникла ошибка: " + explain_error(errs[0]["error"]))
    if bad_basic:
        none = [t for t in bad_basic if "None" in t["message"]]
        if none:
            notes.append("Функция возвращает None: проверьте, что есть оператор return со значением.")
        notes.append(f"Не пройдены базовые тесты ({len(bad_basic)} из {len([t for t in tests if t['kind'] == 'basic'])}). Первый: «{bad_basic[0]['name']}» - {bad_basic[0]['message']}")
    elif bad_edge:
        notes.append("Базовые тесты пройдены, но не пройдены граничные случаи: " + "; ".join(t["name"] for t in bad_edge) +
                     ". Подумайте, чем эти ситуации отличаются от обычных.")
    elif not errs:
        notes.append("Все тесты пройдены.")
    if grading and not grading["answered"]:
        notes.append("Контрольный вопрос без ответа: выберите вариант, это даёт дополнительный балл.")
    elif grading and not grading["answer_ok"]:
        notes.append("Ответ на контрольный вопрос неверный: вернитесь к теории задания.")
    return " ".join(notes)


def hint(task: dict, level: int, attempts: int) -> tuple[bool, str]:
    """(доступно, текст). Полное решение открывается только после нескольких самостоятельных попыток."""
    if level == 4 and attempts < MIN_ATTEMPTS_FOR_SOLUTION:
        return False, (f"Полное решение откроется после {MIN_ATTEMPTS_FOR_SOLUTION} самостоятельных попыток проверки "
                       f"(сейчас: {attempts}). Попробуйте подсказки уровней 1-3.")
    return True, task["hints"][level - 1]
