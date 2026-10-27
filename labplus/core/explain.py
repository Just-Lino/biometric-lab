"""ИИ-наставник на правилах (без ключей): объясняет результат эксперимента.
level 1 — подсказка, 2 — объяснение причины, 3 — практическая рекомендация.
Вместо этого модуля можно подключить LLM: функция получает те же словари."""
from __future__ import annotations


def _lv(level: int, hint: str, why: str, todo: str) -> str:
    return {1: hint, 2: f"{hint} {why}", 3: f"{hint} {why} {todo}"}.get(level, f"{hint} {why}")


def voice(res: dict, level: int = 2) -> str:
    s, t = res.get("score"), res.get("threshold")
    if s is None or t is None:
        return "Сначала запишите голос и выполните проверку."
    if s >= t:
        margin = s - t
        close = " Запас небольшой, решение неустойчиво." if margin < 0.08 else ""
        return _lv(level, f"Оценка сходства {s:.2f} не ниже порога {t:.2f}, поэтому голос принят.{close}",
                   "Это значит, что признаки записи близки к шаблону: похожи тембр и высота тона.",
                   "Проверьте тот же голос с шумом или через динамик телефона и посмотрите, при каком искажении оценка упадёт ниже порога.")
    q = res.get("quality") or {}
    note = ""
    if q.get("speech_sec", 9) < 1.5:
        note = " Речи в записи мало."
    if q.get("clipping", 0) > 0.01:
        note += " Есть перегрузка микрофона."
    return _lv(level, f"Оценка {s:.2f} ниже порога {t:.2f}, голос не принят.{note}",
               "Причины: другой говорящий, шум, другая интонация или расстояние до микрофона.",
               "Запишите ту же фразу ближе к микрофону в тишине. Если это ваш голос, добавьте в шаблон ещё 2–3 фразы или немного снизьте порог, но помните, что ложные допуски вырастут.")


def metrics(res: dict, level: int = 2, idx: dict | None = None) -> str:
    e = res.get("eer")
    if e is None:
        return "Сначала постройте кривые."
    d = res.get("dprime", 0)
    hint = f"Равная ошибка EER = {e * 100:.2f} %, AUC = {res.get('auc', 0):.3f}, d′ = {d:.1f}."
    if idx:
        hint += (f" При выбранном пороге FMR = {idx['fmr'] * 100:.2f} %, FNMR = {idx['fnmr'] * 100:.2f} %.")
    qual = "очень хорошо разделяет" if e < 0.02 else "в целом разделяет" if e < 0.1 else "плохо разделяет"
    why = (f"Система {qual} свои и чужие пары: чем меньше перекрытие гистограмм, тем ниже EER. "
           "Повышение порога снижает FMR (меньше ложных допусков), но повышает FNMR (больше ложных отказов).")
    ops = res.get("ops") or []
    weak = [o for o in ops if not o["reliable"]]
    todo = ("Выберите порог по требованию: для доступа к важным данным берите малый FMR, для удобства — малый FNMR. "
            + (f"Для FMR = {weak[0]['target_fmr']:g} данных мало: нужно хотя бы {weak[0]['need_impostors']} чужих сравнений (правило трёх). " if weak else "")
            + "Сравните ROC и DET двух систем: кривая ближе к левому нижнему углу — лучше.")
    return _lv(level, hint, why, todo)


def robust(res: dict, level: int = 2) -> str:
    items = res.get("results") or []
    if not items:
        return "Сначала запустите проверку устойчивости."
    th = res.get("threshold", 0.5)
    broke = [(r["title"], r["breakpoint"], r["unit"]) for r in items if r["breakpoint"] is not None]
    robust_ = [r["title"] for r in items if r["breakpoint"] is None]
    if broke:
        worst = broke[0]
        hint = f"При пороге {th:.2f} система ошибается при искажении «{worst[0]}» уже на уровне {worst[1]} ({worst[2]})."
    else:
        hint = f"При пороге {th:.2f} ни одно искажение не сломало сопоставление до максимального уровня."
    why = ("Оценка падает, когда искажение разрушает признаки, по которым идёт сравнение. "
           + (f"Устойчиво к: {', '.join(robust_)}. " if robust_ else ""))
    todo = ("Добавьте предобработку (нормализация контраста, шумоподавление, выравнивание по повороту), "
            "либо подключите другой сопоставитель и сравните кривые. Для отчёта по стандарту ISO/IEC 19795-1 фиксируйте условия и размер выборки.")
    return _lv(level, hint, why, todo)


def generic(kind: str, payload: dict, level: int = 2) -> str:
    if kind == "voice":
        return voice(payload, level)
    if kind == "metrics":
        return metrics(payload, level, payload.get("at_threshold"))
    if kind == "robust":
        return robust(payload, level)
    return "Не знаю, какой результат объяснять. Запустите эксперимент и повторите."
