"""ИИ-наставник. Интерфейс TutorProvider позволяет менять «мозг» без изменения остального кода.

rule   - правила по контексту эксперимента (работает офлайн, без ключа)
openai - модель через OpenAI API (включается через .env, когда появится ключ)
Режимы: hint (подсказка), explain (объяснение понятия), diagnose (разбор ошибки).
Наставник не выдаёт готовое решение оцениваемого задания."""
from abc import ABC, abstractmethod

from app.core.config import settings

GLOSSARY = {
    ("бинаризац", "binar"): "Бинаризация превращает полутоновое изображение в чёрно-белое: гребни отделяются от фона. "
        "Слишком низкий порог рвёт линии, слишком высокий сливает соседние гребни. От этого зависит скелет и число ложных minutiae.",
    ("similarity", "сходств", "score"): "Similarity score показывает, насколько похожи два шаблона. "
        "Здесь это доля совпавших minutiae: 2·совпавшие / (minutiae A + minutiae B), значение от 0 до 1.",
    ("threshold", "порог"): "Threshold (порог) задаёт границу решения: score ≥ порога - MATCH, иначе NO MATCH. "
        "Повышение порога уменьшает FAR (ложные допуски), но увеличивает FRR (ложные отказы).",
    ("minutia", "минуци"): "Minutiae - особые точки гребней: окончание (ridge ending) и разветвление (bifurcation). "
        "Шаблон отпечатка - набор (x, y, угол, тип) для каждой такой точки.",
    ("gabor", "габор"): "Фильтр Габора избирательно усиливает периодическую структуру нужной ориентации и частоты. "
        "Для отпечатков он «залечивает» разрывы гребней и подавляет шум. Важно, чтобы период фильтра соответствовал расстоянию между гребнями.",
    ("clahe", "контраст", "нормализац"): "CLAHE выравнивает контраст локально, по небольшим областям. Это снижает влияние неравномерного давления и освещения.",
    ("скелет", "thinning", "skeleton"): "Скелетизация утоньшает гребни до 1 пикселя. Только на тонких линиях просто определить окончания и разветвления по crossing number.",
    ("far",): "FAR (False Acceptance Rate) - доля пар разных людей, ошибочно принятых как совпадение.",
    ("frr",): "FRR (False Rejection Rate) - доля пар одного человека, ошибочно отвергнутых.",
    ("eer",): "EER - точка, где FAR = FRR. Чем она ниже, тем лучше система различает своих и чужих.",
    ("embedding", "эмбеддинг", "вектор признак"): "Face embedding - вектор чисел, в который нейросеть превращает выровненное лицо. "
        "Похожие лица дают близкие векторы, поэтому лица сравнивают не по пикселям, а по расстоянию между векторами.",
    ("cosine", "косинус"): "Cosine similarity - косинус угла между двумя embedding-векторами: 1 - направления совпадают, около 0 - не связаны. Не зависит от длины вектора.",
    ("alignment", "выравнив"): "Alignment поворачивает и масштабирует лицо так, чтобы глаза и рот оказались в стандартных местах. Без него наклон и разный масштаб ухудшают сравнение.",
    ("landmark", "ключевые точки", "точки лица"): "Landmarks - ключевые точки лица: глаза, нос, углы рта. По ним вычисляют преобразование для выравнивания.",
    ("haar", "хаар", "yunet", "детекц"): "Детектор находит рамку лица. Haar - классический каскад на простых признаках: быстрый, но чувствителен к повороту и свету. "
        "YuNet - лёгкая нейросеть: точнее и сразу выдаёт 5 ключевых точек.",
    ("lbp",): "LBP кодирует каждый пиксель по сравнению с соседями, а гистограммы по ячейкам лица образуют вектор признаков. Это классический метод: проще нейросети, но слабее при смене освещения и позы.",
    ("ориентац",): "Поле ориентаций - направление гребней в каждой точке. По нему Gabor-фильтр выбирает, под каким углом усиливать линии.",
}


class TutorProvider(ABC):
    name = "base"

    @abstractmethod
    def ask(self, mode: str, question: str, context: dict) -> str: ...


class RuleBasedTutor(TutorProvider):
    name = "rule"

    def ask(self, mode: str, question: str, context: dict) -> str:
        if mode == "explain":
            return self._explain(question, context)
        if mode == "diagnose":
            return self._diagnose(context)
        return self._hint(context)

    def _explain(self, question: str, ctx: dict) -> str:
        q = (question or "").lower()
        for keys, text in GLOSSARY.items():
            if any(k in q for k in keys):
                return text
        return ("Уточните, какое понятие объяснить: similarity, threshold, FAR/FRR/EER, embedding, cosine, alignment, landmarks, детектор; "
                "для отпечатков ещё бинаризация, minutiae, Gabor, CLAHE, скелетизация.")

    def _diagnose_face(self, c: dict) -> str:
        s, t = c["similarity"], c.get("threshold", 0.4)
        notes = []
        if abs(s - t) < 0.05:
            notes.append(f"Similarity {s:.2f} почти равна порогу {t:.2f}: решение неустойчиво.")
        elif s >= t:
            notes.append(f"Similarity {s:.2f} не ниже порога {t:.2f}, поэтому MATCH.")
        else:
            notes.append(f"Similarity {s:.2f} ниже порога {t:.2f}, поэтому NO MATCH. Если это один человек, причины могут быть в повороте головы, освещении, "
                         "ошибке детекции или выключенном выравнивании.")
        if c.get("embedder") == "lbp":
            notes.append("Используется классический LBP: он чувствителен к освещению и позе. Сравните результат с SFace, если модель доступна.")
        if c.get("align") is False:
            notes.append("Выравнивание выключено: включите его и посмотрите, как изменится similarity.")
        if c.get("detector") == "haar":
            notes.append("Haar даёт грубые ключевые точки; YuNet обычно выравнивает точнее.")
        if c.get("extra_noise", 0) > 20:
            notes.append("Добавлен сильный шум: снижение сходства ожидаемо.")
        return " ".join(notes)

    def _diagnose(self, c: dict) -> str:
        if not c or c.get("similarity") is None:
            return "Сначала выполните сравнение двух образцов, затем я разберу результат."
        if c.get("modality") == "face":
            return self._diagnose_face(c)
        s, t = c["similarity"], c.get("threshold", 0.5)
        na, nb, m = c.get("n_a", 0), c.get("n_b", 0), c.get("matched", 0)
        notes = []
        if min(na, nb) < 15:
            notes.append(f"Найдено мало minutiae ({na} и {nb}). Возможные причины: слишком агрессивная бинаризация, "
                         "малый период Gabor, большое минимальное расстояние между точками или низкое качество образца.")
        if max(na, nb) > 80:
            notes.append(f"Найдено очень много minutiae ({na} и {nb}), часть, скорее всего, ложные: шум создаёт «шипы» и разрывы. "
                         "Попробуйте увеличить σ Gabor или минимальное расстояние между точками.")
        if abs(s - t) < 0.05:
            notes.append(f"Score {s:.2f} почти равен порогу {t:.2f}: решение неустойчиво, небольшое изменение параметров его перевернёт.")
        elif s >= t:
            notes.append(f"Score {s:.2f} не ниже порога {t:.2f}, поэтому система выдала MATCH ({m} совпавших minutiae).")
        else:
            notes.append(f"Score {s:.2f} ниже порога {t:.2f}, поэтому NO MATCH. Если образцы одного пальца, это ложный отказ: "
                         "проверьте качество, шум, допуск по расстоянию и угол поворота.")
        if c.get("extra_noise", 0) > 20:
            notes.append("Вы добавили сильный шум: падение сходства ожидаемо. Сравните результат с шумом и без него.")
        return " ".join(notes)

    def _hint(self, c: dict) -> str:
        if not c or c.get("similarity") is None:
            return "Выберите два образца и нажмите «Сравнить». Затем посмотрите, сколько minutiae найдено на каждом."
        s, t = c["similarity"], c.get("threshold", 0.5)
        if c.get("modality") == "face":
            if abs(s - t) < 0.05:
                return "Similarity рядом с порогом. Что произойдёт с решением, если чуть сдвинуть порог? Проверьте ползунком."
            return "Посмотрите на этапы детекции и выравнивания: правильно ли найдены глаза и рамка? Как это могло повлиять на embedding?"
        if min(c.get("n_a", 0), c.get("n_b", 0)) < 15:
            return "Посмотрите на этапы бинаризации и скелета: все ли гребни сохранились? Какой параметр на них влияет?"
        if abs(s - t) < 0.05:
            return "Score рядом с порогом. Что произойдёт с решением, если чуть сдвинуть порог? Проверьте ползунком."
        return "Измените один параметр и сравните результат с предыдущим экспериментом в журнале. Что изменилось сильнее всего?"


class OpenAITutor(TutorProvider):
    name = "openai"
    SYSTEM = ("Ты ИИ-наставник виртуальной лаборатории по биометрии. Отвечай по-русски, кратко. "
              "Не давай готового решения оцениваемого задания: режим hint - только направление, "
              "explain - объяснение понятия, diagnose - причина ошибки по данным эксперимента.")

    def __init__(self):
        from openai import OpenAI  # импорт здесь, чтобы пакет был нужен только при включении
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.fallback = RuleBasedTutor()

    def ask(self, mode: str, question: str, context: dict) -> str:
        try:
            r = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": self.SYSTEM},
                          {"role": "user", "content": f"Режим: {mode}\nВопрос: {question}\nКонтекст: {context}"}])
            return r.choices[0].message.content
        except Exception:
            return self.fallback.ask(mode, question, context)


_tutor: TutorProvider | None = None


def get_tutor() -> TutorProvider:
    global _tutor
    if _tutor is None:
        if settings.tutor_provider == "openai" and settings.openai_api_key:
            _tutor = OpenAITutor()
        else:
            _tutor = RuleBasedTutor()
    return _tutor
