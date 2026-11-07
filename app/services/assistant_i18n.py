"""Английские и казахские команды помощника: слова переводятся в русские «опорные» формулировки, которые понимает разбор в assistant.py.

Так правила разбора живут в одном месте (русском), а на другие языки добавляется только словарь. Правила применяются к уже
приведённому к нижнему регистру тексту; русские фразы они не затрагивают (шаблоны - только латиница и казахские слова)."""
import re

# --- английский ---------------------------------------------------------------------------------------------------------------
_EN_UNITS = {"zero": "ноль", "one": "один", "two": "два", "three": "три", "four": "четыре", "five": "пять", "six": "шесть", "seven": "семь",
             "eight": "восемь", "nine": "девять", "ten": "десять", "eleven": "одиннадцать", "twelve": "двенадцать", "thirteen": "тринадцать",
             "fourteen": "четырнадцать", "fifteen": "пятнадцать", "sixteen": "шестнадцать", "seventeen": "семнадцать", "eighteen": "восемнадцать",
             "nineteen": "девятнадцать", "twenty": "двадцать", "thirty": "тридцать", "forty": "сорок", "fifty": "пятьдесят", "sixty": "шестьдесят",
             "seventy": "семьдесят", "eighty": "восемьдесят", "ninety": "девяносто"}
_EN_FIRST = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8"}

EN = [
    (r"how (?:the )?(?:image|picture|sample|photo) (?:turns?|becomes?|is turned|gets turned|changes|is converted) (?:in)?to (?:a |the )?decision"
     r"|from (?:the )?(?:image|picture) to (?:the )?decision|guided tour|\btour\b|whole pipeline|entire pipeline|walk me through", "как изображение превращается в решение"),
    (r"what (?:can|do) you (?:do|know)|which commands|list of commands|what is possible|what are the commands", "что ты умеешь"),
    (r"^(?:repeat|again|once more)(?: the)?(?: last)?(?: command)?$", "повтори"),
    (r"\b(?:stop|cancel|abort|enough|quit|halt)\b", "стоп"),
    (r"\b(?:go to|switch to|open|navigate to|take me to)\b", "открой"),
    (r"\b(?:show|display|demonstrate|give me|let me see|visuali[sz]e)\b", "покажи"),
    (r"\bfinger(?:print)? (?:number |no\.? )?(\d+)", r"палец \1"),
    (r"\bfingerprints?\b|\bfingers\b|\bdactyl\w*", "отпечатки"),
    (r"\b(?:impression|print|scan)s? (?:number |no\.? )?(\d+)", r"оттиск \1"),
    (r"\beye (?:number |no\.? )?(\d+)", r"глаз \1"),
    (r"\bspeaker (?:number |no\.? )?(\d+)", r"диктор \1"),
    (r"\b(?:recording|phrase|utterance) (?:number |no\.? )?(\d+)", r"запись \1"),
    (r"\b(?:shot|take|image|photo|picture|capture) (?:number |no\.? )?(\d+)", r"снимок \1"),
    (r"\bsample (a|b)\b|\b(a|b) sample\b", lambda m: "образец " + (m.group(1) or m.group(2))),
    (r"\bsamples?\b", "образец"),
    (r"\b(?:neural network playground|neural net playground|playground)\b", "площадка"),
    (r"\b(?:code lab|coding lab|coding practice|practicum|programming practice)\b", "практикум"),
    (r"\b(?:task|exercise|assignment)\b", "задание"),
    (r"\bbinari[sz]ation\b|\bbinari[sz]e\b", "бинаризация"),
    (r"\bnormali[sz]ation\b|\bnormali[sz]e\b", "нормализация"),
    (r"\bcrossing number\b", "число пересечений"),
    (r"\bverification\b|\bverify\b", "верификация"),
    (r"\bfull solution\b|\bshow the solution\b", "полное решение"),
    (r"\b(?:hint|clue)s?(?: level)?\b", "подсказка"),
    (r"\blevel\b", "уровня"),
    (r"\b(?:reset|restore)(?: the)?(?: starter| initial| original)? code\b|\bstarter code\b", "верни начальный код"),
    (r"\b(?:run|execute|launch|try)\b(?=\W+(?:\w+\W+){0,2}(?:code|example|solution))", "запусти"),
    (r"\b(?:check|submit|grade|verify)\b(?=\W+(?:\w+\W+){0,2}(?:solution|code|task))", "проверь"),
    (r"\bcode\b", "код"),
    (r"\bexample\b", "пример"),
    (r"\bsolution\b", "решение"),
    (r"\bexplain\b|\bclarify\b|\binterpret\b", "объясни"),
    (r"\bwhat does (?:it|this|that) mean\b|\bwhat happened\b|\bhow (?:do i|to) (?:read|understand)\b|\bwhat (?:did we get|was the result)\b", "что получилось"),
    (r"\bwhy\b", "почему"),
    (r"\bcompare (?:all |the )*(?:methods|ways|approaches)\b|\bmethods? comparison\b|\bcompare (?:the )?(?:features|encoders|matchers)\b", "сравни методы"),
    (r"\bcompare (?:the )?(?:detectors|localizers|localisers)\b", "сравни детекторы"),
    (r"\bevaluate (?:all |the )*(?:methods|features|encoders)\b|\brank (?:the )?methods\b|\bbest method\b|\bmethods? by eer\b", "оцени методы"),
    (r"\b(?:evaluate|assess|estimate|measure|calculate|compute)\b(?: the)?(?: system)?(?: recognition)? (?:quality|performance|accuracy|errors?)\b", "оцени качество"),
    (r"\b(?:evaluate|assess|estimate)\b", "оцени"),
    (r"\bcompar(?:e|ison|ing)\b|\bmatch (?:them|the samples)\b", "сравни"),
    (r"\bnext (?:step|stage)\b|\bgo on\b|\bcontinue\b|\bproceed\b", "следующий этап"),
    (r"\b(?:previous|prior) (?:step|stage)\b|\bgo back\b|\bstep back\b", "предыдущий этап"),
    (r"\b(?:last|final) (?:step|stage)\b", "последний этап"),
    (r"\b(?:step|stage) (?:number |no\.? )?(\d)\b", r"этап \1"),
    (r"\b(\d)(?:st|nd|rd|th)? (?:step|stage)\b", r"этап \1"),
    (r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth) (?:step|stage)\b", lambda m: "этап " + _EN_FIRST[m.group(1)]),
    (r"\b(?:stages|steps|processing steps|pipeline steps|processing stages)\b", "этапы"),
    (r"\bthreshold\b(?: to| at| of| equals?| is| =)?\s*(?=\d)", "порог "),
    (r"\bthreshold\b", "порог"),
    (r"\b(?:increase|raise|higher|greater|bigger|up|boost)\b", "увеличь"),
    (r"\b(?:decrease|lower|reduce|drop|smaller|down|cut)\b", "уменьши"),
    (r"\bby (?=\d)", "на "),
    (r"\b(?:relief|surface|height ?map)\b", "рельеф"),
    (r"\b(?:orbit|fly around|fly-through|flythrough|rotate around|fly)\b", "облет"),
    (r"\b(?:stress[- ]test|distortion test)\b", "стресс-тест"),
    (r"\bdistortions?\b", "искажения"),
    (r"\b(?:robustness|resilience|robust)\b", "устойчивость"),
    (r"\bstandards?\b", "стандарты"),
    (r"\blab ?plus\b", "лаб плюс"),
    (r"\bsynthetic\b", "синтетика"),
    (r"\bdatabase\b", "база"),
    (r"\blocal structures?\b|\blocal method\b", "локальные структуры"),
    (r"\bphase correlation\b", "фазовая корреляция"),
    (r"\bvoting\b", "голосование"),
    (r"\blog[- ]?gabor\b", "лог-габор"),
    (r"\bgabor\b", "габор"),
    (r"\b(?:haar|wavelet)\b", "хаар"),
    (r"\bpixels?\b", "пиксел"),
    (r"\b(?:network (?:that )?(?:you|i|we) trained|(?:my|your|the|our) trained (?:neural )?network|trained network|network trained by you)\b", "нейросеть обученная вами"),
    (r"\b(?:delta|dynamics)\b", "дельта"),
    (r"\bcombined\b|\bcombo\b", "комбинир"),
    (r"\b(?:start|begin|launch|kick off)\b", "запусти"),
    (r"\btrain(?:ing)?\b", "обучение"),
    (r"\brecord(?: my)?(?: voice| speech| phrase)?\b", "запиши мой голос"),
    (r"\b(?:play(?: back)?|listen to|listen)\b", "прослушай"),
    (r"\b(?:voices?|speech)\b", "голос"),
    (r"\bspeakers?\b", "диктор"),
    (r"\beyes?\b", "глаз"),
    (r"\bpitch\b", "высота тона"),
    (r"\bpercent\b", "процентов"),
    (r"\bpoint\b", "точка"),
    (r"\b(?:a )?hundred(?: and)?\b", "сто"),
]

# --- казахский ----------------------------------------------------------------------------------------------------------------
KK = [
    (r"экскурсия\w*|суреттің шешімге қалай айналатын\w*|суреттен шешімге дейін\w*|бүкіл конвейер\w*|барлық кезеңдер арқылы", "как изображение превращается в решение"),
    (r"не істей алас\w*|қандай командалар\w*|барлық командалар\w*|\bкөмек\b", "что ты умеешь"),
    (r"^(?:қайтала\w*|тағы бір рет|қайта)(?: команданы)?$", "повтори"),
    (r"\bтоқта\w*|\bболдырма\w*|\bжеткілікті\b|\bдоғар\w*", "стоп"),
    (r"\bаш(?:ып|ыңыз|шы|ып көрсет)?\b", "открой"),
    (r"\bкөрсет\w*", "покажи"),
    (r"\bсаусақ(?: ізі| іздері)? (?:нөмірі |№ ?)?(\d+)", r"палец \1"),
    (r"\bсаусақ(?: ізі| іздері)?\w*|\bсаусақтар\w*|\bдактилоскопия\w*", "отпечатки"),
    (r"\bтаңба (?:нөмірі |№ ?)?(\d+)", r"оттиск \1"),
    (r"\bкөз (?:нөмірі |№ ?)?(\d+)", r"глаз \1"),
    (r"\bдиктор (?:нөмірі |№ ?)?(\d+)", r"диктор \1"),
    (r"\bжазба (?:нөмірі |№ ?)?(\d+)", r"запись \1"),
    (r"\bсурет (?:нөмірі |№ ?)?(\d+)", r"снимок \1"),
    (r"\b(a|b) үлгісі\w*|\bүлгі (a|b)\b", lambda m: "образец " + (m.group(1) or m.group(2))),
    (r"\bүлгі\w*", "образец"),
    (r"нейрондық желі алаң\w*|жұмыс алаңы\w*|ойын алаңы\w*|\bалаң\w*", "площадка"),
    (r"\bпрактикум\w*|код практикумы\w*", "практикум"),
    (r"\bтапсырма\w*", "задание"),
    (r"\bбинаризация\w*|\bбинарлау\w*", "бинаризация"),
    (r"\bнормалау\w*|\bнормаландыр\w*", "нормализация"),
    (r"қиылысу саны\w*", "число пересечений"),
    (r"\bверификация\w*", "верификация"),
    (r"толық шешім\w*", "полное решение"),
    (r"\bкеңес\w*", "подсказка"),
    (r"\bдеңгей\w*", "уровня"),
    (r"бастапқы код\w*(?: қайтар\w*)?|кодты қайтар\w*", "верни начальный код"),
    (r"\bкод\w*", "код"),
    (r"\bмысал\w*", "пример"),
    (r"\bшешім(?:ді|ін|і)?\b", "решение"),
    (r"\bтексер\w*|\bөткіз\w*", "проверь"),
    (r"\bтүсіндір\w*|\bтүсіндіріп бер\w*|\bаш(?:ып) түсіндір\w*", "объясни"),
    (r"нәтиже не (?:болды|білдіреді)\w*|не болды\b|не шықты\b|нені білдіреді\b|қалай түсіну\w*", "что получилось"),
    (r"\bнеге\b|\bне үшін\b|\bқалай\b", "почему"),
    (r"әдістерді салыстыр\w*|әдістер(?:ді)? салыстыру\w*|әдістерді салыстырып\w*|белгілерді салыстыр\w*", "сравни методы"),
    (r"детекторларды салыстыр\w*|локализаторларды салыстыр\w*", "сравни детекторы"),
    (r"әдістерді бағала\w*|әдістерді eer\w*|ең жақсы әдіс\w*", "оцени методы"),
    (r"(?:сапаны|сапасын|жүйені|дәлдікті|қателерді) (?:бағала\w*|есепте\w*|өлше\w*)", "оцени качество"),
    (r"\bбағала\w*|\bбаға бер\w*", "оцени"),
    (r"\bсалыстыр\w*", "сравни"),
    (r"келесі (?:кезең|қадам)\w*|\bарман қарай\b|\bәрі қарай\b|\bжалғастыр\w*", "следующий этап"),
    (r"алдыңғы (?:кезең|қадам)\w*|\bартқа\b|\bқайт\b", "предыдущий этап"),
    (r"соңғы (?:кезең|қадам)\w*", "последний этап"),
    (r"(?:кезең|қадам)\w* (?:нөмірі |№ ?)?(\d)\b", r"этап \1"),
    (r"\b(\d)[- ]?(?:інші|ші)?[- ]?(?:кезең|қадам)\w*", r"этап \1"),
    (r"\b(?:кезеңдер|қадамдар|өңдеу кезеңдері|өңдеу қадамдары)\w*|\bкезеңдерін\b", "этапы"),
    (r"\bшек\w*", "порог"),
    (r"\bарттыр\w*|\bкөтер\w*|\bүлкейт\w*|\bжоғарыла\w*", "увеличь"),
    (r"\bазайт\w*|\bтөмендет\w*|\bкішірейт\w*", "уменьши"),
    (r"(\d+(?:[.,]\d+)?)[- ](?:ге|ға|ке|қа)\b", r"на \1"),
    (r"\bбедер\w*|\bрельеф\w*", "рельеф"),
    (r"\bайнала\w*|\bшолу\b|\bайналып өт\w*", "облет"),
    (r"\bстресс[- ]?тест\w*|\bбұрмалау тест\w*", "стресс-тест"),
    (r"\bбұрмала\w*", "искажения"),
    (r"\bтөзімділік\w*", "устойчивость"),
    (r"\bстандарт\w*", "стандарты"),
    (r"\bлаб ?плюс\b|\blab ?плюс\b", "лаб плюс"),
    (r"\bсинтетика\w*", "синтетика"),
    (r"\bбаза(?:сы)?\b", "база"),
    (r"жергілікті құрылым\w*|локальді құрылым\w*|локальды құрылым\w*", "локальные структуры"),
    (r"фазалық корреляция\w*|фазалы корреляция\w*", "фазовая корреляция"),
    (r"дауыс беру\w*|дауыспен таңдау\w*", "голосование"),
    (r"\bлог[- ]?габор\w*", "лог-габор"),
    (r"\bхаар\w*|\bвейвлет\w*", "хаар"),
    (r"\bпиксель\w*", "пиксел"),
    (r"(?:сіз|өзіңіз) оқытқан (?:нейрондық )?желі\w*|оқытылған (?:нейрондық )?желі\w*", "нейросеть обученная вами"),
    (r"\bдельта\w*|\bдинамика\w*", "дельта"),
    (r"біріктірілген|\bкомбинир\w*", "комбинир"),
    (r"\bоқыт\w*|\bүйрет\w*", "обучение"),
    (r"\bіске қос\w*|\bорында\w*|\bжүгіртіп көр\w*|\bбаста\w*|\bқос\b", "запусти"),
    (r"\bдауысымды жаз\w*|\bдауысты жаз\w*|\bжазып ал\w*|\bжазыңыз\b|\bжазу\b|\bжаз\b", "запиши мой голос"),
    (r"\bтыңда\w*|\bойнат\w*|\bқайта ойнат\w*", "прослушай"),
    (r"\bтон биіктігі\w*", "высота тона"),
    (r"\bнегізгі тон\w*", "основной тон"),
    (r"\bдауыс\w*", "голос"),
    (r"\bбет(?:ті|ін|ке|те|тің)?\b", "лицо"),
    (r"\bнұрлы қабық\w*", "радужка"),
    (r"\bкөз(?:ді|дің|ге)?\b", "глаз"),
    (r"\bпроцент\w*|\bпайыз\w*", "процентов"),
    (r"\bүтір\b|\bнүкте\b|\bбүтін\b", "точка"),
    (r"\bжүз\b", "сто"),
]
_KK_NUM = {"нөл": "ноль", "бір": "один", "екі": "два", "үш": "три", "төрт": "четыре", "бес": "пять", "алты": "шесть", "жеті": "семь", "сегіз": "восемь",
           "тоғыз": "девять", "жиырма": "двадцать", "отыз": "тридцать", "қырық": "сорок", "елу": "пятьдесят", "алпыс": "шестьдесят", "жетпіс": "семьдесят",
           "сексен": "восемьдесят", "тоқсан": "девяносто"}


def _compile(rules):
    return [(re.compile(p), r) for p, r in rules]


_EN_RX, _KK_RX = _compile(EN), _compile(KK)
_EN_NUM = re.compile(r"\b(" + "|".join(_EN_UNITS) + r")\b")
_KK_NUMRX = re.compile(r"(?<!\w)(" + "|".join(_KK_NUM) + r")(?!\w)")
_LATIN = re.compile(r"[a-z]")
_KK_SPECIFIC = re.compile(r"[әіңғүұқөһ]|\b(?:аш|көрсет|салыстыр|тексер|түсіндір|баста|тоқта|шек|кезең|қадам|үлгі|дауыс|көмек|тапсырма|кеңес|нормалау|бағала|оқыт|жаз|тыңда|таңба|жазба|бедер|алаң)")


def to_ru(t: str) -> str:
    """Английские и казахские слова команды -> русские опорные слова. Русский текст возвращается без изменений."""
    if _LATIN.search(t):
        for rx, rep in _EN_RX:
            t = rx.sub(rep, t)
        t = _EN_NUM.sub(lambda m: _EN_UNITS[m.group(1)], t)
    if _KK_SPECIFIC.search(t):
        for rx, rep in _KK_RX:
            t = rx.sub(rep, t)
        t = _KK_NUMRX.sub(lambda m: _KK_NUM[m.group(1)], t)
    return re.sub(r"\s+", " ", t).strip()
