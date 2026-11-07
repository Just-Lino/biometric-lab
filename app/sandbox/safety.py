"""Статическая проверка кода студента и урезанные builtins.

ВАЖНО: это учебная защита «по мере сил», а не промышленная изоляция. Для боевого сервера запускайте
выполнение в Docker (--network none, --memory, --read-only, пользователь без прав)."""
import ast
import builtins
import sys

ALLOWED_IMPORTS = {"math", "numpy", "cv2", "skimage", "itertools", "functools", "collections", "statistics",
                   "typing", "operator", "random", "fractions", "decimal"}
BANNED_SUBMODULES = {"ctypeslib", "f2py", "testing", "distutils", "_core", "core", "lib", "io", "data"}  # skimage.io пишет файлы, skimage.data качает из сети
BANNED_NAMES = {"eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview", "dir"}
BANNED_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "save", "savez", "savez_compressed", "savetxt",
                "tofile", "dump", "dumps", "imread", "imwrite", "VideoCapture", "system", "popen", "ctypeslib",
                "f2py", "testing", "distutils", "fromregex", "memmap",
                # через модули-обёртки (cv2.os, statistics.sys ...) можно добраться до файловой системы и процессов
                "os", "sys", "importlib", "modules", "subprocess", "builtins", "environ", "startfile", "execv", "spawnv",
                # доступ к атрибутам по строке в обход проверки имён
                "attrgetter", "methodcaller",
                # OpenCV и scikit-image: запись и чтение файлов, загрузка моделей
                "imsave", "imread_collection", "FileStorage", "VideoWriter", "dnn", "CascadeClassifier", "readNet",
                "readNetFromONNX", "readNetFromCaffe", "readNetFromTensorflow", "readNetFromDarknet", "readNetFromTorch"}


# Служебные модули numpy, которые сам numpy подгружает при a.mean(), a.std(), a.var(), a.clip() и т.п. Это C-код: он вызывает
# __import__ из builtins текущего кода, то есть из ограниченных builtins песочницы. Пока студент не написал такой импорт сам
# (check_code его запрещает: «_core»/«core» в BANNED_SUBMODULES), пропускаем только уже загруженные модули, новый код не выполняется.
NUMPY_INTERNAL = {"numpy._core._methods", "numpy._core._internal", "numpy._core._exceptions", "numpy._core._dtype",
                  "numpy.core._methods", "numpy.core._internal", "numpy.core._exceptions", "numpy.core._dtype"}


class SafetyError(Exception):
    def __init__(self, message: str, line: int | None = None):
        super().__init__(message)
        self.line = line


def check_code(code: str) -> None:
    """Бросает SyntaxError или SafetyError, если код нельзя выполнять."""
    tree = ast.parse(code, "<student>")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Import):
            for a in node.names:
                _check_module(a.name, line)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise SafetyError("Относительные импорты запрещены.", line)
            _check_module(node.module or "", line)
            for a in node.names:  # from numpy import load, from cv2 import os - те же запреты, что и для атрибутов
                if a.name in BANNED_ATTRS or a.name in BANNED_NAMES or a.name.startswith("__"):
                    raise SafetyError(f"Импорт «{a.name}» запрещён (доступ к файлам и системе).", line)
                _check_module(f"{node.module}.{a.name}", line)
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SafetyError(f"Использование «{node.id}» запрещено в учебной песочнице.", line)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SafetyError("Обращение к служебным атрибутам (__...__) запрещено.", line)
            if node.attr in BANNED_ATTRS:
                raise SafetyError(f"Атрибут «{node.attr}» запрещён (доступ к файлам и системе).", line)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise SafetyError("Строки с двойным подчёркиванием (__) запрещены.", line)


def _check_module(name: str, line):
    parts = name.split(".")
    if parts[0] not in ALLOWED_IMPORTS:
        raise SafetyError(f"Импорт модуля «{parts[0]}» запрещён. Разрешены: {', '.join(sorted(ALLOWED_IMPORTS))}.", line)
    if any(p in BANNED_SUBMODULES for p in parts[1:]):
        raise SafetyError(f"Импорт «{name}» запрещён.", line)


def safe_builtins() -> dict:
    names = ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance",
             "iter", "len", "list", "map", "max", "min", "next", "pow", "print", "range", "reversed", "round",
             "set", "slice", "sorted", "str", "sum", "tuple", "zip", "repr", "chr", "ord", "format", "hash",
             "frozenset", "complex", "bytes", "object", "type", "callable", "hex", "bin", "oct", "ascii",
             "ValueError", "TypeError", "Exception", "ZeroDivisionError", "IndexError", "KeyError",
             "StopIteration", "AssertionError", "NotImplementedError", "ArithmeticError", "OverflowError",
             "RuntimeError", "AttributeError", "NameError", "True", "False", "None", "__build_class__"]
    b = {n: getattr(builtins, n) for n in names if hasattr(builtins, n)}

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in NUMPY_INTERNAL and name in sys.modules:
            return builtins.__import__(name, globals, locals, fromlist, level)
        _check_module(name, None)
        return builtins.__import__(name, globals, locals, fromlist, level)

    b["__import__"] = _import
    b["__name__"] = "student"
    return b
