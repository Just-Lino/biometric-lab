"""Скачать Pyodide (Python в браузере) в app/web/vendor/pyodide/ для запуска примеров практикума в Web Worker.

    python -m scripts.download_pyodide                  # ядро + numpy, scipy, scikit-image (нужен интернет, около 60 МБ)
    python -m scripts.download_pyodide --with-opencv    # дополнительно opencv-python (ещё около 10 МБ)
    python -m scripts.download_pyodide --force          # скачать заново, даже если файлы уже есть

Версия зафиксирована в PYODIDE_VERSION (Python 3.14 в Pyodide; на сервере лабораторию запускают на 3.11-3.13).
Ядро берётся из npm-пакета pyodide (целостность проверяется по sha512 из реестра npm), колёса пакетов - с CDN jsDelivr
(целостность проверяется по sha256 из pyodide-lock.json). Файлы в Git не попадают: каталог app/web/vendor/ в .gitignore.
Если файлов нет, практикум работает как раньше: пример выполняется на сервере (/api/v1/code/run)."""
import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

PYODIDE_VERSION = "314.0.7"
NPM_META = f"https://registry.npmjs.org/pyodide/{PYODIDE_VERSION}"
CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
TARGET = Path(__file__).resolve().parents[1] / "app" / "web" / "vendor" / "pyodide"
CORE_FILES = ["pyodide.js", "pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json", "package.json"]
BASE_PACKAGES = ["numpy", "scipy", "scikit-image"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "biometric-lab-download-pyodide"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_core(force: bool) -> None:
    if not force and all((TARGET / f).exists() for f in CORE_FILES):
        print("ядро Pyodide уже скачано")
        return
    meta = json.loads(fetch(NPM_META))
    integrity = meta["dist"]["integrity"]                       # sha512-<base64>
    algo, _, expected = integrity.partition("-")
    print(f"скачиваю npm-пакет pyodide@{PYODIDE_VERSION} ...")
    blob = fetch(meta["dist"]["tarball"])
    if algo != "sha512" or base64.b64encode(hashlib.sha512(blob).digest()).decode() != expected:
        sys.exit("Ошибка: контрольная сумма npm-пакета не совпала, файлы не сохранены.")
    TARGET.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            name = m.name.split("/", 1)[-1]                      # package/pyodide.js -> pyodide.js
            if m.isfile() and name in CORE_FILES:                # только известные файлы: произвольные пути из архива не извлекаем
                (TARGET / name).write_bytes(tar.extractfile(m).read())
                found.add(name)
    missing = set(CORE_FILES) - found
    if missing:
        sys.exit(f"В npm-пакете не найдены файлы: {', '.join(sorted(missing))}")
    print("ядро Pyodide готово:", ", ".join(sorted(found)))


def resolve(lock: dict, names: list[str]) -> list[str]:
    """Пакеты и все их зависимости по pyodide-lock.json."""
    pk = lock["packages"]
    out, stack = [], list(names)
    while stack:
        n = stack.pop().lower().replace("_", "-")
        key = next((k for k in pk if k.lower().replace("_", "-") == n), None)
        if key is None:
            sys.exit(f"Пакета «{n}» нет в pyodide-lock.json версии {PYODIDE_VERSION}")
        if key not in out:
            out.append(key)
            stack.extend(pk[key].get("depends", []))
    return out


def download_packages(names: list[str], force: bool) -> list[str]:
    lock = json.loads((TARGET / "pyodide-lock.json").read_text(encoding="utf-8"))
    todo = resolve(lock, names)
    total = 0
    for key in todo:
        info = lock["packages"][key]
        f = TARGET / info["file_name"]
        if not force and f.exists() and sha256(f.read_bytes()) == info["sha256"]:
            continue
        print(f"  {key} ({info['file_name']}) ...", flush=True)
        data = fetch(CDN + info["file_name"])
        if sha256(data) != info["sha256"]:
            sys.exit(f"Ошибка: контрольная сумма {info['file_name']} не совпала, файл не сохранён.")
        f.write_bytes(data)
        total += len(data)
    print(f"пакеты готовы: {', '.join(todo)} (скачано {total / 1e6:.1f} МБ)")
    return todo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-opencv", action="store_true", help="добавить opencv-python")
    ap.add_argument("--force", action="store_true", help="скачать заново")
    a = ap.parse_args()
    print("Папка:", TARGET)
    download_core(a.force)
    pkgs = BASE_PACKAGES + (["opencv-python"] if a.with_opencv else [])
    done = download_packages(pkgs, a.force)
    (TARGET / "lab-packages.json").write_text(json.dumps({"version": PYODIDE_VERSION, "packages": done}, ensure_ascii=False), encoding="utf-8")
    print("Готово. Версия Pyodide:", PYODIDE_VERSION)
