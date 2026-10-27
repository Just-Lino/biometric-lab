"""Копирует скачанные вручную наборы в datasets/real/, чтобы лаборатория видела их как источники данных.
Ничего не скачивает из интернета; исходные папки не меняются; повторный запуск пропускает уже скопированные файлы.

  python -m scripts.prepare_real --casia "C:\\Users\\<вы>\\Downloads\\casia" --soco "C:\\Users\\<вы>\\Downloads\\soco"

  --casia   папка с CASIA-Iris-Interval (внутри папки людей 001, 002 ... с подпапками L и R)
  --soco    папка с SOCOFing (внутри Real и Altered)
  --soco-people N   скопировать только первых N человек (по умолчанию все 600; для экономии места и времени)

Наборы FVC2002 и FVC2004 уже лежат в datasets/real/fvc2002 и datasets/real/fvc2004 (см. README). Условия использования
наборов: только исследования и обучение, не распространять."""
import argparse
import re
import shutil
import sys
from pathlib import Path

from app.core import realdata


def _find(base: Path, has: tuple[str, ...], depth: int = 3) -> Path | None:
    """Папка внутри base (до depth уровней вглубь), в которой есть все подпапки has."""
    if all((base / h).is_dir() for h in has):
        return base
    if depth > 0 and base.is_dir():
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            found = _find(sub, has, depth - 1)
            if found:
                return found
    return None


def _find_casia(base: Path) -> Path | None:
    """Корень CASIA: папка, где лежат папки людей (трёхзначные номера), в которых есть L или R."""
    if base.is_dir():
        people = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{3}", p.name)]
        if people and any((p / s).is_dir() for p in people for s in ("L", "R")):
            return base
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            f = _find_casia(sub) if sub.name != "__MACOSX" else None
            if f:
                return f
    return None


def _copy(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def prepare_casia(src: Path) -> int:
    root = _find_casia(src)
    if root is None:
        print(f"CASIA: в «{src}» не найдены папки людей (001, 002 ...) с подпапками L и R.")
        return 0
    out = realdata.subdir("casia_iris_interval")
    n = new = 0
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and re.fullmatch(r"\d{3}", f.parent.parent.name) and f.parent.name in ("L", "R"):
            n += 1
            new += _copy(f, out / f.parent.parent.name / f.parent.name / f.name)
    print(f"CASIA: файлов {n}, скопировано новых {new}  ->  {out}")
    return n


def prepare_soco(src: Path, people: int) -> int:
    root = _find(src, ("Real", "Altered"))
    if root is None:
        print(f"SOCOFing: в «{src}» не найдены папки Real и Altered.")
        return 0
    out = realdata.subdir("socofing")
    pids = sorted(p.name for p in (root / "Real").iterdir() if p.is_dir())
    if people:
        pids = pids[:people]
    keep = set(pids)
    n = new = 0
    for f in sorted(root.rglob("*.[bB][mM][pP]")):
        rel = f.relative_to(root)
        parts = rel.parts
        if parts[0] not in ("Real", "Altered"):
            continue
        pid = parts[1] if parts[0] == "Real" else (parts[2] if len(parts) > 3 else "")
        if pid not in keep:
            continue
        n += 1
        new += _copy(f, out / rel)
        if n % 5000 == 0:
            print(f"  SOCOFing: обработано {n} файлов ...", flush=True)
    print(f"SOCOFing: файлов {n} ({len(keep)} человек), скопировано новых {new}  ->  {out}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Копирование CASIA-Iris-Interval и SOCOFing в datasets/real/")
    ap.add_argument("--casia", type=Path, help="папка со скачанным CASIA-Iris-Interval")
    ap.add_argument("--soco", type=Path, help="папка со скачанным SOCOFing")
    ap.add_argument("--soco-people", type=int, default=0, help="сколько человек SOCOFing копировать (0 - всех)")
    a = ap.parse_args(argv)
    if not a.casia and not a.soco:
        ap.error("укажите --casia и/или --soco")
    total = 0
    if a.casia:
        total += prepare_casia(a.casia)
    if a.soco:
        total += prepare_soco(a.soco, a.soco_people)
    print("Готово. Перезапустите сервер, чтобы лаборатория увидела новые наборы." if total else "Ничего не скопировано.")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
