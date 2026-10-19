"""Подготовить галерею лиц из открытого набора LFW (Labeled Faces in the Wild).

    python -m scripts.prepare_lfw                    # 10 человек по 4 фото, архив скачается сам (~173 МБ)
    python -m scripts.prepare_lfw 12 5               # 12 человек по 5 фото
    python -m scripts.prepare_lfw 10 4 <источник>    # источник: .zip, .tgz или папка с набором

Если автоматическая загрузка не работает, скачайте LFW сами (например, с Kaggle:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset, нужен бесплатный аккаунт),
затем укажите третьим аргументом путь к скачанному .zip или к распакованной папке.
Подойдёт любая версия набора: внутри должны быть папки людей с их фотографиями.

LFW - открытый исследовательский набор. Используйте его только в учебных и научных целях."""
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

from app.modules.face.module import GALLERY

URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
DEFAULT_ARCHIVE = Path("data") / "lfw.tgz"
MIN_SIZE = 100_000_000  # настоящий архив ~173 МБ


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    print(f"Скачиваю {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "biometric-lab"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = f" ({done * 100 // total}%)" if total else ""
            print(f"\r  {done / 1e6:.0f} МБ{pct}, {done / 1e6 / max(time.time() - t0, 1e-6):.1f} МБ/с   ", end="", flush=True)
    print()
    part.replace(dest)


def _person(name: str) -> str:
    return PurePosixPath(name).parent.name


def prepare(archive: Path, n_people: int, per_person: int) -> list[str]:
    print("Проход 1: подсчёт фото по людям (около минуты)...")
    counts: Counter = Counter()
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".jpg"):
                counts[_person(m.name)] += 1
    chosen = [p for p, c in counts.most_common() if c >= per_person][:n_people]
    if not chosen:
        raise SystemExit("В архиве не нашлось людей с нужным числом фото. Уменьшите второй аргумент.")
    print("Выбраны:", ", ".join(chosen))

    print("Проход 2: копирую фотографии...")
    taken: Counter = Counter()
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.lower().endswith(".jpg")):
                continue
            person = _person(m.name)
            if person in chosen and taken[person] < per_person:
                taken[person] += 1
                folder = GALLERY / person
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{taken[person]:02d}.jpg").write_bytes(tf.extractfile(m).read())
            if all(taken[p] >= per_person for p in chosen):
                break
    return chosen


def _group(paths: list[str]) -> list[tuple[str, list[str]]]:
    """Группировка файлов по папкам (папка = человек); уникальные имена, больше всего фото вперёд."""
    groups: dict[str, list[str]] = {}
    for p in paths:
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            groups.setdefault(str(PurePosixPath(p).parent), []).append(p)
    seen, out = set(), []
    for parent, files in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        name = PurePosixPath(parent).name
        if name and name not in seen:
            seen.add(name)
            out.append((name, sorted(files)))
    return out


def _save(person: str, k: int, data: bytes, suffix: str) -> None:
    folder = GALLERY / person
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{k:02d}{suffix}").write_bytes(data)


def prepare_from_zip(archive: Path, n_people: int, per_person: int) -> list[str]:
    import zipfile
    with zipfile.ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        groups = [(n, f) for n, f in _group(names) if len(f) >= per_person][:n_people]
        for person, files in groups:
            for k, f in enumerate(files[:per_person], 1):
                _save(person, k, z.read(f), Path(f).suffix.lower())
            print(f"  {person}: готово")
    return [g[0] for g in groups]


def prepare_from_folder(root: Path, n_people: int, per_person: int) -> list[str]:
    files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    groups = [(n, f) for n, f in _group(files) if len(f) >= per_person][:n_people]
    for person, fl in groups:
        for k, f in enumerate(fl[:per_person], 1):
            _save(person, k, (root / f).read_bytes(), Path(f).suffix.lower())
        print(f"  {person}: готово")
    return [g[0] for g in groups]


def prepare_via_sklearn(n_people: int, per_person: int) -> list[str]:
    """Запасной источник: LFW через scikit-learn (скачивается с figshare, около 200 МБ, нужен Pillow)."""
    import cv2
    import numpy as np
    from sklearn.datasets import fetch_lfw_people

    print("Пробую запасной источник (scikit-learn / figshare), это может занять несколько минут...")
    data = fetch_lfw_people(min_faces_per_person=per_person, resize=1.0, color=True,
                            slice_=(slice(0, 250), slice(0, 250)), download_if_missing=True)
    names = data.target_names[data.target]
    uniq, counts = np.unique(names, return_counts=True)
    chosen = [str(n) for n in uniq[np.argsort(-counts)][:n_people]]
    for name in chosen:
        folder = GALLERY / name.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        for k, img in enumerate(data.images[names == name][:per_person], 1):
            img = np.asarray(img, np.float32)
            if img.max() <= 1.5:  # в разных версиях масштаб 0..1 или 0..255
                img = img * 255
            bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr)
            (folder / f"{k:02d}.jpg").write_bytes(buf.tobytes())
    return chosen


def main() -> None:
    args = sys.argv[1:]
    n_people = int(args[0]) if len(args) > 0 else 10
    per_person = int(args[1]) if len(args) > 1 else 4
    archive = Path(args[2]) if len(args) > 2 else DEFAULT_ARCHIVE
    if len(args) > 2:
        if not archive.exists():
            raise SystemExit(f"Не найден источник: {archive}")
        if archive.is_dir():
            chosen = prepare_from_folder(archive, n_people, per_person)
        elif archive.suffix.lower() == ".zip":
            chosen = prepare_from_zip(archive, n_people, per_person)
        else:
            chosen = prepare(archive, n_people, per_person)
        if not chosen:
            raise SystemExit("Не нашлось людей с нужным числом фото. Уменьшите второй аргумент или проверьте источник.")
        print(f"Готово: {len(chosen)} человек по {per_person} фото в папке {GALLERY}")
        return
    if not archive.exists() or archive.stat().st_size < MIN_SIZE:
        try:
            download(URL, archive)
        except Exception as e:
            print(f"\nОсновной сайт недоступен: {e}")
            try:
                chosen = prepare_via_sklearn(n_people, per_person)
                print(f"Готово: {len(chosen)} человек по {per_person} фото в папке {GALLERY}")
                return
            except Exception as e2:
                raise SystemExit(f"\nЗапасной источник тоже не сработал: {e2}\n"
                                 "Скачайте LFW сами (например, Kaggle: jessicali9530/lfw-dataset) и запустите:\n"
                                 "    python -m scripts.prepare_lfw 10 4 <путь_к_zip_или_папке>\n"
                                 "Либо положите фото вручную в datasets/faces/<имя>/ (минимум 2 человека по 2 фото).")
    chosen = prepare(archive, n_people, per_person)
    print(f"Готово: {len(chosen)} человек по {per_person} фото в папке {GALLERY}")


if __name__ == "__main__":
    main()
