"""Скачать русские голоса Piper в models/tts/ (по 63 МБ): Дмитрий и Денис (набор данных CC0, репозиторий rhasspy/piper-voices).
   python -m scripts.download_tts            (нужен интернет; без голосов помощник озвучивает голосом браузера)
Движок Piper ставится отдельно: pip install -r requirements-tts.txt (лицензия движка GPL-3.0-or-later)."""
import sys
import urllib.request

from app.services.tts import VOICES, TTS_DIR

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/{name}/medium/ru_RU-{name}-medium"


def fetch(url: str, dst, min_bytes: int) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "biometric-lab"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size < min_bytes:
            raise ValueError("файл слишком мал (возможно, это не модель)")
        tmp.replace(dst)
        return True
    except Exception as e:  # сеть недоступна и т. п.
        print(f"  не удалось скачать {dst.name}: {e}")
        tmp.unlink(missing_ok=True)
        return False


if __name__ == "__main__":
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for v in VOICES:
        base = BASE.format(name=v["id"])
        model, cfg = TTS_DIR / f"ru_RU-{v['id']}-medium.onnx", TTS_DIR / f"ru_RU-{v['id']}-medium.onnx.json"
        if model.exists() and model.stat().st_size > 10_000_000 and cfg.exists():
            print(f"{v['title']}: уже скачан")
            continue
        print(f"{v['title']}: скачиваю (около 63 МБ)...")
        done = fetch(base + ".onnx.json", cfg, 1000) and fetch(base + ".onnx", model, 10_000_000)
        print(f"{v['title']}: {'готово' if done else 'НЕ СКАЧАНО (проверьте интернет)'}")
        ok = ok and done
    sys.exit(0 if ok else 1)
