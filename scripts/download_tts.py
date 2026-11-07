"""Скачать голоса Piper в models/tts/ (репозиторий rhasspy/piper-voices): русские Дмитрий и Денис (по 63 МБ, набор данных CC0)
и казахская модель issai (122 МБ, четыре диктора, корпус ISSAI KazakhTTS).
   python -m scripts.download_tts                (русские; нужен интернет; без голосов помощник озвучивает голосом браузера)
   python -m scripts.download_tts --lang kk      (казахские)    --lang all  (все)
Движок Piper ставится отдельно: pip install -r requirements-tts.txt (лицензия движка GPL-3.0-or-later)."""
import sys
import urllib.request

from app.services.tts import VOICES, TTS_DIR

BASE_KK = "https://huggingface.co/rhasspy/piper-voices/resolve/main/kk/kk_KZ/issai/high/kk_KZ-issai-high"
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
    langs = {"ru"} if "--lang" not in sys.argv else set(sys.argv[sys.argv.index("--lang") + 1].replace("all", "ru,kk").split(","))
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for stem in dict.fromkeys(v["stem"] for v in VOICES if v["lang"] in langs):
        model, cfg = TTS_DIR / f"{stem}.onnx", TTS_DIR / f"{stem}.onnx.json"
        if model.exists() and model.stat().st_size > 10_000_000 and cfg.exists():
            print(f"{stem}: уже скачан")
            continue
        base = BASE_KK if stem.startswith("kk_") else BASE.format(name=stem.split("-")[1])
        size = "около 122 МБ" if stem.startswith("kk_") else "около 63 МБ"
        print(f"{stem}: скачиваю ({size})...")
        done = fetch(base + ".onnx.json", cfg, 1000) and fetch(base + ".onnx", model, 10_000_000)
        print(f"{stem}: {'готово' if done else 'НЕ СКАЧАНО (проверьте интернет)'}")
        ok = ok and done
    sys.exit(0 if ok else 1)
