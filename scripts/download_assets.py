"""Скачать Plotly и Three.js в app/web/vendor: лаборатория станет работать без интернета (кроме редактора кода Monaco).
   python -m scripts.download_assets"""
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "app" / "web" / "vendor"
FILES = {
    "plotly.min.js": ("https://cdn.plot.ly/plotly-2.35.2.min.js", 1_000_000),
    "three.min.js": ("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js", 200_000),
}

if __name__ == "__main__":
    VENDOR.mkdir(parents=True, exist_ok=True)
    for name, (url, min_size) in FILES.items():
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "biometric-lab"}), timeout=60) as r:
                data = r.read()
            if len(data) < min_size:
                raise ValueError("файл слишком мал")
            (VENDOR / name).write_bytes(data)
            print(f"{name}: готово ({len(data) // 1024} КБ)")
        except Exception as e:
            print(f"{name}: НЕ СКАЧАНО ({e}). Проверьте интернет.")
    print("Перезапустите сервер: страница будет брать библиотеки из app/web/vendor.")
