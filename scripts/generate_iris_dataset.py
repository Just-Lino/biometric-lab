"""Сгенерировать синтетический набор глаз: datasets/synthetic_iris/eNN_iK.png и truth.json (истинные окружности).
   python -m scripts.generate_iris_dataset"""
import json

import cv2

from app.modules.iris import synthetic


def main():
    synthetic.DATA.mkdir(parents=True, exist_ok=True)
    truth = {}
    for e in range(1, synthetic.N_EYES + 1):
        for i in range(1, synthetic.N_IMPRESSIONS + 1):
            img, t = synthetic.render(e, i)
            key = f"e{e:02d}_i{i}"
            cv2.imwrite(str(synthetic.DATA / f"{key}.png"), img)
            truth[key] = t
    (synthetic.DATA / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {len(truth)} снимков в {synthetic.DATA}")


if __name__ == "__main__":
    main()
