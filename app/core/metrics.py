"""Кривые FAR/FRR и EER по оценкам сходства «своих» и «чужих» пар (одинаково для всех биометрий)."""
import numpy as np


def error_curves(genuine, impostor) -> dict:
    th = np.round(np.arange(0, 1.0001, 0.01), 2)
    g, im = np.asarray(genuine, float), np.asarray(impostor, float)
    frr = [float((g < t).mean()) for t in th]
    far = [float((im >= t).mean()) for t in th]
    k = int(np.argmin(np.abs(np.array(far) - np.array(frr))))
    return {"thresholds": th.tolist(), "far": far, "frr": frr, "eer": round((far[k] + frr[k]) / 2, 4), "eer_threshold": float(th[k]),
            "genuine_scores": [round(float(s), 4) for s in g], "impostor_scores": [round(float(s), 4) for s in im],
            "n_genuine": len(g), "n_impostor": len(im)}
