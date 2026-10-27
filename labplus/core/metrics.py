"""Метрики биометрической верификации (терминология ISO/IEC 19795-1 / 2382-37):
FMR, FNMR, EER, ROC, DET, AUC, d′, FNMR при заданном FMR, доверительный интервал EER."""
from __future__ import annotations

import math

import numpy as np

MAX_SCORES = 200_000


class MetricsError(ValueError):
    pass


# ───────────────────────── обратная функция нормального распределения ─────────────────────────
def probit(p):
    """Φ⁻¹(p) (алгоритм Acklam, точность ~1e-9). Для DET-оси."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    out = np.empty_like(p)
    lo, hi = 0.02425, 1 - 0.02425
    m = p < lo
    q = np.sqrt(-2 * np.log(p[m]))
    out[m] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5
    r = q * q
    out[m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    m = p > hi
    q = np.sqrt(-2 * np.log(1 - p[m]))
    out[m] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out


# ───────────────────────── данные ─────────────────────────
def validate(genuine, impostor):
    g = np.asarray(genuine, dtype=float).ravel()
    i = np.asarray(impostor, dtype=float).ravel()
    if len(g) < 5 or len(i) < 5:
        raise MetricsError("Нужно минимум по 5 оценок «своих» и «чужих».")
    if len(g) > MAX_SCORES or len(i) > MAX_SCORES:
        raise MetricsError(f"Слишком много оценок (максимум {MAX_SCORES} в каждой группе).")
    if not (np.isfinite(g).all() and np.isfinite(i).all()):
        raise MetricsError("В оценках есть нечисловые значения.")
    return g, i


PRESETS = {
    "good": dict(mu_g=0.82, sd_g=0.08, mu_i=0.25, sd_i=0.10),
    "medium": dict(mu_g=0.70, sd_g=0.12, mu_i=0.38, sd_i=0.14),
    "poor": dict(mu_g=0.60, sd_g=0.16, mu_i=0.45, sd_i=0.16),
}


def generate_scores(n_g=500, n_i=5000, mu_g=0.75, sd_g=0.1, mu_i=0.3, sd_i=0.12, seed=1):
    if not (5 <= n_g <= 50_000 and 5 <= n_i <= 100_000):
        raise MetricsError("Размер выборки вне допустимых пределов.")
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(mu_g, max(sd_g, 1e-3), n_g), 0, 1)
    i = np.clip(rng.normal(mu_i, max(sd_i, 1e-3), n_i), 0, 1)
    return g, i


def parse_scores_text(text: str):
    """Строки вида «genuine,0.91» / «impostor 0.22» / «1;0.9» / «чужой,0.3»."""
    g, i = [], []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.lower().startswith(("label", "тип", "#")):
            continue
        parts = [p for p in line.replace(";", ",").replace("\t", ",").replace(" ", ",").split(",") if p]
        if len(parts) != 2:
            raise MetricsError(f"Строка {ln}: нужен формат «метка,оценка».")
        lab, val = parts[0].lower(), parts[1]
        try:
            v = float(val.replace(",", "."))
        except ValueError:
            raise MetricsError(f"Строка {ln}: «{val}» — не число.")
        if lab in ("genuine", "g", "1", "свой", "своя", "match", "true"):
            g.append(v)
        elif lab in ("impostor", "i", "0", "чужой", "чужая", "nonmatch", "false"):
            i.append(v)
        else:
            raise MetricsError(f"Строка {ln}: метка «{parts[0]}» не распознана (genuine/impostor).")
    return np.array(g), np.array(i)


# ───────────────────────── расчёты ─────────────────────────
def rates(g, i, t):
    """Решение: принять, если оценка ≥ порога."""
    fnmr = float((g < t).mean())
    fmr = float((i >= t).mean())
    return fmr, fnmr


def eer(g, i):
    """EER по всем уникальным значениям с линейной интерполяцией в точке пересечения."""
    ts = np.unique(np.concatenate([g, i]))
    ts = np.concatenate([[ts[0] - 1e-9], ts, [ts[-1] + 1e-9]])
    gs, is_ = np.sort(g), np.sort(i)
    fnmr = np.searchsorted(gs, ts, side="left") / len(g)
    fmr = 1 - np.searchsorted(is_, ts, side="left") / len(i)
    diff = fmr - fnmr                       # убывает по порогу, от + к −
    k = int(np.argmax(diff <= 0))
    if k == 0:
        return float((fmr[0] + fnmr[0]) / 2), float(ts[0])
    d0, d1 = diff[k - 1], diff[k]
    w = d0 / (d0 - d1) if d0 != d1 else 0.5
    t = ts[k - 1] + w * (ts[k] - ts[k - 1])
    e = (fmr[k - 1] + w * (fmr[k] - fmr[k - 1]) + fnmr[k - 1] + w * (fnmr[k] - fnmr[k - 1])) / 2
    return float(e), float(t)


def auc(g, i):
    """P(g > i) + 0.5·P(g = i) через ранги."""
    is_ = np.sort(i)
    lo = np.searchsorted(is_, g, side="left")
    hi = np.searchsorted(is_, g, side="right")
    return float(((lo + 0.5 * (hi - lo)) / len(i)).mean())


def dprime(g, i):
    v = 0.5 * (g.var() + i.var())
    return float((g.mean() - i.mean()) / math.sqrt(v)) if v > 0 else float("inf")


def fnmr_at_fmr(g, i, target):
    """Минимальный FNMR при FMR ≤ target; reliable — достаточно ли чужих сравнений (правило трёх)."""
    ts = np.unique(np.concatenate([g, i, [g.max() + 1e-9]]))
    is_ = np.sort(i)
    fmr = 1 - np.searchsorted(is_, ts, side="left") / len(i)
    ok = fmr <= target
    t = float(ts[np.argmax(ok)]) if ok.any() else float(ts[-1])
    fm, fn = rates(g, i, t)
    return {"target_fmr": target, "threshold": round(t, 4), "fmr": fm, "fnmr": fn,
            "reliable": bool(len(i) * target >= 3),
            "need_impostors": int(math.ceil(3 / target))}


def bootstrap_eer_ci(g, i, n=200, seed=7, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(eer(rng.choice(g, len(g)), rng.choice(i, len(i)))[0])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def analyze(genuine, impostor, n_points=201, ci=True):
    g, i = validate(genuine, impostor)
    lo, hi = float(min(g.min(), i.min())), float(max(g.max(), i.max()))
    if hi - lo < 1e-9:
        raise MetricsError("Все оценки одинаковы, кривые построить нельзя.")
    pad = 0.01 * (hi - lo)
    ts = np.linspace(lo - pad, hi + pad, n_points)
    gs, is_ = np.sort(g), np.sort(i)
    fnmr = np.searchsorted(gs, ts, side="left") / len(g)
    fmr = 1 - np.searchsorted(is_, ts, side="left") / len(i)
    tp, fn = (1 - fnmr) * len(g), fnmr * len(g)
    fp, tn = fmr * len(i), (1 - fmr) * len(i)
    acc = (tp + tn) / (len(g) + len(i))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        rec = tp / len(g)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    e, et = eer(g, i)
    lo_ci, hi_ci = bootstrap_eer_ci(g, i) if ci and len(g) * len(i) <= 5e7 else (None, None)
    n_det = len(g) + len(i)
    clipf = 0.5 / min(len(g), len(i))
    det_x = probit(np.clip(fmr, clipf, 1 - clipf))
    det_y = probit(np.clip(fnmr, clipf, 1 - clipf))
    return {
        "n_genuine": int(len(g)), "n_impostor": int(len(i)),
        "thresholds": ts.round(5).tolist(),
        "fmr": fmr.round(6).tolist(), "fnmr": fnmr.round(6).tolist(),
        "gar": (1 - fnmr).round(6).tolist(), "accuracy": acc.round(6).tolist(),
        "precision": np.asarray(prec).round(6).tolist(), "recall": np.asarray(rec).round(6).tolist(),
        "f1": np.asarray(f1).round(6).tolist(),
        "det_x": det_x.round(4).tolist(), "det_y": det_y.round(4).tolist(),
        "eer": round(e, 5), "eer_threshold": round(et, 5), "eer_ci": [lo_ci, hi_ci],
        "auc": round(auc(g, i), 5), "dprime": round(dprime(g, i), 3),
        "ops": [fnmr_at_fmr(g, i, t) for t in (1e-1, 1e-2, 1e-3)],
        "hist": histogram(g, i, lo - pad, hi + pad),
        "range": [lo - pad, hi + pad],
        "mean_g": float(g.mean()), "mean_i": float(i.mean()), "n_total": int(n_det),
    }


def histogram(g, i, lo, hi, bins=40):
    edges = np.linspace(lo, hi, bins + 1)
    hg, _ = np.histogram(g, edges)
    hi_, _ = np.histogram(i, edges)
    return {"edges": edges.round(4).tolist(), "genuine": (hg / len(g)).round(5).tolist(),
            "impostor": (hi_ / len(i)).round(5).tolist()}
