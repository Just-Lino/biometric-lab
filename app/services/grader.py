"""Автоматическая оценка по рубрике задания (веса задаёт JSON задания, преподаватель может их менять)."""


def grade(task: dict, result: dict, answer: int | None) -> dict:
    r = task["rubric"]
    tests = result.get("tests", [])
    basic = [t for t in tests if t["kind"] == "basic"]
    edge = [t for t in tests if t["kind"] == "edge"]
    pb, pe = sum(t["ok"] for t in basic), sum(t["ok"] for t in edge)
    run_ok = bool(result.get("defined")) and result.get("error") is None and bool(basic) and not any(t["error"] or t.get("returned_none") for t in basic)
    answer_ok = answer is not None and answer == task["question"]["correct"]
    parts = {
        "run": r["run"] if run_ok else 0.0,
        "basic": r["basic"] * pb / len(basic) if basic else 0.0,
        "edge": r["edge"] * pe / len(edge) if edge else 0.0,
        "question": r["question"] if answer_ok else 0.0,
    }
    return {"score": round(sum(parts.values()), 2), "max_score": float(sum(r.values())), "parts": {k: round(v, 2) for k, v in parts.items()},
            "passed_basic": pb, "total_basic": len(basic), "passed_edge": pe, "total_edge": len(edge),
            "run_ok": run_ok, "answer_ok": answer_ok, "answered": answer is not None}
