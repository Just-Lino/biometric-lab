"""Нагрузочный тест: 20-30 студентов работают одновременно.

Запуск ТОЛЬКО против локального сервера (временная база, TUTOR_PROVIDER=rule):
    uvicorn app.main:app
    locust -f scripts/loadtest/locustfile.py --headless -u 30 -r 5 -t 2m --host http://127.0.0.1:8000
"""
import random

from locust import HttpUser, between, task

BASE = "/api/v1"
FINGERS = [f"f{n:02d}_i{k}" for n in range(1, 13) for k in (1, 2, 3)]
SOLUTION_BAD = "def binarize(gray, threshold):\n    return gray\n"


class Student(HttpUser):
    wait_time = between(1, 4)

    def on_start(self):
        self.client.get("/")                                    # открыть страницу, получить cookie сессии
        faces = self.client.get(f"{BASE}/biometrics/face/samples?limit=40").json()
        self.faces = [s["id"] for s in faces] if isinstance(faces, list) else []

    @task(5)
    def fingerprint(self):
        a, b = random.sample(FINGERS, 2)
        self.client.post(f"{BASE}/biometrics/fingerprint/process", json={"sample_id": a}, name="fingerprint/process")
        self.client.post(f"{BASE}/biometrics/fingerprint/compare", json={"sample_a": a, "sample_b": b}, name="fingerprint/compare")

    @task(3)
    def face(self):
        if len(self.faces) >= 2:
            a, b = random.sample(self.faces, 2)
            self.client.post(f"{BASE}/biometrics/face/compare", json={"sample_a": a, "sample_b": b}, name="face/compare")

    @task(2)
    def practicum(self):
        # 503 «Сервер занят» при перегрузке песочницы - ожидаемый ответ, а не ошибка сервера
        with self.client.post(f"{BASE}/code/run", json={"task_id": "binarize", "code": SOLUTION_BAD}, name="code/run", catch_response=True) as r:
            if r.status_code in (200, 503):
                r.success()

    @task(1)
    def tutor(self):
        self.client.post(f"{BASE}/tutor/ask", json={"mode": "explain", "question": "что такое EER", "context": {}}, name="tutor/ask")
