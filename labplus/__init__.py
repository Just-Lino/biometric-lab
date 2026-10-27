"""LAB+ — дополнение к виртуальной лаборатории «Биометрия».

Подключение (одна строка в вашем main.py):

    from labplus import router as labplus_router
    app.include_router(labplus_router)

Страницы: /labplus/voice, /labplus/metrics, /labplus/robustness, /labplus/standards
API:      /api/labplus/...
"""
from .router import router  # noqa: F401

__version__ = "1.0.0"
