from app.modules.base import BiometricModule
from app.modules.face.module import FaceModule
from app.modules.fingerprint.module import FingerprintModule
from app.modules.iris.module import IrisModule
from app.modules.voice.module import VoiceModule

_MODULES: dict[str, BiometricModule] = {}


def register(module: BiometricModule) -> None:
    _MODULES[module.name] = module


def get_module(name: str) -> BiometricModule:
    if name not in _MODULES:
        raise KeyError(name)
    return _MODULES[name]


def list_modules() -> list[dict]:
    return [{"name": m.name, "title": m.title, "default_threshold": m.default_threshold} for m in _MODULES.values()]


register(FingerprintModule())
register(FaceModule())
register(IrisModule())
register(VoiceModule())
