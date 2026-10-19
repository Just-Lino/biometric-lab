"""Маленькая свёрточная нейросеть на NumPy (без PyTorch): conv-relu-pool ×3 → embedding → softmax.
Нужна для учебной визуализации обучения: все веса, карты признаков и embeddings доступны напрямую."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _im2col(x: np.ndarray, k: int, pad: int) -> np.ndarray:
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (k, k), axis=(2, 3))  # B,C,H,W,k,k
    B, C, H, W = x.shape
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(B * H * W, C * k * k)


class Conv:
    def __init__(self, cin, cout, k, rng):
        self.k, self.cin, self.cout = k, cin, cout
        self.W = (rng.randn(cout, cin * k * k) * np.sqrt(2.0 / (cin * k * k))).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def forward(self, x):
        B, C, H, W = x.shape
        self.cols = _im2col(x, self.k, self.k // 2)
        self.shape = x.shape
        out = self.cols @ self.W.T + self.b
        return out.reshape(B, H, W, self.cout).transpose(0, 3, 1, 2)

    def backward(self, dout):
        B, C, H, W = self.shape
        d2 = dout.transpose(0, 2, 3, 1).reshape(-1, self.cout)
        self.dW = d2.T @ self.cols
        self.db = d2.sum(0)
        dcols = (d2 @ self.W).reshape(B, H, W, C, self.k, self.k)
        pad = self.k // 2
        dxp = np.zeros((B, C, H + 2 * pad, W + 2 * pad), np.float32)
        for i in range(self.k):
            for j in range(self.k):
                dxp[:, :, i:i + H, j:j + W] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, pad:pad + H, pad:pad + W]

    params = property(lambda self: [(self, "W", "dW"), (self, "b", "db")])


class Dense:
    def __init__(self, nin, nout, rng):
        self.W = (rng.randn(nin, nout) * np.sqrt(2.0 / nin)).astype(np.float32)
        self.b = np.zeros(nout, np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW, self.db = self.x.T @ d, d.sum(0)
        return d @ self.W.T

    params = property(lambda self: [(self, "W", "dW"), (self, "b", "db")])


def _pool(x):
    B, C, H, W = x.shape
    xr = x.reshape(B, C, H // 2, 2, W // 2, 2)
    out = xr.max(axis=(3, 5))
    mask = (xr == out[:, :, :, None, :, None]).astype(np.float32)
    mask /= mask.sum(axis=(3, 5), keepdims=True)
    return out, mask


class CNN:
    """Вход: (B,1,S,S), S кратно 8. Выход: logits и embedding."""

    def __init__(self, size=64, n_classes=12, emb=32, filters=(8, 16, 16), seed=0):
        rng = np.random.RandomState(seed)
        self.size, self.n_classes, self.emb_dim = size, n_classes, emb
        f1, f2, f3 = filters
        self.c1, self.c2, self.c3 = Conv(1, f1, 3, rng), Conv(f1, f2, 3, rng), Conv(f2, f3, 3, rng)
        flat = f3 * (size // 8) ** 2
        self.d1, self.d2 = Dense(flat, emb, rng), Dense(emb, n_classes, rng)
        self.layers = [self.c1, self.c2, self.c3, self.d1, self.d2]
        self._m = {}
        self._v = {}
        self._t = 0

    def forward(self, x, keep=False):
        a1 = np.maximum(self.c1.forward(x), 0); p1, m1 = _pool(a1)
        a2 = np.maximum(self.c2.forward(p1), 0); p2, m2 = _pool(a2)
        a3 = np.maximum(self.c3.forward(p2), 0); p3, m3 = _pool(a3)
        flat = p3.reshape(len(x), -1)
        e = np.maximum(self.d1.forward(flat), 0)
        logits = self.d2.forward(e)
        self.cache = (a1, m1, a2, m2, a3, m3, p3.shape, e)
        if keep:
            self.maps = {"conv1": p1, "conv2": p2, "conv3": p3}
        return logits, e

    def embed(self, x):
        out = []
        for i in range(0, len(x), 64):
            out.append(self.forward(x[i:i + 64])[1])
        return np.concatenate(out)

    def loss_grad(self, logits, y):
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
        d = p.copy(); d[np.arange(n), y] -= 1
        return loss, d / n, (p.argmax(1) == y).mean()

    def backward(self, dlogits):
        a1, m1, a2, m2, a3, m3, p3shape, e = self.cache
        d = self.d2.backward(dlogits)
        d = d * (e > 0)
        d = self.d1.backward(d).reshape(p3shape)
        d = self._unpool(d, m3) * (a3 > 0); d = self.c3.backward(d)
        d = self._unpool(d, m2) * (a2 > 0); d = self.c2.backward(d)
        d = self._unpool(d, m1) * (a1 > 0); self.c1.backward(d)

    @staticmethod
    def _unpool(d, mask):
        B, C, H, W = d.shape
        return (mask * d[:, :, :, None, :, None]).reshape(B, C, H * 2, W * 2)

    def step(self, x, y, lr):
        logits, _ = self.forward(x)
        loss, d, acc = self.loss_grad(logits, y)
        self.backward(d)
        self._t += 1
        for li, layer in enumerate(self.layers):
            for pi, (obj, pn, gn) in enumerate(layer.params):
                key = (li, pi)
                g = getattr(obj, gn)
                m = self._m.setdefault(key, np.zeros_like(g)); v = self._v.setdefault(key, np.zeros_like(g))
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                mh, vh = m / (1 - 0.9 ** self._t), v / (1 - 0.999 ** self._t)
                setattr(obj, pn, getattr(obj, pn) - lr * mh / (np.sqrt(vh) + 1e-8))
        return float(loss), float(acc)
