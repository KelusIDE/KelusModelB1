"""Reverse-mode automatic differentiation, built on NumPy array arithmetic.

No automatic-differentiation or neural-network library is used.
Graphs are consumed on backward to release intermediate activations promptly.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np

_tracking = ContextVar("kelus_grad_tracking", default=True)


@contextmanager
def no_grad():
    token = _tracking.set(False)
    try:
        yield
    finally:
        _tracking.reset(token)


def _unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad=False, parents=()):
        self.data = np.asarray(data)
        if self.data.dtype.kind != "f":
            self.data = self.data.astype(np.float32)
        self.requires_grad = bool(requires_grad and _tracking.get())
        self.grad = None
        self._parents = tuple(parents) if self.requires_grad else ()
        self._backward = None
        self._consumed = False

    def _add_grad(self, value):
        if self.requires_grad:
            value = np.asarray(value, dtype=self.data.dtype)
            self.grad = value.copy() if self.grad is None else self.grad + value

    @staticmethod
    def _coerce(value):
        return value if isinstance(value, Tensor) else Tensor(np.asarray(value, dtype=np.float32))

    def __add__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other))
        if out.requires_grad:
            def backward():
                self._add_grad(_unbroadcast(out.grad, self.data.shape))
                other._add_grad(_unbroadcast(out.grad, other.data.shape))
            out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self._coerce(other)

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other))
        if out.requires_grad:
            def backward():
                self._add_grad(_unbroadcast(out.grad * other.data, self.data.shape))
                other._add_grad(_unbroadcast(out.grad * self.data, other.data.shape))
            out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, self.requires_grad, (self,))
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * exponent * self.data ** (exponent - 1))
        return out

    def __truediv__(self, other):
        return self * self._coerce(other) ** -1

    def __matmul__(self, other):
        other = self._coerce(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("Kelus matmul requires tensors with at least two dimensions")
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other))
        if out.requires_grad:
            def backward():
                self._add_grad(_unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.data.shape))
                other._add_grad(_unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.data.shape))
            out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,))
        if out.requires_grad:
            def backward():
                gradient = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for item in sorted(item % self.data.ndim for item in axes):
                        gradient = np.expand_dims(gradient, item)
                self._add_grad(np.broadcast_to(gradient, self.data.shape))
            out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        axes = tuple(range(self.data.ndim)) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
        count = np.prod([self.data.shape[item] for item in axes])
        return self.sum(axis, keepdims) / float(count)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,))
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad.reshape(self.data.shape))
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), self.requires_grad, (self,))
        if out.requires_grad:
            inverse = np.argsort(axes)
            out._backward = lambda: self._add_grad(out.grad.transpose(*inverse))
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,))
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * (1 - value * value))
        return out

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        value = np.exp(shifted)
        value /= value.sum(axis=axis, keepdims=True)
        out = Tensor(value, self.requires_grad, (self,))
        if out.requires_grad:
            out._backward = lambda: self._add_grad(value * (out.grad - (out.grad * value).sum(axis=axis, keepdims=True)))
        return out

    def backward(self):
        if not self.requires_grad or self.data.size != 1:
            raise ValueError("backward requires a scalar loss with gradients enabled")
        if self._consumed:
            raise RuntimeError("This graph was already consumed. Run a new forward pass.")
        visited, ordered = set(), []

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            ordered.append(node)

        visit(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(ordered):
            if node._backward:
                node._backward()
            node._parents = ()
            node._backward = None
        self._consumed = True


def embedding(weight, indices):
    indices = np.asarray(indices, dtype=np.int64)
    out = Tensor(weight.data[indices], weight.requires_grad, (weight,))
    if out.requires_grad:
        def backward():
            gradient = np.zeros_like(weight.data)
            np.add.at(gradient, indices, out.grad)
            weight._add_grad(gradient)
        out._backward = backward
    return out


def cross_entropy(logits, targets):
    """Stable next-token loss. A target of -1 is ignored (prompt/padding)."""
    flat = logits.data.reshape(-1, logits.data.shape[-1])
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if flat.shape[0] != labels.size:
        raise ValueError("Target shape does not match logits")
    valid = labels >= 0
    if not valid.any() or np.any(labels[valid] >= flat.shape[-1]):
        raise ValueError("Loss requires valid target tokens")
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizer = exponentials.sum(axis=-1, keepdims=True)
    rows = np.flatnonzero(valid)
    value = (np.log(normalizer[valid, 0]) - shifted[rows, labels[valid]]).mean()
    out = Tensor(value, logits.requires_grad, (logits,))
    if out.requires_grad:
        def backward():
            gradient = exponentials / normalizer
            gradient[rows, labels[valid]] -= 1
            gradient[~valid] = 0
            gradient *= out.grad / valid.sum()
            logits._add_grad(gradient.reshape(logits.data.shape))
        out._backward = backward
    return out
