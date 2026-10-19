"""AdamW, gradient clipping, and checkpoint state implemented locally."""
import numpy as np


class AdamW:
    def __init__(self, parameters, learning_rate=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.step_count = 0
        self.m = {name: np.zeros_like(parameter.data) for name, parameter in parameters.items()}
        self.v = {name: np.zeros_like(parameter.data) for name, parameter in parameters.items()}

    def zero_grad(self):
        for parameter in self.parameters.values():
            parameter.grad = None

    def step(self, max_norm=1.0):
        gradients = [parameter.grad for parameter in self.parameters.values() if parameter.grad is not None]
        if not gradients or not all(np.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Missing or nonfinite gradients; no optimizer update was made")
        norm = float(np.sqrt(sum(float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients)))
        factor = min(1.0, max_norm / max(norm, 1e-12))
        self.step_count += 1
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                continue
            gradient = parameter.grad * factor
            self.m[name] *= self.beta1
            self.m[name] += (1 - self.beta1) * gradient
            self.v[name] *= self.beta2
            self.v[name] += (1 - self.beta2) * gradient * gradient
            m = self.m[name] / (1 - self.beta1 ** self.step_count)
            v = self.v[name] / (1 - self.beta2 ** self.step_count)
            if parameter.data.ndim >= 2:
                parameter.data *= 1 - self.learning_rate * self.weight_decay
            parameter.data -= self.learning_rate * m / (np.sqrt(v) + self.epsilon)
        return norm

    def metadata(self):
        return {"learning_rate": self.learning_rate, "weight_decay": self.weight_decay, "beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon, "step_count": self.step_count}

    def arrays(self):
        return {**{"adam.m." + name: value for name, value in self.m.items()}, **{"adam.v." + name: value for name, value in self.v.items()}}

    def restore(self, metadata, states):
        for key in ("learning_rate", "weight_decay", "beta1", "beta2", "epsilon", "step_count"):
            if key not in metadata:
                raise ValueError("Incomplete optimizer checkpoint")
        for prefix, destination in (("m", self.m), ("v", self.v)):
            for name, array in destination.items():
                value = states.get(f"adam.{prefix}.{name}")
                if value is None or value.shape != array.shape or not np.isfinite(value).all():
                    raise ValueError("Invalid optimizer state")
                array[:] = value
        for key, value in metadata.items():
            if key in self.metadata():
                setattr(self, key, value)
