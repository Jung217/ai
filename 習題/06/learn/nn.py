import numpy as np
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Layer(Module):

    def __init__(self, nin, nout, nonlin=True):
        self.w = Value(0.1 * np.random.randn(nin, nout))
        self.b = Value(0.1 * np.random.randn(nout))
        self.nonlin = nonlin

    def __call__(self, x):
        act = x.matmul(self.w) + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w, self.b

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Layer{self.w.data.shape}"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
