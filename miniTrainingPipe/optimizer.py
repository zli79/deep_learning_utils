class SGD:
    def __init__(self, params, lr = 1e-2):
        self.params = params
        self.lr = lr

    def step(self, grads):
        for k in self.params:
            self.params[k] -= self.lr * grads[k]
