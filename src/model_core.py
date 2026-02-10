import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.last_input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def update(self, learning_rate):
        if self.grad_weights is not None:
            self.weights -= learning_rate * self.grad_weights
            self.bias -= learning_rate * self.grad_bias

class ReLU:
    def __init__(self):
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.last_input <= 0] = 0
        return grad_input

class SoftmaxCrossEntropy:
    def __init__(self):
        self.last_predictions = None
        self.last_labels = None

    def forward(self, logits, labels):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.last_predictions = probabilities
        self.last_labels = labels
        m = labels.shape[0]
        loss = -np.sum(labels * np.log(probabilities + 1e-8)) / m
        return loss

    def backward(self):
        m = self.last_labels.shape[0]
        grad = (self.last_predictions - self.last_labels) / m
        return grad

class ForceMiniModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            LinearLayer(input_size, hidden_size),
            ReLU(),
            LinearLayer(hidden_size, output_size)
        ]
        self.loss_layer = SoftmaxCrossEntropy()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def loss(self, predictions, labels):
        return self.loss_layer.forward(predictions, labels)
    
    def backward(self):
        grad = self.loss_layer.backward()
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
    
    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(learning_rate)
    
    def get_config(self):
        return {
            'input_size': self.layers[0].weights.shape[0],
            'hidden_size': self.layers[0].weights.shape[1],
            'output_size': self.layers[2].weights.shape[1]
        }
    
    @staticmethod
    def from_config(config):
        return ForceMiniModel(
            config['input_size'],
            config['hidden_size'],
            config['output_size']
        )
