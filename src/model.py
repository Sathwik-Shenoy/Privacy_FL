import numpy as np
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=256, output_size=10, learning_rate=0.01, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights with He initialization
        self.weights = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros((1, output_size))
        }
        
        # Initialize batch normalization parameters
        self.bn_params = {
            'gamma1': np.ones((1, hidden_size)),
            'beta1': np.zeros((1, hidden_size)),
            'gamma2': np.ones((1, output_size)),
            'beta2': np.zeros((1, output_size)),
            'running_mean1': np.zeros((1, hidden_size)),
            'running_var1': np.ones((1, hidden_size)),
            'running_mean2': np.zeros((1, output_size)),
            'running_var2': np.ones((1, output_size))
        }
        
        # Initialize momentum
        self.velocity = {
            'W1': np.zeros_like(self.weights['W1']),
            'b1': np.zeros_like(self.weights['b1']),
            'W2': np.zeros_like(self.weights['W2']),
            'b2': np.zeros_like(self.weights['b2'])
        }
        
        # Initialize batch statistics
        self.batch_stats = {
            'mean1': np.zeros((1, hidden_size)),
            'var1': np.ones((1, hidden_size)),
            'mean2': np.zeros((1, output_size)),
            'var2': np.ones((1, output_size))
        }
        
        # L2 regularization parameter
        self.l2_lambda = 0.0001
        
        # Dropout rate
        self.dropout_rate = 0.3
        
        # Dropout mask
        self.dropout_mask = None

    def forward(self, X, training=True):
        # First layer
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        
        # Batch normalization
        if training:
            self.batch_stats['mean1'] = np.mean(z1, axis=0, keepdims=True)
            self.batch_stats['var1'] = np.var(z1, axis=0, keepdims=True) + 1e-8
            z1_norm = (z1 - self.batch_stats['mean1']) / np.sqrt(self.batch_stats['var1'])
        else:
            z1_norm = (z1 - self.bn_params['running_mean1']) / np.sqrt(self.bn_params['running_var1'])
        
        z1_scaled = self.bn_params['gamma1'] * z1_norm + self.bn_params['beta1']
        
        # ReLU activation
        a1 = np.maximum(0, z1_scaled)
        
        # Dropout
        if training:
            self.dropout_mask = (np.random.rand(*a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            a1 *= self.dropout_mask
        
        # Output layer
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        
        # Batch normalization
        if training:
            self.batch_stats['mean2'] = np.mean(z2, axis=0, keepdims=True)
            self.batch_stats['var2'] = np.var(z2, axis=0, keepdims=True) + 1e-8
            z2_norm = (z2 - self.batch_stats['mean2']) / np.sqrt(self.batch_stats['var2'])
        else:
            z2_norm = (z2 - self.bn_params['running_mean2']) / np.sqrt(self.bn_params['running_var2'])
        
        z2_scaled = self.bn_params['gamma2'] * z2_norm + self.bn_params['beta2']
        
        # Softmax activation
        exp_z = np.exp(z2_scaled - np.max(z2_scaled, axis=1, keepdims=True))
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        return a1, a2

    def backward(self, X, y, a1, a2):
        m = X.shape[0]
        
        # Compute gradients
        dz2 = a2 - y
        
        # Batch normalization backprop
        dz2_norm = dz2 * self.bn_params['gamma2']
        dvar2 = np.sum(dz2_norm * (a2 - self.batch_stats['mean2']) * -0.5 * self.batch_stats['var2']**(-1.5), axis=0, keepdims=True)
        dmean2 = np.sum(dz2_norm * -1/np.sqrt(self.batch_stats['var2']), axis=0, keepdims=True) + dvar2 * np.mean(-2 * (a2 - self.batch_stats['mean2']), axis=0, keepdims=True)
        dz2 = dz2_norm / np.sqrt(self.batch_stats['var2']) + dvar2 * 2 * (a2 - self.batch_stats['mean2']) / m + dmean2 / m
        
        # Compute gradients for weights and biases
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Backprop through dropout
        if self.dropout_mask is not None:
            dz1 = np.dot(dz2, self.weights['W2'].T) * self.dropout_mask
        else:
            dz1 = np.dot(dz2, self.weights['W2'].T)
        
        # ReLU gradient
        dz1[a1 <= 0] = 0
        
        # Batch normalization backprop
        dz1_norm = dz1 * self.bn_params['gamma1']
        dvar1 = np.sum(dz1_norm * (a1 - self.batch_stats['mean1']) * -0.5 * self.batch_stats['var1']**(-1.5), axis=0, keepdims=True)
        dmean1 = np.sum(dz1_norm * -1/np.sqrt(self.batch_stats['var1']), axis=0, keepdims=True) + dvar1 * np.mean(-2 * (a1 - self.batch_stats['mean1']), axis=0, keepdims=True)
        dz1 = dz1_norm / np.sqrt(self.batch_stats['var1']) + dvar1 * 2 * (a1 - self.batch_stats['mean1']) / m + dmean1 / m
        
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Add L2 regularization
        dW1 += self.l2_lambda * self.weights['W1']
        dW2 += self.l2_lambda * self.weights['W2']
        
        # Update weights with momentum
        self.velocity['W1'] = self.momentum * self.velocity['W1'] - self.learning_rate * dW1
        self.velocity['b1'] = self.momentum * self.velocity['b1'] - self.learning_rate * db1
        self.velocity['W2'] = self.momentum * self.velocity['W2'] - self.learning_rate * dW2
        self.velocity['b2'] = self.momentum * self.velocity['b2'] - self.learning_rate * db2
        
        self.weights['W1'] += self.velocity['W1']
        self.weights['b1'] += self.velocity['b1']
        self.weights['W2'] += self.velocity['W2']
        self.weights['b2'] += self.velocity['b2']
        
        # Update batch normalization parameters
        self.bn_params['gamma1'] -= self.learning_rate * np.sum(dz1_norm * (a1 - self.batch_stats['mean1']) / np.sqrt(self.batch_stats['var1']), axis=0, keepdims=True)
        self.bn_params['beta1'] -= self.learning_rate * np.sum(dz1_norm, axis=0, keepdims=True)
        self.bn_params['gamma2'] -= self.learning_rate * np.sum(dz2_norm * (a2 - self.batch_stats['mean2']) / np.sqrt(self.batch_stats['var2']), axis=0, keepdims=True)
        self.bn_params['beta2'] -= self.learning_rate * np.sum(dz2_norm, axis=0, keepdims=True)
        
        # Update running statistics
        self.bn_params['running_mean1'] = 0.9 * self.bn_params['running_mean1'] + 0.1 * self.batch_stats['mean1']
        self.bn_params['running_var1'] = 0.9 * self.bn_params['running_var1'] + 0.1 * self.batch_stats['var1']
        self.bn_params['running_mean2'] = 0.9 * self.bn_params['running_mean2'] + 0.1 * self.batch_stats['mean2']
        self.bn_params['running_var2'] = 0.9 * self.bn_params['running_var2'] + 0.1 * self.batch_stats['var2']

    def predict(self, X):
        a1, a2 = self.forward(X, training=False)
        return np.argmax(a2, axis=1)
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)