import numpy as np

class BDPNoise:
    def __init__(self, epsilon, delta, sensitivity):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
    
    def add_noise(self, data):
        """Add Gaussian noise for (ε,δ)-differential privacy"""
        noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25/self.delta)) / self.epsilon
        noise = np.random.normal(0, noise_scale, data.shape)
        return data + noise