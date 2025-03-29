import numpy as np

class FLClient:
    def __init__(self, model, data, labels, client_id):
        self.model = model
        self.data = data
        self.labels = labels
        self.client_id = client_id
        
        # Training parameters
        self.batch_size = 64
        self.local_epochs = 3
        self.learning_rate = 0.01
        self.momentum = 0.9
        
        # Privacy parameters - much less noise
        self.epsilon = 48.0  # Further increased epsilon
        self.delta = 1e-5
        self.sensitivity = 0.001  # Further reduced sensitivity
        
        # Sparse gradient threshold - more lenient
        self.sparse_threshold = 0.0000001
        
    def add_differential_privacy_noise(self, weights):
        """Add differential privacy noise to weights."""
        noise = np.random.normal(
            0,
            self.sensitivity * np.sqrt(2 * np.log(2/self.delta)) / self.epsilon,
            weights.shape
        )
        return weights + noise
    
    def evaluate(self, test_data, test_labels):
        """Evaluate the model on test data."""
        # Forward pass without training mode
        a1, a2 = self.model.forward(test_data, training=False)
        
        # Get predictions
        predictions = np.argmax(a2, axis=1)
        true_labels = np.argmax(test_labels, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == true_labels) * 100
        return accuracy
    
    def train(self):
        """Train the model locally with differential privacy."""
        # Store original weights
        original_weights = {k: v.copy() for k, v in self.model.weights.items()}
        
        n_samples = len(self.data)
        n_batches = n_samples // self.batch_size
        
        # Train normally without privacy noise
        for epoch in range(self.local_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            shuffled_data = self.data[indices]
            shuffled_labels = self.labels[indices]
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                
                batch_data = shuffled_data[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]
                
                # Forward pass
                a1, a2 = self.model.forward(batch_data, training=True)
                
                # Backward pass
                self.model.backward(batch_data, batch_labels, a1, a2)
        
        # Calculate weight updates
        weight_updates = {k: self.model.weights[k] - original_weights[k] 
                         for k in self.model.weights}
        
        # Apply sparsification to updates
        for key in weight_updates:
            mask = np.abs(weight_updates[key]) > self.sparse_threshold
            weight_updates[key] *= mask
        
        # Add privacy noise to sparse updates
        for key in weight_updates:
            weight_updates[key] = self.add_differential_privacy_noise(weight_updates[key])
        
        # Apply the final updates
        for key in self.model.weights:
            self.model.weights[key] = original_weights[key] + weight_updates[key]