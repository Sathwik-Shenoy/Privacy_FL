import numpy as np

class FLServer:
    def __init__(self, model, test_data, test_labels):
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.best_accuracy = 0
        self.patience = 5
        self.patience_counter = 0
        
    def aggregate_models(self, client_models):
        """Aggregate client models using FedAvg with momentum."""
        n_clients = len(client_models)
        
        # Initialize aggregated weights
        aggregated_weights = {
            'W1': np.zeros_like(self.model.weights['W1']),
            'b1': np.zeros_like(self.model.weights['b1']),
            'W2': np.zeros_like(self.model.weights['W2']),
            'b2': np.zeros_like(self.model.weights['b2'])
        }
        
        # Average weights from all clients
        for client_model in client_models:
            for key in aggregated_weights:
                aggregated_weights[key] += client_model.weights[key] / n_clients
        
        # Update global model with momentum
        momentum = 0.9
        for key in self.model.weights:
            self.model.weights[key] = momentum * self.model.weights[key] + (1 - momentum) * aggregated_weights[key]
        
        # Update batch normalization parameters
        for key in self.model.bn_params:
            if key.startswith('running'):
                self.model.bn_params[key] = np.mean([client_model.bn_params[key] for client_model in client_models], axis=0)
    
    def evaluate(self):
        """Evaluate the global model on test data."""
        a1, a2 = self.model.forward(self.test_data, training=False)
        predictions = np.argmax(a2, axis=1)
        true_labels = np.argmax(self.test_labels, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        # Early stopping check
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return accuracy
    
    def should_stop(self):
        """Check if training should stop based on early stopping."""
        return self.patience_counter >= self.patience
    
    def get_global_weights(self):
        """Get the current global model weights."""
        return {
            'weights': {k: v.copy() for k, v in self.model.weights.items()},
            'bn_params': {k: v.copy() for k, v in self.model.bn_params.items()}
        }