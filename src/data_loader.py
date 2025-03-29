import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def to_one_hot(y, num_classes=10):
    """Convert label indices to one-hot encoded vectors."""
    return np.eye(num_classes)[y]

def load_mnist():
    """Load and preprocess MNIST dataset."""
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert labels to one-hot encoding
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    
    return X_train, y_train, X_test, y_test

def load_client_data(client_id, n_clients=3):
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize
    X = MinMaxScaler().fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Split into shards
    shard_size = len(X_train) // n_clients
    start = client_id * shard_size
    end = start + shard_size
    
    # Create client data
    client_train = {
        'X': X_train[start:end],
        'y': y_train[start:end]
    }
    
    client_test = {
        'X': X_test,
        'y': y_test
    }
    
    return client_train, client_test