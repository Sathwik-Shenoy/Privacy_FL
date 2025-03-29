import numpy as np
from data_loader import load_mnist
from model import NeuralNetwork
from client import FLClient
from server import FLServer

def main():
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_data, train_labels, test_data, test_labels = load_mnist()
    print("Dataset loaded successfully!")
    print(f"Training set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}\n")

    # Initialize model and server
    input_size = train_data.shape[1]
    hidden_size = 512
    output_size = 10
    model = NeuralNetwork(input_size, hidden_size, output_size)
    server = FLServer(model, test_data, test_labels)

    # Create clients
    n_clients = 3
    print(f"🚀 Starting Federated Learning with {n_clients} clients")
    print("📈 Training for 100 rounds\n")

    # Split data among clients
    client_data = np.array_split(train_data, n_clients)
    client_labels = np.array_split(train_labels, n_clients)
    clients = [FLClient(model, client_data[i], client_labels[i], i) for i in range(n_clients)]

    # Training parameters
    n_rounds = 100
    best_accuracy = 0
    patience = 5  # Number of rounds to wait before stopping if accuracy degrades
    no_improvement_count = 0

    # Training loop
    for round in range(1, n_rounds + 1):
        print(f"\n🔁 Round {round}/{n_rounds}")
        
        # Train on each client
        client_accuracies = []
        for client in clients:
            client.train()
            # Evaluate client's model
            client_accuracy = client.evaluate(test_data, test_labels)
            client_accuracies.append(client_accuracy)
        
        # Average client accuracy
        avg_client_accuracy = np.mean(client_accuracies)
        print(f"   Client Avg Accuracy: {avg_client_accuracy:.2f}%")
        
        # Aggregate models
        server.aggregate_models([client.model for client in clients])
        
        # Evaluate global model
        global_accuracy = server.evaluate() * 100  # Convert to percentage
        print(f"   Global Model Accuracy: {global_accuracy:.2f}%\n")
        
        # Update best accuracy and check for degradation
        if global_accuracy > best_accuracy:
            best_accuracy = global_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early stopping if accuracy starts to degrade
        if no_improvement_count >= patience:
            print(f"🎯 Early stopping at round {round}")
            print(f"   Best accuracy: {best_accuracy:.2f}%")
            break

    print("\n✨ Training completed!")
    print(f"   Final accuracy: {global_accuracy:.2f}%")
    print(f"   Best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()