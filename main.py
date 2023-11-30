# SageMaker Training Script
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Load pre-trained GPT model and tokenizer
gpt_model_name = "gpt2"
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)

# Custom transformer to handle IP addresses
class IPTransformer:
    def __init__(self, col_indices):
        self.col_indices = col_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col_index in self.col_indices:
            for row in X:
                # Convert IP addresses to numeric representation
                row[col_index] = sum(int(x) << (8 * i) for i, x in enumerate(row[col_index].split('.')))
        return X


# Custom PyTorch model for firewall optimization
class FirewallOptimizationModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FirewallOptimizationModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


# Function to train the PyTorch model
def train_pytorch_model(features, labels, input_size, hidden_size, output_size, learning_rate, epochs):
    model = FirewallOptimizationModel(input_size, hidden_size, output_size)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert features to float
    features = features.astype(np.float32)

    X_tensor = torch.FloatTensor(features)
    y_tensor = torch.FloatTensor(labels).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


# Function to generate firewall rules using GPT
def generate_firewall_rule(prompt, max_length=50):
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    output = gpt_model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50,
                                top_p=0.95)
    generated_rule = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_rule

if __name__ == '__main__':
    # Your firewall rules (replace with your actual firewall rules dataset)
    firewall_rules = [
        {'source_ip': '192.168.1.1', 'destination_ip': '10.0.0.1', 'protocol': 'TCP', 'action': 'ALLOW'},
        {'source_ip': '192.168.1.2', 'destination_ip': '10.0.0.2', 'protocol': 'UDP', 'action': 'DENY'},
        {'source_ip': '192.168.1.3', 'destination_ip': '10.0.0.3', 'protocol': 'TCP', 'action': 'ALLOW'},
        # Add more rules as needed
    ]

    # Convert firewall rules to features and labels
    features = []
    labels = []

    for rule in firewall_rules:
        feature = [rule['source_ip'], rule['destination_ip'], rule['protocol']]
        label = 1 if rule['action'] == 'ALLOW' else 0
        features.append(feature)
        labels.append(label)

    # Convert features and labels to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Set of firewall-related prompts
    firewall_prompts = [
        "Allow traffic from IP address ",
        "Deny access to port ",
        "Log all incoming connections from ",
        "Create a rule to block traffic to ",
        # Add more prompts based on your use case
    ]

    # Apply IP transformation to handle IP addresses
    ip_transformer = IPTransformer(col_indices=[0, 1])  # Apply transformation to both source and destination IP
    X = ip_transformer.transform(X)

    # One-hot encode the 'protocol' column
    protocol_col_index = 2
    encoder = OneHotEncoder(categories='auto', sparse=False)
    protocol_encoded = encoder.fit_transform(X[:, protocol_col_index].reshape(-1, 1))

    # Concatenate the one-hot encoded protocol with the rest of the features
    X = np.concatenate((X[:, :protocol_col_index], protocol_encoded, X[:, protocol_col_index + 1:]), axis=1)

    # Set input size dynamically
    input_size = X.shape[1]

    # Train PyTorch model
    trained_pytorch_model = train_pytorch_model(X, y, input_size, 20, 1, 0.001, 50)

    # Save PyTorch model artifacts
    torch.save(trained_pytorch_model.state_dict(), 'pytorch_model.pth')

    # Generate and save firewall rules
    generated_rules = []
    for prompt in firewall_prompts:
        generated_rule = generate_firewall_rule(prompt)
        generated_rules.append(generated_rule)

    # Save the generated rules (you need to define the saving mechanism based on your use case)
    np.savetxt("generated_rules.txt", generated_rules, fmt="%s")
