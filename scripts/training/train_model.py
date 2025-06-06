import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into os.environ

tag = os.environ.get('MODEL_TYPE_TAG', 'default_tag')
tensor_path_tag = os.environ.get('TENSOR_PATH_TAG', 'default_tag')
model_path_tag = os.environ.get('MODEL_PATH_TAG', 'default_tag')

# Tensors (log-transformed labels)
X_tensor_path = f'{model_path_tag}{tensor_path_tag}X_{tag}.pt'
y_tensor_path = f'{model_path_tag}{tensor_path_tag}y_{tag}.pt'
model_path = f'{model_path_tag}lstm_regressor_{tag}.pt'

X = torch.load(X_tensor_path)
y = torch.load(y_tensor_path)

num_features = 18
num_epochs = 150

# Train-test split
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]  # (batch_size, hidden_dim)
        return self.fc(last_output).squeeze(1)  # (batch_size)

model = LSTMRegressor(input_dim=num_features, hidden_dim=64, num_layers=2)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# training loop with early stopping
best_val_loss = float('inf')
patience = 5
epochs_without_improvement = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):  # You can increase this as needed
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item() * xb.size(0)

    val_loss /= len(val_dataset)
    val_losses.append(val_loss)

    print(f"📈 Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), model_path)
        print("✅ New best model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("⏹️ Early stopping triggered.")
            break

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"models/loss_curve_{tag}.png")
plt.show()


# Save model
torch.save(model.state_dict(), model_path)
print("✅ Model saved.")
