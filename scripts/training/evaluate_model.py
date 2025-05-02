import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load model class
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

# Load data
X = torch.load("data/training/X.pt")
y = torch.load("data/training/y.pt")
dataset = TensorDataset(X, y)

# Train/validation split (same as training script)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Load model
model = LSTMRegressor(input_dim=17, hidden_dim=64, num_layers=2)
model.load_state_dict(torch.load("models/lstm_regressor.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate a few samples
samples_to_plot = 10
predictions = []
actuals = []

with torch.no_grad():
    for i, (xb, yb) in enumerate(val_loader):
        xb = xb.to(device)
        pred = model(xb).cpu().item()
        actual = yb.item()
        predictions.append(pred)
        actuals.append(actual)
        if i >= samples_to_plot - 1:
            break

# Plot
plt.figure(figsize=(10, 5))
plt.plot(actuals, label="Actual Price", marker="o")
plt.plot(predictions, label="Predicted Price", marker="x")
plt.title("Predicted vs Actual Prime Set Prices")
plt.xlabel("Sample")
plt.ylabel("Avg Price (Platinum)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("models/model_eval.png")




# Assuming val_loader and model are already defined
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

# Concatenate predictions and targets
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# Compute metrics
rmse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)

print(f"📊 RMSE: {rmse:.2f}")
print(f"📉 MAE: {mae:.2f}")

