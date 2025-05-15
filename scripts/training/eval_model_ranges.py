import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
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

# === Load validation data and model ===
X = torch.load(X_tensor_path)
y = torch.load(y_tensor_path)

num_features = 18
# Load validation split
from torch.utils.data import DataLoader, TensorDataset, random_split
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
val_loader = DataLoader(val_dataset, batch_size=32)

# Load model class
class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]
        return self.fc(last_output).squeeze(1)

# Initialize and load saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_dim=num_features, hidden_dim=64, num_layers=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Run prediction
all_preds, all_targets = [], []

with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds_log = model(xb)
        preds = torch.exp(preds_log)         # Exponentiate predictions
        targets = torch.exp(yb)              # Exponentiate targets
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

# Overall metrics
overall_rmse = mean_squared_error(all_targets, all_preds)
overall_mae = mean_absolute_error(all_targets, all_preds)
overall_mape = mean_absolute_percentage_error(all_targets, all_preds)

print(f"\n📊 Overall RMSE: {overall_rmse:.2f}")
print(f"📉 Overall MAE: {overall_mae:.2f}")
print(f"📈 Overall MAPE: {overall_mape:.2f}%")

# Price range evaluation
ranges = [
    ("<10p", 0, 10),
    ("10-30p", 10, 30),
    ("30-60p", 30, 60),
    ("60-80p", 60, 80),
    ("80-100p", 80, 100),
    ("100p+", 100, float("inf")),
]

print("\n📦 Error by Price Range:")
for label, low, high in ranges:
    indices = [i for i, val in enumerate(all_targets) if low <= val < high]
    if not indices:
        continue
    y_true = all_targets[indices]
    y_pred = all_preds[indices]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{label} | Samples: {len(indices)} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
