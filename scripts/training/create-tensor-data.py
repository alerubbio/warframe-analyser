import json
import torch
import numpy as np

# Load the combined data
with open('data/training/final_combined_data.json', 'r') as f:
    data = json.load(f)

# Define item type mapping (one-hot style: 5 types)
type_mapping = {
    'warframe': 0,
    'weapon': 1,
    'sentinel': 2,
    'archwing': 3,
    'other': 4
}

X = []
y = []

for slug, content in data.items():
    meta = content['meta']
    sequence = content['market_sequence']

    if len(sequence) != 50:
        continue  # Skip if not exactly 50 days

    vaulted = int(meta['vaulted'])
    ducats = meta['ducats']
    item_type = meta.get('type', 'other')

    # One-hot encode item type
    type_encoded = [0] * len(type_mapping)
    type_encoded[type_mapping.get(item_type, 4)] = 1

    seq_features = []
    for day in sequence:
        avg_price = day['avg_price']
        delta = 0.0 if last_avg is None else avg_price - last_avg
        last_avg = avg_price
        daily_features = [
            day['volume'],
            day['min_price'],
            day['max_price'],
            day['open_price'],
            day['closed_price'],
            avg_price,
            day['wa_price'],
            day['median'],
            day['donch_top'],
            day['donch_bot'],
            ducats,
            vaulted,
            *type_encoded
        ]

        seq_features.append(daily_features)

    X.append(seq_features)
    y.append(sequence[-1]['avg_price'])  # Predict next-day average price

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Save
print(f"✅ X shape: {X_tensor.shape}")  # Expect (672, 50, 17)
print("📦 First item sample features:", X_tensor[0][0])

torch.save(X_tensor, "data/training/X_delta.pt")
torch.save(y_tensor, "data/training/y_delta.pt")
print("💾 Saved tensors to data/training/X_delta.pt and data/training/y_delta.pt")
