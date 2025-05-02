import json

# Load the market data
with open('data/processed/prime_items_market_data.json', 'r', encoding='utf-8') as f:
    market_data = json.load(f)

# Loop through each item and print the number of days (entries)
for slug, entries in market_data.items():
    print(f"{slug}: {len(entries)} days of data")
