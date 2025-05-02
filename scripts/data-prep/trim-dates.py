import json
from datetime import datetime

# Load your stats JSON
with open('data/processed/prime_items_market_data.json', 'r', encoding='utf-8') as f:
    item_stats = json.load(f)

date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
cutoff_date = datetime(2025, 4, 26)
days_to_keep = 50

trimmed_stats = {}

for slug, days in item_stats.items():
    if not days:
        continue
    
    # Sort the data by datetime ascending
    days_sorted = sorted(days, key=lambda d: datetime.strptime(d['datetime'], date_format))
    
    # Filter only dates up to cutoff_date
    days_filtered = [d for d in days_sorted if datetime.strptime(d['datetime'], date_format).date() <= cutoff_date.date()]
    
    # Take the last 50 days
    trimmed_days = days_filtered[-days_to_keep:]
    
    trimmed_stats[slug] = trimmed_days

# Save the trimmed dataset
with open('data/processed/trimmed_market_data.json', 'w', encoding='utf-8') as f:
    json.dump(trimmed_stats, f, indent=2)

print(f"Trimmed data saved! {len(trimmed_stats)} items processed.")
