import json
from datetime import datetime

# Load your stats JSON
with open('data/processed/prime_items_market_data.json', 'r', encoding='utf-8') as f:
    item_stats = json.load(f)

date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
latest_expected_date = datetime(2025, 4, 27)

days_per_item = {}
items_wrong_date = []

for slug, days in item_stats.items():
    if not days:
        continue
    # Sort the data by datetime (just in case)
    days_sorted = sorted(days, key=lambda d: datetime.strptime(d['datetime'], date_format))
    latest_date = datetime.strptime(days_sorted[-1]['datetime'], date_format)
    num_days = len(days_sorted)
    
    days_per_item[slug] = num_days

    # Check if latest date matches expected
    if latest_date.date() != latest_expected_date.date():
        items_wrong_date.append((slug, latest_date.date()))

# Find minimum number of days
min_days = min(days_per_item.values())

print(f"Minimum number of days across all items: {min_days}")
print(f"Number of items with wrong most recent date: {len(items_wrong_date)}")
if items_wrong_date:
    print("Items with wrong dates:")
    for slug, date in items_wrong_date:
        print(f"{slug}: {date}")
