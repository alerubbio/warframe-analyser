import json
import time
import requests

# Load prime items
with open('data/processed/prime_items_final.json', 'r', encoding='utf-8') as f:
    prime_items = json.load(f)

# Result container
market_data = {}

# Function to get 90-day stats for a slug
def get_item_statistics(slug):
    url = f'https://api.warframe.market/v1/items/{slug}/statistics'
    try:
        response = requests.get(url)
        response.raise_for_status()
        stats = response.json()
        # Go into payload -> statistics_closed -> find "90days"
        statistics = stats.get('payload', {}).get('statistics_closed', {}).get('90days', [])
        return statistics
    except Exception as e:
        print(f"Error fetching statistics for {slug}: {e}")
        return None

# Iterate through each prime item
for count, (slug, item_data) in enumerate(prime_items.items(), 1):
    print(f"Fetching {slug} ({count}/{len(prime_items)})...")
    stats_90days = get_item_statistics(slug)
    if stats_90days:
        market_data[slug] = stats_90days
    time.sleep(0.4)  # sleep to avoid rate-limiting (2-3 calls/sec max)

# Save the collected market data
with open('data/processed/prime_items_market_data.json', 'w', encoding='utf-8') as f:
    json.dump(market_data, f, indent=2)

print("✅ Finished fetching and saving market statistics!")
