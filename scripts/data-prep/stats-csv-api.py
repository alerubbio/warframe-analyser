import csv
import json
import requests
import time
from datetime import datetime

DATA_FILE = 'data/processed/prime_items_final.json'
BASE_URL_V1 = "https://api.warframe.market/v1/"

# Load your metadata
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    prime_items = json.load(f)

output_rows = []
total_items = len(prime_items)
processed = 0
successful = 0
failed = 0

print(f"Starting to process {total_items} items...")

for slug, item_data in prime_items.items():
    processed += 1
    
    # Progress information
    print(f"[{processed}/{total_items}] Processing: {item_data['name']} ({slug})")
    
    url = f"{BASE_URL_V1}items/{slug}/statistics"
    
    try:
        response = requests.get(url)
        
        if response.status_code != 200:
            failed += 1
            print(f"  ❌ Failed with status code {response.status_code}")
            continue
        
        stats = response.json()['payload']['statistics_closed']['90days']
        
        # Log number of data points found
        successful += 1
        print(f"  ✅ Success! Found {len(stats)} data points (Vaulted: {item_data['vaulted']})")
        
        for entry in stats:
            output_rows.append({
                'slug': slug,
                'date': entry['datetime'].split('T')[0],
                'avg_price': entry['avg_price'],
                'wa_price': entry['wa_price'],
                'median': entry['median'],
                'volume': entry['volume'],
                'vaulted': item_data['vaulted'],
                'ducats': item_data['ducats']
            })
        
        # Adding a short delay to be respectful to the API
        time.sleep(0.5)
        
    except Exception as e:
        failed += 1
        print(f"  ❌ Error: {str(e)}")
    
    # Print progress summary every 10 items
    if processed % 10 == 0 or processed == total_items:
        print(f"\nProgress Summary:")
        print(f"Processed: {processed}/{total_items} ({processed/total_items*100:.1f}%)")
        print(f"Successful: {successful} | Failed: {failed}")
        print(f"Data points collected: {len(output_rows)}\n")

# Final summary
print(f"\n=== FINAL SUMMARY ===")
print(f"Total items processed: {processed}/{total_items}")
print(f"Successful requests: {successful}")
print(f"Failed requests: {failed}")
print(f"Total data points collected: {len(output_rows)}")

# Write to a single CSV
csv_filename = 'data/csvs/market_data_90days.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nData saved to {csv_filename}")