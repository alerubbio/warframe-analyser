import json

# Load market and meta data
with open('data/processed/trimmed_market_data.json', 'r') as f:
    market_data = json.load(f)

with open('data/processed/prime_items_meta_data.json', 'r') as f:
    meta_data = json.load(f)

final_data = {}

for item_slug, market_entries in market_data.items():
    if item_slug not in meta_data:
        print(f"Warning: {item_slug} missing in meta data!")
        continue

    meta_info = meta_data[item_slug]

    final_data[item_slug] = {
        "meta": {
            "name": meta_info.get("name"),
            "type": meta_info.get("type"),
            "ducats": meta_info.get("ducats"),
            "vaulted": meta_info.get("vaulted")
        },
        "market_sequence": market_entries  # the 50 market points
    }

# Save the merged output
with open('data/processed/final_combined_data.json', 'w') as f:
    json.dump(final_data, f, indent=2)

print("✅ Final combined data saved as 'final_combined_data.json'")
