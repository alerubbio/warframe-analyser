import json
import time
import requests

# Load data
with open('data/processed/prime_items.json', 'r', encoding='utf-8') as f:
    prime_items = json.load(f)

with open('data/processed/extracted_relics.json', 'r', encoding='utf-8') as f:
    relics = json.load(f)

# Drop sources API template
DROP_SOURCES_URL = "https://api.warframe.market/v1/items/{}/dropsources"

def get_drops(item_slug):
    try:
        url = DROP_SOURCES_URL.format(item_slug)
        res = requests.get(url)
        res.raise_for_status()
        return res.json()['payload']['dropsources']
    except Exception as e:
        print(f"Failed to fetch drops for {item_slug}: {e}")
        return []

def is_item_vaulted(drop_data, relics_dict):
    relic_ids = [d['relic'] for d in drop_data if d['type'] == 'relic']
    if not relic_ids:
        return True  # no relics = not farmable = vaulted
    for relic_id in relic_ids:
        if relic_id in relics_dict and not relics_dict[relic_id]['vaulted']:
            return False  # if any relic is unvaulted
    return True  # all relics are vaulted

# Process each prime item
for slug, data in prime_items.items():
    print(f"Checking: {slug}")
    drop_data = get_drops(slug)
    vaulted = is_item_vaulted(drop_data, relics)
    prime_items[slug]['vaulted'] = vaulted
    time.sleep(0.5)  # respectful rate limiting

# Save updated prime_items with vaulted info
with open('data/processed/prime_items_updated.json', 'w', encoding='utf-8') as f:
    json.dump(prime_items, f, indent=2)

print("Vaulted status update complete.")
