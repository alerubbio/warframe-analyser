import json

# Load the data
with open('data/processed/prime_items_updated.json', 'r', encoding='utf-8') as f:
    prime_items = json.load(f)

# Update set vaulted status to match blueprint
for slug, data in prime_items.items():
    if '_set' in slug:
        blueprint_slug = slug.replace('_set', '_blueprint')
        blueprint_data = prime_items.get(blueprint_slug)
        if blueprint_data:
            data['vaulted'] = blueprint_data['vaulted']
        else:
            print(f"Warning: No blueprint found for set {slug}")

# Save the updated data
with open('data/processed/prime_items_final.json', 'w', encoding='utf-8') as f:
    json.dump(prime_items, f, indent=2)

print("Set vaulted statuses synced with blueprint statuses.")
