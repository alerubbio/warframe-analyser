import json
import requests
import time

BASE_URL_V1 = "https://api.warframe.market/v1/"
BASE_URL_V2 = "https://api.warframe.market/v2/"
ASSET_URL = "https://warframe.market/static/assets/"
def parse_item_info():
    with open('data/processed/item_info.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    prime_items = {}

    for item in data['data']:
        name = item['i18n']['en']['name']
        tags = item.get('tags', [])
        
        if 'prime' in tags or 'prime' in name.lower():
            if 'mod' in tags:
                continue  # skip prime mods
            
            slug = item['slug']
            item_type = (
                'warframe' if 'warframe' in tags else
                'weapon' if 'weapon' in tags else
                'companion' if 'companion' in tags else
                'sentinel' if 'sentinel' in tags else
                'other'
            )
            
            prime_items[slug] = {
                'name': name,
                'type': item_type
            }

    # Save as flat JSON dict
    with open('prime_items.json', 'w', encoding='utf-8') as f:
        json.dump(prime_items, f, indent=2)


    print(f"Collected {len(prime_items)} prime items.")

# get specific item info
# using v2 for smaller request
def get_item_info(item_name):
    url = f"{BASE_URL_V2}items/{item_name}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error, failed to retrieve item_info: {response.status_code}")
        return None

# wikilink
# ducats
# tradable
def store_prime_items():
    # Load your existing prime items
    with open('data/prime_items.json', 'r', encoding='utf-8') as f:
        prime_items = json.load(f)
    
    # Track progress
    count = 0
    total = len(prime_items)

    for slug, info in prime_items.items():
        try:
            response = get_item_info(slug)

            item_data = response['data']
            
            # Add the desired fields
            info['wiki_link'] = item_data['i18n']['en'].get('wikiLink')
            info['ducats'] = item_data.get('ducats')
            info['tradable'] = item_data.get('tradable', True)

            count += 1
            print(f"[{count}/{total}] Updated {slug} ✓")

            if count % 10 == 0:
                with open('data/prime_items.json', 'w', encoding='utf-8') as f:
                    json.dump(prime_items, f, indent=2)

            time.sleep(1.5)

        except Exception as e:
            print(f"[{slug}] Failed: {e}")
            time.sleep(2)

    # Final save
    with open('data/prime_items.json', 'w', encoding='utf-8') as f:
        json.dump(prime_items, f, indent=2)




# MAIN
if __name__ == "__main__":
    parse_item_info()