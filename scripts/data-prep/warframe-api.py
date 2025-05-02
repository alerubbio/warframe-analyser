import json
import time
import requests

BASE_URL_V1 = "https://api.warframe.market/v1/"
BASE_URL_V2 = "https://api.warframe.market/v2/"
ASSET_URL = "https://warframe.market/static/assets/"

#store all items info in a json file
def get_items_info():
    url = f"{BASE_URL_V2}items"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error, failed to retrieve item_info: {response.status_code}")
        return None
    
def store_items_info():
    item_info = get_items_info()
    if item_info:
        with open('data/item_info.json', 'w', encoding='utf-8') as f:
            json.dump(item_info, f, ensure_ascii=False, indent=4)


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

# store item info in a json file
def store_item_info(item_name):
    item_data = get_item_info(item_name)
    with open(f'data/info/{item_name}.json', 'w', encoding='utf-8') as f:
        json.dump(item_data, f, ensure_ascii=False, indent=4)


# get specific item info
def get_item_drop(item_name):
    url = f"{BASE_URL_V1}items/{item_name}/dropsources"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error, failed to retrieve item_info: {response.status_code}")
        return None

#store item info in a json file
def store_item_drop(item_name):
    item_data = get_item_drop(item_name)
    with open(f'data/drops/{item_name}.json', 'w', encoding='utf-8') as f:
        json.dump(item_data, f, ensure_ascii=False, indent=4)

# MAIN
if __name__ == "__main__":
    # store_items_info()
    # store_item_drop("mirage_prime_systems_blueprint")
    store_item_info("axi_h3_relic")
    # store_prime_items()
