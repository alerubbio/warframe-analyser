import json
import sys

def extract_relics(json_file_path):
    """
    Extract all relic items and their IDs from the Warframe item JSON data.
    Exclude requiem relics.
    
    Args:
        json_file_path (str): Path to the JSON file containing item data
        
    Returns:
        list: A list of dictionaries containing relic information
    """
    try:
        # Load the JSON data from file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        relic_map = {}
        
        # Check if the expected structure exists
        if 'data' not in data:
            print("Error: JSON structure doesn't contain 'data' key")
            return {}
            
        # Process each item in the data
        for item in data['data']:
            # Check if the item has tags and is a relic
            if 'tags' in item and 'relic' in item['tags']:
                # Skip requiem relics
                if 'requiem' not in item['tags']:

                    
                    relic_id = item['id']

                    relic_name = item['i18n']['en']['name']
                    slug = item['slug']
                    is_vaulted = item['vaulted']
                    # Map the relic name to its ID and vaulted status
                    relic_map[relic_id] = {
                        'name': relic_name,
                        'slug': slug,
                        'vaulted': is_vaulted
                    }
        
        return relic_map
    
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: '{json_file_path}' is not a valid JSON file")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def main():

    json_file_path = 'data/processed/item_info.json'
    relics = extract_relics(json_file_path)

    # Optionally save to a new JSON file
    with open('data/processed/extracted_relics.json', 'w', encoding='utf-8') as outfile:
        json.dump(relics, outfile, indent=2)
        print(f"\nExtracted relics have been saved to 'extracted_relics.json'")

if __name__ == "__main__":
    main()