
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time
import random

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASTER_LIST_PATH = os.path.join(DATA_DIR, 'breed_master_list.csv')
IMAGES_PER_BREED = 50 # Start with a smaller number to test

# --- User-Agent to mimic a real browser ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}

def download_images(query, download_path, num_images):
    """Downloads images for a given query from Google Images."""
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Searching for: '{query}'...")
    # Using a different search URL that tends to be more stable for scraping
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  -> Failed to fetch search page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # This is the tricky part: Google obfuscates image data.
    # We need to find the script tags containing image information.
    # The exact structure changes over time.
    image_urls = []
    # A common pattern is to find all script tags and parse their content for URLs.
    # This is a simplified example and might need adjustment.
    for script_tag in soup.find_all("script"):
        if "AF_initDataCallback" in str(script_tag):
            # The data is often inside a complex JS object.
            # This requires careful parsing, which is beyond a simple script.
            # For this PoC, we will use a simpler, less reliable method of finding img tags.
            pass # Placeholder for more complex parsing

    # Fallback to finding image tags directly (less effective for full-res images)
    img_tags = soup.find_all('img', {'class': 'YQ4gaf'}, limit=num_images*2) # Get extra to filter
    
    count = 0
    for img in img_tags:
        if count >= num_images:
            break
        try:
            img_url = img['src']
            if not img_url.startswith('http'):
                continue # Skip data URIs

            img_data = requests.get(img_url, headers=HEADERS, timeout=10).content
            filename = os.path.join(download_path, f"{count+1:03d}.jpg")
            with open(filename, 'wb') as f:
                f.write(img_data)
            count += 1
            # Add a small delay
            time.sleep(random.uniform(0.2, 0.5))

        except Exception as e:
            # This will catch errors from missing src, timeouts, etc.
            continue
    
    print(f"  -> Downloaded {count} images.")

def main():
    if not os.path.exists(MASTER_LIST_PATH):
        print(f"Error: Master list not found at {MASTER_LIST_PATH}")
        return

    breeds_df = pd.read_csv(MASTER_LIST_PATH)

    for index, row in breeds_df.iterrows():
        breed_name = row['breed_name']
        filesystem_name = row['filesystem_name']
        species = row['species']
        
        print(f"\nProcessing breed {index+1}/{len(breeds_df)}: {breed_name} ({species})")
        
        # Construct a more specific query
        query = f"{breed_name} {species} India"
        
        breed_image_path = os.path.join(IMAGE_DIR, filesystem_name)
        
        if os.path.exists(breed_image_path) and len(os.listdir(breed_image_path)) >= IMAGES_PER_BREED:
            print(f"  -> Already have {len(os.listdir(breed_image_path))} images. Skipping.")
            continue

        download_images(query, breed_image_path, IMAGES_PER_BREED)
        
        # Be a good web citizen
        time.sleep(random.uniform(1, 3))

if __name__ == '__main__':
    main()
