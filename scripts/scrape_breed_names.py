
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_breeds(url, table_id):
    """Scrapes breed names from a table on the NBAGR website."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id=table_id)
        
        if table is None:
            print(f"Could not find table with id '{table_id}' on {url}")
            return []

        breeds = []
        # Assuming the breed name is in the second column (index 1)
        for row in table.find_all('tr')[1:]: # Skip header row
            cells = row.find_all('td')
            if len(cells) > 1:
                breed_name = cells[1].text.strip()
                if breed_name:
                    breeds.append(breed_name)
        return breeds
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

def main():
    # URLs from the web search results
    cattle_url = 'https://nbagr.res.in/en/registered-cattle-breeds-of-india/'
    buffalo_url = 'https://nbagr.res.in/en/registered-buffalo-breeds-of-india/'

    # The table IDs might need to be inspected from the website's HTML
    # For now, I'll use a placeholder and adjust if needed. A common practice is to
    # look for tables with specific classes or IDs. Let's assume a simple table structure.
    # After inspection, I will find the correct table identifier.
    # For this initial script, I will assume a generic table search.
    
    print("Scraping cattle breeds...")
    # I will need to inspect the page to find the correct table identifier.
    # Let's assume for now it's the first table on the page.
    cattle_breeds = get_breeds_generic(cattle_url)
    
    print("Scraping buffalo breeds...")
    buffalo_breeds = get_breeds_generic(buffalo_url)

    if cattle_breeds:
        print(f"Found {len(cattle_breeds)} cattle breeds.")
        df_cattle = pd.DataFrame({'Breed': cattle_breeds, 'Species': 'Cattle'})
        df_cattle.to_csv('cattle_breeds.csv', index=False)
        print("Saved cattle breeds to cattle_breeds.csv")

    if buffalo_breeds:
        print(f"Found {len(buffalo_breeds)} buffalo breeds.")
        df_buffalo = pd.DataFrame({'Breed': buffalo_breeds, 'Species': 'Buffalo'})
        df_buffalo.to_csv('buffalo_breeds.csv', index=False)
        print("Saved buffalo breeds to buffalo_breeds.csv")

def get_breeds_generic(url):
    """A more generic scraper that tries to find the most likely table."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            print(f"No tables found on {url}")
            return []

        # Heuristic: assume the largest table is the one we want.
        main_table = sorted(tables, key=lambda t: len(t.find_all('tr')), reverse=True)[0]
        
        breeds = []
        for row in main_table.find_all('tr')[1:]: # Skip header
            cells = row.find_all('td')
            if len(cells) > 1:
                breed_name = cells[1].text.strip()
                if breed_name:
                    breeds.append(breed_name)
        return breeds
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

if __name__ == '__main__':
    main()
