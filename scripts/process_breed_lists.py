
import pandas as pd
import os

# Data manually gathered from the web search results
cattle_breeds = [
    'Alambadi', 'Amritmahal', 'Bachaur', 'Bargur', 'Belahi', 'Bengali', 
    'Binjharpuri', 'Badri', 'Dangi', 'Deoni', 'Gangabari', 'Gangadiri', 
    'Gaolao', 'Ghumsari', 'Gir', 'Goomsur', 'Hallikar', 'Hariana', 'Jhari', 
    'Kangayam', 'Kankrej', 'Kenkatha', 'Kherigarh', 'Khillar', 'Kosal', 
    'Krishna Valley', 'Kumauni', 'Ladakhi', 'Lakhimi', 'Malnad Gidda', 
    'Malvi', 'Mampati', 'Manapari', 'Mewati', 'Motu', 'Nagori', 'Nimari', 
    'Ongole', 'Poda Thurpu', 'Pulikulam', 'Punganur', 'Rathi', 'Red Kandhari', 
    'Red Sindhi', 'Sahiwal', 'Sanchori', 'Siri', 'Tharparkar', 'Umblachery', 
    'Vechur', 'Wedchur', 'Zobawng', 'Frieswal' # Synthetic breed
]

buffalo_breeds = [
    'Murrah', 'Nili Ravi', 'Bhadawari', 'Jaffarabadi', 'Surti', 'Mehsana', 
    'Nagpuri', 'Pandharpuri', 'Marathwadi', 'Toda', 'Banni', 'Chilika', 
    'Kalahandi', 'Luit', 'Bargur', 'Chhattisgarhi', 'Gojri', 
    'Dharwadi', 'Purnathadi', 'Manah'
    # The list from the search has 20 distinct names. The 21st might be a regional variant or a new addition not listed.
    # We will proceed with these 20 for now.
]

def create_breed_csv(output_dir):
    """Creates a consolidated CSV of cattle and buffalo breeds."""
    
    df_cattle = pd.DataFrame({'breed_name': cattle_breeds, 'species': 'Cattle'})
    df_buffalo = pd.DataFrame({'breed_name': buffalo_breeds, 'species': 'Buffalo'})
    
    df_combined = pd.concat([df_cattle, df_buffalo], ignore_index=True)
    
    # Clean names to be filesystem-friendly
    df_combined['filesystem_name'] = df_combined['breed_name'].str.lower().str.replace(' ', '_', regex=False)
    
    output_path = os.path.join(output_dir, 'breed_master_list.csv')
    df_combined.to_csv(output_path, index=False)
    
    print(f"Successfully created breed master list at: {output_path}")
    print(f"Total breeds: {len(df_combined)}")
    print(f"Cattle breeds: {len(df_cattle)}")
    print(f"Buffalo breeds: {len(df_buffalo)}")

if __name__ == '__main__':
    # The script is in 'scripts', so the output dir is '../data'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    create_breed_csv(data_dir)

