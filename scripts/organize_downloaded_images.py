
import os
import pandas as pd
import shutil

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MASTER_LIST_PATH = os.path.join(DATA_DIR, 'breed_master_list.csv')

# Source directory for the downloaded dataset
DOWNLOADED_DATASET_PATH = os.path.join(DATA_DIR, 'images', 'archive', 'Indian_bovine_breeds')

# Target directory for organized images
ORGANIZED_IMAGE_PATH = os.path.join(DATA_DIR, 'images')

def organize_images():
    if not os.path.exists(MASTER_LIST_PATH):
        print(f"Error: Master breed list not found at {MASTER_LIST_PATH}")
        return

    if not os.path.exists(DOWNLOADED_DATASET_PATH):
        print(f"Error: Downloaded dataset not found at {DOWNLOADED_DATASET_PATH}")
        return

    breeds_df = pd.read_csv(MASTER_LIST_PATH)
    master_breeds_map = dict(zip(breeds_df['breed_name'].str.lower(), breeds_df['filesystem_name']))
    
    print(f"Organizing images from {DOWNLOADED_DATASET_PATH} to {ORGANIZED_IMAGE_PATH}")
    
    processed_breeds = set()
    unmatched_downloaded_breeds = set()

    for breed_folder in os.listdir(DOWNLOADED_DATASET_PATH):
        source_breed_path = os.path.join(DOWNLOADED_DATASET_PATH, breed_folder)
        
        if os.path.isdir(source_breed_path):
            lower_breed_folder = breed_folder.lower().replace('_', ' ') # Handle potential underscores in downloaded names
            
            matched_filesystem_name = None
            for master_breed_name_lower, fs_name in master_breeds_map.items():
                if master_breed_name_lower == lower_breed_folder or master_breed_name_lower.replace(' ', '_') == lower_breed_folder:
                    matched_filesystem_name = fs_name
                    break
            
            if matched_filesystem_name:
                target_breed_path = os.path.join(ORGANIZED_IMAGE_PATH, matched_filesystem_name)
                os.makedirs(target_breed_path, exist_ok=True)
                
                print(f"  Copying images for {breed_folder} to {target_breed_path}")
                num_copied = 0
                for img_file in os.listdir(source_breed_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        shutil.copy(os.path.join(source_breed_path, img_file), target_breed_path)
                        num_copied += 1
                print(f"    Copied {num_copied} images.")
                processed_breeds.add(matched_filesystem_name)
            else:
                unmatched_downloaded_breeds.add(breed_folder)
                print(f"  Warning: Breed folder '{breed_folder}' from downloaded dataset not found in master list.")

    print("\n--- Summary ---")
    print(f"Successfully organized images for {len(processed_breeds)} breeds.")
    
    if unmatched_downloaded_breeds:
        print("The following breed folders from the downloaded dataset were NOT in your master list:")
        for breed in sorted(list(unmatched_downloaded_breeds)):
            print(f"- {breed}")
            
    missing_from_downloaded = set(master_breeds_map.values()) - processed_breeds
    if missing_from_downloaded:
        print("\nThe following breeds from your master list were NOT found in the downloaded dataset:")
        for breed_fs_name in sorted(list(missing_from_downloaded)):
            original_name = breeds_df[breeds_df['filesystem_name'] == breed_fs_name]['breed_name'].iloc[0]
            print(f"- {original_name} ({breed_fs_name})")

if __name__ == '__main__':
    organize_images()
