import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check the number of images in each species folder.")
    parser.add_argument("target", type=int, help="The required number of images per folder")
    args = parser.parse_args()

    downloads_dir = Path("downloads")
    if not downloads_dir.exists() or not downloads_dir.is_dir():
        print(f"Error: {downloads_dir} directory not found.")
        return

    target_count = args.target
    not_meeting_target = []

    # Get only directories in downloads
    species_folders = sorted([f for f in downloads_dir.iterdir() if f.is_dir()])

    for folder in species_folders:
        # Count files in the folder
        # We assume any file inside is an image as per user request "how many images inside there"
        image_count = len([f for f in folder.iterdir() if f.is_file()])
        
        print(f"{folder.name}: {image_count}/{target_count}")
        
        if image_count < target_count:
            not_meeting_target.append(folder.name)

    print("\nFolders not meeting the target count:")
    if not_meeting_target:
        for species in not_meeting_target:
            print(species)
        
        with open("redownload.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(not_meeting_target) + "\n")
        print(f"\nSaved {len(not_meeting_target)} species to redownload.txt")
    else:
        print("All folders meet the target count!")

if __name__ == "__main__":
    main()
