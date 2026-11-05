#!/usr/bin/env python3
"""
Dataset setup helper for bridge crack detection experiments
This script helps you organize your dataset in the required structure
"""

import os
import shutil
from pathlib import Path
import argparse

def check_dataset_structure(base_dir='.'):
    """Check if the dataset structure is correct"""
    base_path = Path(base_dir)
    
    print("=" * 60)
    print("Bridge Crack Dataset Structure Check")
    print("=" * 60)
    
    # Check current directory
    print(f"\nCurrent directory: {os.getcwd()}")
    
    # List all directories and files in current directory
    print("\nContents of current directory:")
    items = list(os.listdir('.'))[:20]  # Show first 20 items
    for item in items:
        item_path = Path(item)
        if item_path.is_dir():
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    if len(os.listdir('.')) > 20:
        print(f"  ... and {len(os.listdir('.'))-20} more items")
    
    # Check for archive directory
    archive_path = base_path / 'archive'
    if archive_path.exists():
        print(f"\n✅ Found 'archive' directory at: {archive_path.absolute()}")
        
        # List contents of archive
        print("\nContents of 'archive' directory:")
        for item in os.listdir(archive_path):
            item_path = archive_path / item
            if item_path.is_dir():
                print(f"  📁 {item}/")
                # Count files in subdirectory
                try:
                    file_count = len([f for f in os.listdir(item_path) if (item_path / f).is_file()])
                    dir_count = len([f for f in os.listdir(item_path) if (item_path / f).is_dir()])
                    print(f"     └── {file_count} files, {dir_count} subdirectories")
                except:
                    pass
            else:
                print(f"  📄 {item}")
        
        # Check for Decks directory
        decks_path = archive_path / 'Decks'
        if decks_path.exists():
            print(f"\n✅ Found 'Decks' directory")
            
            # Check for class directories
            expected_classes = ['cracked', 'non_cracked', 'Cracked', 'Non-Cracked', 'Non_Cracked']
            found_classes = []
            
            for class_name in expected_classes:
                class_path = decks_path / class_name
                if class_path.exists():
                    found_classes.append(class_name)
                    count = len(list(class_path.glob('*')))
                    print(f"  ✅ Found '{class_name}' with {count} items")
            
            if not found_classes:
                print("\n❌ No class directories found!")
                print("   Expected: 'cracked' and 'non_cracked' (or similar)")
                
                # List what's actually in Decks
                print("\n   Actual contents of Decks:")
                for item in list(os.listdir(decks_path))[:10]:
                    print(f"     - {item}")
        else:
            print(f"\n❌ 'Decks' directory not found in archive!")
            print(f"   Expected at: {decks_path}")
    else:
        print(f"\n❌ 'archive' directory not found!")
        print(f"   Expected at: {archive_path.absolute()}")
        
        # Look for common dataset locations
        print("\n🔍 Looking for dataset in common locations...")
        
        possible_locations = [
            Path('data'),
            Path('dataset'),
            Path('bridge_dataset'),
            Path('bridge-dataset'),
            Path('Decks'),
            Path('../archive'),
            Path('../data'),
            Path('crack_detection_dataset')
        ]
        
        found_datasets = []
        for loc in possible_locations:
            if loc.exists() and loc.is_dir():
                found_datasets.append(loc)
                print(f"  📁 Found directory: {loc}")
        
        if found_datasets:
            print("\n💡 Tip: You might need to reorganize your data.")
    
    print("\n" + "=" * 60)
    print("Expected Dataset Structure:")
    print("=" * 60)
    print("""
    archive/
    └── Decks/
        ├── cracked/          (or 'Cracked')
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── non_cracked/      (or 'Non_Cracked', 'Non-Cracked')
            ├── image1.jpg
            ├── image2.jpg
            └── ...
    """)
    
    return archive_path.exists() and (archive_path / 'Decks').exists()


def setup_dataset_from_kaggle(source_dir, target_dir='archive'):
    """
    Setup dataset from common Kaggle bridge crack dataset structure
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print("\n" + "=" * 60)
    print("Setting up dataset from Kaggle structure")
    print("=" * 60)
    
    # Create target directory
    target_path.mkdir(exist_ok=True)
    decks_path = target_path / 'Decks'
    decks_path.mkdir(exist_ok=True)
    
    # Look for class directories in source
    possible_class_names = {
        'positive': 'cracked',
        'negative': 'non_cracked',
        'Positive': 'cracked',
        'Negative': 'non_cracked',
        'crack': 'cracked',
        'no_crack': 'non_cracked',
        'Crack': 'cracked',
        'No_Crack': 'non_cracked',
        'cracked': 'cracked',
        'non_cracked': 'non_cracked',
        'Cracked': 'cracked',
        'Non_Cracked': 'non_cracked',
        'Non-Cracked': 'non_cracked'
    }
    
    copied_count = 0
    
    for source_name, target_name in possible_class_names.items():
        source_class = source_path / source_name
        if source_class.exists() and source_class.is_dir():
            target_class = decks_path / target_name
            
            print(f"\n📁 Found '{source_name}' directory")
            
            if not target_class.exists():
                print(f"   Copying to '{target_name}'...")
                shutil.copytree(source_class, target_class)
                file_count = len(list(target_class.glob('*')))
                print(f"   ✅ Copied {file_count} files")
                copied_count += file_count
            else:
                print(f"   ⚠️  '{target_name}' already exists, skipping")
    
    if copied_count > 0:
        print(f"\n✅ Successfully set up dataset with {copied_count} total files")
        print(f"   Dataset is ready at: {target_path.absolute()}")
    else:
        print("\n❌ No recognized class directories found in source")
        print("   Please check your source directory structure")
    
    return copied_count > 0


def main():
    parser = argparse.ArgumentParser(description='Setup bridge crack dataset')
    parser.add_argument('--check', action='store_true', help='Check current dataset structure')
    parser.add_argument('--setup', type=str, help='Setup dataset from source directory')
    parser.add_argument('--target', type=str, default='archive', help='Target directory for setup')
    
    args = parser.parse_args()
    
    if args.setup:
        # Try to set up from source
        success = setup_dataset_from_kaggle(args.setup, args.target)
        if success:
            check_dataset_structure(args.target)
    else:
        # Just check structure
        is_valid = check_dataset_structure()
        
        if not is_valid:
            print("\n💡 Suggestions:")
            print("1. Download the bridge crack dataset from Kaggle")
            print("2. Extract it to a folder")
            print("3. Run: python check_dataset.py --setup <path_to_extracted_folder>")
            print("\nExample:")
            print("   python check_dataset.py --setup ./bridge-crack-dataset")
            
            # Additional help for Windows users
            if os.name == 'nt':
                print("\n📝 Windows Users:")
                print("   Make sure to use forward slashes (/) or double backslashes (\\\\)")
                print("   Example: python check_dataset.py --setup C:/Users/YourName/Downloads/dataset")


if __name__ == "__main__":
    main()