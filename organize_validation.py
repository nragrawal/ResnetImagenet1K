import os
import shutil
from tqdm import tqdm
from collections import defaultdict

def organize_validation_set(data_dir):
    """
    Organize validation images into class folders
    Args:
        data_dir: Path to ILSVRC subset directory
    """
    val_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'val')
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return False
    
    # Create temporary directory
    temp_dir = val_dir + '_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get list of classes from training set
    train_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'train')
    class_names = set(os.listdir(train_dir))
    print(f"Found {len(class_names)} classes in training set")
    
    # Count files by class before moving
    class_counts = defaultdict(int)
    print("\nCounting validation files by class...")
    for filename in os.listdir(val_dir):
        if filename.endswith('.JPEG'):
            class_id = filename.split('_')[0]
            if class_id in class_names:
                class_counts[class_id] += 1
    
    print("\nValidation files per class:")
    for class_id in class_names:
        count = class_counts[class_id]
        if count == 0:
            print(f"❌ Warning: No validation files found for class {class_id}")
    
    # Move all files to temporary directory
    print("\nMoving files to temporary directory...")
    moved_files = 0
    for filename in os.listdir(val_dir):
        if filename.endswith('.JPEG'):
            src_path = os.path.join(val_dir, filename)
            dst_path = os.path.join(temp_dir, filename)
            shutil.move(src_path, dst_path)
            moved_files += 1
    print(f"Moved {moved_files} files to temporary directory")
    
    # Create class directories in validation folder
    print("\nCreating class directories...")
    for class_name in class_names:
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # Move files to their class directories
    print("\nOrganizing validation images into class folders...")
    organized_counts = defaultdict(int)
    for filename in tqdm(os.listdir(temp_dir)):
        if filename.endswith('.JPEG'):
            class_id = filename.split('_')[0]
            if class_id in class_names:
                src_path = os.path.join(temp_dir, filename)
                dst_dir = os.path.join(val_dir, class_id)
                dst_path = os.path.join(dst_dir, filename)
                shutil.move(src_path, dst_path)
                organized_counts[class_id] += 1
    
    # Verify organization
    print("\nVerifying organization...")
    success = True
    for class_id in class_names:
        original_count = class_counts[class_id]
        organized_count = organized_counts[class_id]
        if original_count != organized_count:
            print(f"❌ Mismatch for class {class_id}: Expected {original_count}, got {organized_count}")
            success = False
        if organized_count == 0:
            print(f"❌ No images in validation folder for class {class_id}")
            success = False
    
    # Clean up
    print("\nCleaning up...")
    remaining_files = os.listdir(temp_dir)
    if remaining_files:
        print(f"Warning: {len(remaining_files)} files remained in temporary directory")
        print("These files don't belong to any training class:")
        for filename in remaining_files:
            print(f"  {filename}")
    shutil.rmtree(temp_dir)
    
    if success:
        print("\n✓ Validation set organization completed successfully!")
    else:
        print("\n❌ Validation set organization completed with errors!")
    return success

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize ImageNet validation set')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ILSVRC subset directory')
    
    args = parser.parse_args()
    
    if organize_validation_set(args.data_dir):
        print("✓ Validation set is now properly organized!")
    else:
        print("❌ Failed to organize validation set. Please check the errors above.")