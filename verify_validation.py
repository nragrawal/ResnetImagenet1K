import os
import xml.etree.ElementTree as ET
from collections import defaultdict

def verify_validation_files(src_root, class_id):
    """
    Verify validation files for a specific class in the original ILSVRC dataset
    Args:
        src_root: Path to original ILSVRC directory
        class_id: Class ID to verify (e.g., 'n02102973')
    """
    # Check paths
    val_dir = os.path.join(src_root, 'Data', 'CLS-LOC', 'val')
    val_anno_dir = os.path.join(src_root, 'Annotations', 'CLS-LOC', 'val')
    train_dir = os.path.join(src_root, 'Data', 'CLS-LOC', 'train', class_id)
    
    print(f"\nChecking class {class_id} in original dataset:")
    
    # 1. Check validation files using annotations
    print("\nValidation files:")
    if not os.path.exists(val_anno_dir):
        print(f"❌ Validation annotation directory not found: {val_anno_dir}")
        return
    
    # Find validation files for this class using annotations
    val_files = []
    for anno_file in os.listdir(val_anno_dir):
        if not anno_file.endswith('.xml'):
            continue
            
        # Parse annotation file
        tree = ET.parse(os.path.join(val_anno_dir, anno_file))
        root = tree.getroot()
        
        # Check if this image contains our class
        for obj in root.findall('object'):
            if obj.find('name').text == class_id:
                # Get image filename from annotation
                filename = root.find('filename').text + '.JPEG'
                val_files.append(filename)
                break  # Found our class in this image, move to next annotation
    
    if val_files:
        print(f"✓ Found {len(val_files)} validation files:")
        for f in val_files:
            img_path = os.path.join(val_dir, f)
            if os.path.exists(img_path):
                print(f"  ✓ {f}")
            else:
                print(f"  ❌ {f} (file missing)")
    else:
        print("❌ No validation files found for this class")
    
    # 2. Check training files for context
    print("\nTraining files:")
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.JPEG')]
        print(f"✓ Found {len(train_files)} training files")
    else:
        print("❌ No training directory found")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify validation files in ILSVRC dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to original ILSVRC directory')
    parser.add_argument('--class_id', type=str, required=True,
                      help='Class ID to verify (e.g., n02102973)')
    
    args = parser.parse_args()
    
    verify_validation_files(args.data_dir, args.class_id) 