import os
import shutil
import tarfile
from tqdm import tqdm

# ImageNet-100 classes
IMAGENET100_CLASSES = [
    'n02701002', 'n02814533', 'n02930766', 'n03100240', 'n03594945', 'n03670208', 'n03770679', 
    'n03777568', 'n04037443', 'n04285008', 'n02099601', 'n02106662', 'n02389026', 'n02906734', 
    'n03124170', 'n03272010', 'n03544143', 'n03649909', 'n03676483', 'n03777754', 'n04026417', 
    'n04467665', 'n02132136', 'n02441942', 'n02871525', 'n03017168', 'n03394916', 'n03425413', 
    'n03584829', 'n03688605', 'n03785016', 'n03947888', 'n04118538', 'n04254777', 'n02279972', 
    'n02484975', 'n02880940', 'n03042490', 'n03425595', 'n03584254', 'n03697007', 'n03785801', 
    'n04004767', 'n04127249', 'n04275548', 'n02321529', 'n02509815', 'n02892201', 'n03089624', 
    'n03445777', 'n03602883', 'n03706229', 'n03786901', 'n04008634', 'n04136333', 'n04310018', 
    'n02364673', 'n02666196', 'n02917067', 'n03095699', 'n03457902', 'n03627232', 'n03710721', 
    'n03804744', 'n04009552', 'n04141076', 'n04326547', 'n02410509', 'n02672831', 'n02917357', 
    'n03134739', 'n03476684', 'n03642806', 'n03720891', 'n03814639', 'n04019541', 'n04141975', 
    'n04371430', 'n02445715', 'n02699494', 'n02927161', 'n03141823', 'n03544143', 'n03649909', 
    'n03733131', 'n03837869', 'n04023962', 'n04146614', 'n04371774', 'n02480495', 'n02690373', 
    'n02930766', 'n03344393', 'n03584254', 'n03662601', 'n03759954', 'n03840681', 'n04026417', 
    'n04147183', 'n04376876'
]

def prepare_subset(src_path, dst_dir, class_list):
    """Extract only the specified classes from the source tar file"""
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"Extracting classes from {src_path}...")
    with tarfile.open(src_path) as tar:
        for member in tqdm(tar.getmembers()):
            class_id = member.name.split('/')[0]
            if class_id in class_list:
                tar.extract(member, dst_dir)

def prepare_imagenet100(imagenet_dir, output_dir):
    """
    Create ImageNet-100 from local ImageNet files
    Args:
        imagenet_dir: Directory containing ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar
        output_dir: Directory where ImageNet-100 will be created
    """
    # Check if source files exist
    train_file = os.path.join(imagenet_dir, "ILSVRC2012_img_train.tar")
    val_file = os.path.join(imagenet_dir, "ILSVRC2012_img_val.tar")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    # Extract training subset
    print("Processing training data...")
    prepare_subset(train_file, train_dir, IMAGENET100_CLASSES)
    
    # Extract validation subset
    print("Processing validation data...")
    prepare_subset(val_file, val_dir, IMAGENET100_CLASSES)
    
    # Save class list
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for class_id in IMAGENET100_CLASSES:
            f.write(f"{class_id}\n")
    
    print(f"\nImageNet-100 dataset prepared in: {output_dir}")
    print(f"Number of classes: {len(IMAGENET100_CLASSES)}")

if __name__ == "__main__":
    # Specify your local paths here
    imagenet_dir = "./downloaded_imagenet"  # Directory containing the original tar files
    output_dir = "./imagenet/imagenet100"   # Where to create ImageNet-100
    
    prepare_imagenet100(imagenet_dir, output_dir)
    print("Done!") 