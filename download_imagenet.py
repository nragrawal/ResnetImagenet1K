import os
import shutil
import tarfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from a URL showing a progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def prepare_imagenet(root_dir):
    """
    Download and prepare ImageNet dataset
    Note: You need to have an account and download the ImageNet dataset from:
    https://image-net.org/download-images.php
    
    The authentication tokens below are placeholders and won't work.
    You need to:
    1. Register at https://image-net.org/
    2. Accept the terms of access
    3. Get your own download URLs/authentication tokens
    """
    
    os.makedirs(root_dir, exist_ok=True)
    
    # These URLs require authentication and are just placeholders
    # You need to get the actual URLs from image-net.org after registration
    train_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
    val_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
    
    print("Please note: You need to:")
    print("1. Register at https://image-net.org/")
    print("2. Accept the terms of access")
    print("3. Get your own download URLs/authentication tokens")
    print("4. Replace the URLs in this script with your authenticated URLs")
    print("\nThe current URLs are placeholders and won't work.\n")
    
    # Download training data
    print("Downloading training data...")
    train_path = os.path.join(root_dir, "ILSVRC2012_img_train.tar")
    if not os.path.exists(train_path):
        download_file(train_url, train_path)
    
    # Download validation data
    print("Downloading validation data...")
    val_path = os.path.join(root_dir, "ILSVRC2012_img_val.tar")
    if not os.path.exists(val_path):
        download_file(val_url, val_path)
    
    # Extract and organize training data
    train_dir = os.path.join(root_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print("Extracting training data...")
        with tarfile.open(train_path) as tar:
            tar.extractall(train_dir)
    
    # Extract and organize validation data
    val_dir = os.path.join(root_dir, "val")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        print("Extracting validation data...")
        with tarfile.open(val_path) as tar:
            tar.extractall(val_dir)

if __name__ == "__main__":
    # Specify your desired download directory
    root_dir = "./imagenet"
    prepare_imagenet(root_dir)
    print("Done!") 