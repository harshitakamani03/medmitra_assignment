import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

DATASET_DIR = "flickr8k_dataset" 
IMAGES_DIR = os.path.join(DATASET_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.csv")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_images")

os.makedirs(PROCESSED_DIR, exist_ok=True)

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def preprocess_images():
    processed_count = 0
    for image_file in tqdm(os.listdir(IMAGES_DIR), desc="Processing images"):
        try:
            image_path = os.path.join(IMAGES_DIR, image_file)
            with Image.open(image_path) as img:
                processed_img = image_transforms(img.convert("RGB"))
                torch.save(processed_img, os.path.join(PROCESSED_DIR, f"{image_file}.pt"))
                processed_count += 1
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(f"Successfully processed {processed_count} images!")

preprocess_images()
