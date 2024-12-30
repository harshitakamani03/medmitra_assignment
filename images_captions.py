import os
import shutil
import random
import pandas as pd

DATASET_DIR = "flickr8k_dataset"
PROCESSED_IMAGES_DIR = os.path.join(DATASET_DIR, "processed_images") 
OUTPUT_DIR = os.path.join(DATASET_DIR, "subset_images")  
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.csv")  
ZIP_FILE_PATH = os.path.join(DATASET_DIR, "subset_images.zip")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

captions_df = pd.read_csv(CAPTIONS_FILE)
processed_image_ids = [f.split('.')[0] for f in os.listdir(PROCESSED_IMAGES_DIR) if f.endswith('.jpg.pt')]

selected_image_ids = random.sample(processed_image_ids, 2000)
captions_df['image_id'] = captions_df['image_name'].apply(lambda x: x.split('.')[0])  
selected_captions_df = captions_df[captions_df['image_id'].isin(selected_image_ids)]
for image_id in selected_image_ids:
    src_image_path = os.path.join(PROCESSED_IMAGES_DIR, f"{image_id}.jpg.pt")
    dst_image_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg.pt")
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)

filtered_captions_file = os.path.join(OUTPUT_DIR, "captions_subset.csv")
selected_captions_df.to_csv(filtered_captions_file, index=False)

shutil.make_archive(ZIP_FILE_PATH.replace(".zip", ""), 'zip', OUTPUT_DIR)

print(f"Copied {len(selected_image_ids)} images and saved filtered captions to {filtered_captions_file}")
print(f"Zipped the folder and saved it as {ZIP_FILE_PATH}")
