import os
import pandas as pd

DATASET_DIR = "flickr8k_dataset"
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.txt")

data = []
with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        
        #skipping lines with no text
        if not line.strip():
            continue
        
        try:
            #working according to the given regex
            image_name, caption = line.strip().split(",", 1)
            data.append({"image_name": image_name.strip(), "caption": caption.strip()})
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")

df = pd.DataFrame(data)

output_file = os.path.join(DATASET_DIR, "captions.csv")
df.to_csv(output_file, index=False)

print(f"Data parsed: {len(df)} captions")
print(f"Output saved to: {output_file}")
print(df.head())
