import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

DATASET_DIR = "flickr8k_dataset"
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.csv")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_images")

captions_df = pd.read_csv(CAPTIONS_FILE)

#tokenizer gor token ids
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class Flickr8kDataset(Dataset):
    def __init__(self, captions_df, processed_dir, tokenizer, max_len=128): #default as 128
        self.captions_df = captions_df
        self.processed_dir = processed_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        image_name = self.captions_df.iloc[idx, 0]
        caption = self.captions_df.iloc[idx, 1]

        image_path = os.path.join(self.processed_dir, f"{image_name}.pt")
        image_tensor = torch.load(image_path)

        tokens = self.tokenizer(
            caption, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "image": image_tensor,
            "caption": tokens["input_ids"].squeeze(0),
            #set actual ids as 1 and padding as 0
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }

dataset = Flickr8kDataset(captions_df, PROCESSED_DIR, tokenizer)

#80% training dataset 20% validation
train_size = int(0.8 * len(dataset))  
val_size = len(dataset) - train_size 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print(f"Dataset prepared: {len(dataset)} samples")
print(f"Training set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples")

__all__ = ["train_loader", "val_loader", "train_dataset", "val_dataset"]
