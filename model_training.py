import torch
from torch import nn, optim
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from tqdm import tqdm
from dataset_preparation import train_dataset, val_dataset  
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

DATASET_DIR = "flickr8k_dataset"
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(device)

feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

small_train_size = 1600  
small_val_size = 400 
small_train_dataset, _ = random_split(train_dataset, [small_train_size, len(train_dataset) - small_train_size])
small_val_dataset, _ = random_split(val_dataset, [small_val_size, len(val_dataset) - small_val_size])

train_loader = DataLoader(small_train_dataset, batch_size=8, shuffle=True) 
val_loader = DataLoader(small_val_dataset, batch_size=8)

def train_model(model, train_loader, val_loader, epochs=1, accumulation_steps=4, patience=3, warmup_steps=100):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,  
        num_training_steps=len(train_loader) * epochs
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")):
            optimizer.zero_grad()
            
            pixel_values = batch["image"].to(device)
            input_ids = batch["caption"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                pixel_values=pixel_values,
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

        print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                pixel_values = batch["image"].to(device)
                input_ids = batch["caption"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                    labels=input_ids,
                )
                val_loss += outputs.loss.item()

        print(f"Epoch {epoch + 1} Validation Loss: {val_loss / len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            model.save_pretrained(f"best_model_epoch_{epoch + 1}")
            tokenizer.save_pretrained(f"best_model_epoch_{epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

train_model(model, train_loader, val_loader, epochs=10)

model.save_pretrained("fine_tuned_vit_gpt2")
tokenizer.save_pretrained("fine_tuned_vit_gpt2")
print("Model saved to fine_tuned_vit_gpt2/")