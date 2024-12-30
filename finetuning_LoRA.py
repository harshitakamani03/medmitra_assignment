from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast, AdamW
from torch.utils.data import DataLoader
import torch
from peft import get_peft_model, LoraConfig, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained('google/vit-base-patch16-224-in21k')
tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased')

lora_config = LoraConfig(
    r=8,  
    lora_alpha=16, 
    lora_dropout=0.1,  
    task_type=TaskType.IMAGE_CAPTIONING 
)

lora_model = get_peft_model(model, lora_config)

#freezing the model layers and finetuning only the LoRA layers
for param in lora_model.base_model.parameters():
    param.requires_grad = False

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]['image']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        caption = self.dataset[idx]['caption']
        encoding = self.tokenizer(caption, padding='max_length', truncation=True, max_length=128)

        return img, encoding

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = ImageTextDataset(dataset, tokenizer, transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optimizer = AdamW(lora_model.parameters(), lr=5e-5)

lora_model.train()
for epoch in range(5):
    total_loss = 0
    for imgs, encodings in train_dataloader:
        optimizer.zero_grad()

        imgs = imgs.to(device)  
        input_ids = encodings['input_ids'].to(device)
        outputs = lora_model(input_ids=input_ids, pixel_values=imgs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")

lora_model.save_pretrained('./lora_finetuned_model')
