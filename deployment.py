from fastapi import FastAPI, File, UploadFile
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTFeatureExtractor
from PIL import Image
import torch
import io

app = FastAPI()

MODEL_PATH = "fine_tuned_vit_gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    #pre-processing the uploaded image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    
    #generating caption
    outputs = model.generate(pixel_values)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"caption": caption}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



