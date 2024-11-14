from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from io import BytesIO

app = FastAPI()

def preprocess_image(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Adjust the level as needed

    # Resize image
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

    # Binarize (Thresholding)
    image = image.point(lambda x: 0 if x < 128 else 255, '1')

    return image

class ImageRequest(BaseModel):
    image: str

@app.post("/api/retrieve-text")
async def retrieve_text(request: ImageRequest):
    image_url = request.image
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve image {e}")
    
    
    try:
        image = Image.open(BytesIO(response.content))
        processed_image = preprocess_image(image)
        config = "--oem 3"
        extracted_text = pytesseract.image_to_string(processed_image, config=config, lang="eng")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image {e}")
    
    
    return {"extracted_text": extracted_text}