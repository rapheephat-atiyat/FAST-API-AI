import tensorflow as tf #type: ignore
from tensorflow import keras #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
import numpy as np #type: ignore
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
import io

class PredictionResponse(BaseModel):
    prediction_en: str
    prediction_th: str
    confidence: float

IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    'alpaca', 'american_bison', 'anteater', 'arctic_fox', 'armadillo', 
    'baboon', 'badger', 'blue_whale', 'brown_bear', 'camel', 'cats', 
    'dogs', 'dolphin', 'elephant', 'giraffe', 'groundhog', 
    'highland_cattle', 'horse', 'jackal', 'kangaroo', 'koala', 
    'manatee', 'mongoose', 'mountain_goat', 'opossum', 'orangutan', 
    'otter', 'panda', 'polar_bear', 'porcupine', 'red_panda', 
    'rhinoceros', 'seal', 'sea_lion', 'snow_leopard', 'squirrel', 
    'sugar_glider', 'tapir', 'vampire_bat', 'vicuna', 'walrus', 
    'warthog', 'water_buffalo', 'weasel', 'wildebeest', 'wombat', 
    'yak', 'zebra'
]
ANIMAL_NAMES_TH = {
    'alpaca': 'อัลปากา',
    'american_bison': 'ควายไบซันอเมริกัน',
    'anteater': 'ตัวกินมด',
    'arctic_fox': 'สุนัขจิ้งจอกอาร์กติก',
    'armadillo': 'ตัวนิ่ม',
    'baboon': 'ลิงบาบูน',
    'badger': 'หมูดิน',
    'blue_whale': 'วาฬสีน้ำเงิน',
    'brown_bear': 'หมีสีน้ำตาล',
    'camel': 'อูฐ',
    'cats': 'แมว',
    'dogs': 'หมา',
    'dolphin': 'โลมา',
    'elephant': 'ช้าง',
    'giraffe': 'ยีราฟ',
    'groundhog': 'กระรอกดิน',
    'highland_cattle': 'วัวไฮแลนด์',
    'horse': 'ม้า',
    'jackal': 'หมาจิ้งจอก',
    'kangaroo': 'จิงโจ้',
    'koala': 'โคอาลา',
    'manatee': 'พะยูนแมนนาที',
    'mongoose': 'พังพอน',
    'mountain_goat': 'แพะภูเขา',
    'opossum': 'โอพอสซัม',
    'orangutan': 'ลิงอุรังอุตัง',
    'otter': 'นาก',
    'panda': 'แพนด้า',
    'polar_bear': 'หมีขั้วโลก',
    'porcupine': 'เม่น',
    'red_panda': 'แพนด้าแดง',
    'rhinoceros': 'แรด',
    'seal': 'แมวน้ำ',
    'sea_lion': 'สิงโตทะเล',
    'snow_leopard': 'เสือดาวหิมะ',
    'squirrel': 'กระรอก',
    'sugar_glider': 'ชูการ์ไกลเดอร์',
    'tapir': 'สมเสร็จ',
    'vampire_bat': 'ค้างคาวแวมไพร์',
    'vicuna': 'บิกุญญา',
    'walrus': 'วอลรัส',
    'warthog': 'หมูป่า',
    'water_buffalo': 'ควายบ้าน',
    'weasel': 'เพียงพอน',
    'wildebeest': 'นู',
    'wombat': 'วอมแบต',
    'yak': 'จามรี',
    'zebra': 'ม้าลาย'
}

model = keras.models.load_model("animal_detector_model.h5")
print("Model loaded successfully!")

app = FastAPI()

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize(IMAGE_SIZE)
    
    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch


@app.get("/")
@app.head("/")
async def root():
    return JSONResponse({"Hello": "World"})

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) :
    image_bytes = await file.read()
    processed_image = prepare_image(image_bytes)
    predictions = model.predict(processed_image)
    
    predicted_index = np.argmax(predictions[0])
    englishName = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions[0]))

    CONFIDENCE_THRESHOLD = 0.5
    
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction_en": "unknown",
            "prediction_th": "ไม่รู้จัก",
            "confidence": confidence,
            "detail": "Model is not confident enough."
        }
    else:
        thaiName = ANIMAL_NAMES_TH.get(englishName, englishName)

        return JSONResponse({
            "prediction_en": englishName,
            "prediction_th": thaiName,
            "confidence": confidence,
            # "raw_scores": predictions[0].tolist()
        })