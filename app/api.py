from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from pathlib import Path

# Загрузка NLP-ресурсов
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Инициализация FastAPI
app = FastAPI(title="Sentiment Analysis API")

# Инициализация шаблонов Jinja2
templates = Jinja2Templates(directory="templates")

# Модель для предсказания (загрузим её позже)
model = None

# Класс для входных данных
class TextInput(BaseModel):
    text: str

# Функция предобработки текста
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Загрузка модели при старте сервера
@app.on_event("startup")
def load_model():
    global model
    try:
        MODEL_PATH = Path("/model/model.pkl")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# HTML форма для ввода текста
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Обработка формы
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, text: str = Form(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        processed_text = preprocess_text(text)
        prediction = model.predict([processed_text])[0]
        return templates.TemplateResponse("result.html", {
            "request": request,
            "text": text,
            "sentiment": prediction
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# Эндпоинт для проверки работы сервера
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "API is running"}