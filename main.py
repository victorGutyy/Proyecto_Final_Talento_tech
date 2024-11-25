from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Crear la aplicación FastAPI
app = FastAPI()

# Montar la carpeta 'static' para servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Diccionario para almacenar el nombre del usuario
user_data = {"name": None}

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())

@app.get("/chat", response_class=HTMLResponse)
async def get_chat():
    with open("chat.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())


# Datos de preguntas y respuestas
data = [
    {"category": "Viabilidad por región", "phrase": "¿Qué zonas del país son más efectivas para instalar paneles solares?"},
    {"category": "Época de productividad", "phrase": "¿Cuándo son más productivos los paneles solares?"},
    {"category": "Incentivos", "phrase": "¿Qué incentivos hay para paneles solares?"},
    {"category": "Sector agrícola", "phrase": "¿Es útil la energía solar para una finca hotel?"},
    {"category": "Sector agrícola", "phrase": "¿Cómo puede ayudar la energía solar en galpones?"},
    {"category": "Sector agrícola", "phrase": "¿Es rentable usar paneles solares en fincas lecheras?"}
]

responses = {
    "Viabilidad por región": "Las zonas con mayor irradiación solar, como La Guajira, son ideales para paneles solares.",
    "Época de productividad": "Los meses con menos lluvias y más sol son los más productivos para los paneles solares.",
    "Incentivos": "Existen incentivos fiscales en Colombia para fomentar la instalación de paneles solares.",
    "Sector agrícola": "La energía solar es altamente útil en el sector agrícola para riego, iluminación y otros usos."
}

# Entrenar el modelo para encontrar respuestas basadas en similitud
vectorizer = TfidfVectorizer().fit([item["phrase"] for item in data])

def find_best_match(question: str) -> Optional[str]:
    question_vector = vectorizer.transform([question])
    data_vectors = vectorizer.transform([item["phrase"] for item in data])
    similarities = cosine_similarity(question_vector, data_vectors).flatten()
    max_similarity_index = similarities.argmax()
    if similarities[max_similarity_index] > 0.5:
        return data[max_similarity_index]["category"]
    return None

# Modelo de entrada para preguntas y nombre
class UserQuery(BaseModel):
    question: str

class NameRequest(BaseModel):
    name: str

@app.post("/set_name")
async def set_name(request: NameRequest):
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío.")
    user_data["name"] = request.name.strip()
    return {"message": f"¡Hola, {user_data['name']}! Bienvenido al Chatbot de Energías Renovables. ¿En qué puedo ayudarte?"}

@app.post("/chat")
async def chat(request: UserQuery):
    if not user_data["name"]:
        return {"response": "Por favor, proporciona tu nombre primero."}
    category = find_best_match(request.question.lower())
    response = responses.get(category, "Lo siento, no entiendo la pregunta. ¿Puedes reformularla?")

    return {"response": response}


    #return {"response": response} 

solar_production_by_department = {
    "amazonas": 160,
    "antioquia": 150,
    "arauca": 170,
    "atlantico": 200,
    "bolivar": 190,
    "boyaca": 140,
    "caldas": 130,
    "caqueta": 150,
    "casanare": 170,
    "cauca": 150,
    "cesar": 180,
    "choco": 120,
    "cordoba": 190,
    "cundinamarca": 140,
    "guainia": 160,
    "guaviare": 160,
    "huila": 160,
    "la guajira": 210,
    "magdalena": 200,
    "meta": 160,
    "nariño": 130,
    "norte de santander": 170,
    "putumayo": 140,
    "quindio": 130,
    "risaralda": 130,
    "san andres y providencia": 190,
    "santander": 160,
    "sucre": 190,
    "tolima": 150,
    "valle del cauca": 140,
    "vaupes": 160,
    "vichada": 170,
    "bogota": 130,
}

# Modelo de solicitud para calcular el ahorro
class SavingsRequest(BaseModel):
    department: str
    monthly_consumption_kwh: float
    price_per_kwh: float

@app.post("/calculate_savings")
async def calculate_savings(request: SavingsRequest):
    # Normalizar el nombre del departamento
    normalized_department = request.department.lower()
    if normalized_department not in solar_production_by_department:
        raise HTTPException(
            status_code=400,
            detail=f"Departamento no reconocido. Los departamentos disponibles son: {', '.join(solar_production_by_department.keys())}."
        )

    # Obtener la producción solar mensual promedio del departamento
    monthly_solar_production = solar_production_by_department[normalized_department]
    # Calcular el ahorro mensual y anual
    savings_per_month = min(request.monthly_consumption_kwh, monthly_solar_production) * request.price_per_kwh
    annual_savings = savings_per_month * 12

    return {
        
        "message": f"En {request.department}, con un consumo mensual de {request.monthly_consumption_kwh} kWh y un precio de {request.price_per_kwh} COP por kWh, podrías ahorrar aproximadamente {round(annual_savings, 2)} COP al año instalando paneles solares."
    }
