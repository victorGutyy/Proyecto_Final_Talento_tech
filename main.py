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
    
@app.get("/portal", response_class=HTMLResponse)
async def get_portal():
    with open("portal.html", "r", encoding="utf-8") as file:
        return HTMLResponse(file.read())



# Datos de preguntas y respuestas
data = [
    {"category": "Viabilidad por región", "phrase": "¿Qué zonas del país son más efectivas para instalar paneles solares?"},
    {"category": "Época de productividad", "phrase": "¿Cuándo son más productivos los paneles solares?"},
    {"category": "Incentivos", "phrase": "¿Qué incentivos hay para paneles solares?"},
    {"category": "Sector agrícola", "phrase": "¿Es útil la energía solar para una finca hotel?"},
    {"category": "Sector agrícola", "phrase": "¿Cómo puede ayudar la energía solar en galpones?"},
    {"category": "Sector agrícola", "phrase": "¿Es rentable usar paneles solares en fincas lecheras?"},
    {"category": "Impacto ambiental", "phrase": "¿Qué beneficios ambientales ofrecen los paneles solares?"},
    {"category": "Vida útil", "phrase": "¿Cuánto tiempo duran los paneles solares?"},
    {"category": "Mantenimiento", "phrase": "¿Qué tipo de mantenimiento requieren los paneles solares?"},
    {"category": "Autonomía energética", "phrase": "¿Es posible ser completamente independiente con energía solar?"},
    {"category": "Energías renovables", "phrase": "¿Qué otras energías renovables existen además de la solar?"},
    {"category": "Almacenamiento", "phrase": "¿Qué tipo de baterías se usan para almacenar energía solar?"},
    {"category": "Rentabilidad", "phrase": "¿En cuánto tiempo se recupera la inversión en paneles solares?"},
    {"category": "Instalación", "phrase": "¿Es complicada la instalación de paneles solares?"},
    {"category": "Normatividad", "phrase": "¿Qué normativas existen en Colombia sobre energía solar?"},
    {"category": "Costo inicial", "phrase": "¿Cuál es el costo inicial promedio de un sistema de paneles solares?"},
    {"category": "Potencia requerida", "phrase": "¿Cómo se calcula la potencia necesaria para un hogar?"},
    {"category": "Clima y producción", "phrase": "¿Cómo afecta el clima la producción de energía solar?"},
    {"category": "Futuro", "phrase": "¿Cuál es el futuro de la energía solar en Colombia?"},
    {"category": "Empresas", "phrase": "¿Cómo pueden las empresas beneficiarse de la energía solar?"},
    {"category": "Ahorro", "phrase": "¿Cuánto se puede ahorrar al instalar paneles solares?"},
    {"category": "Innovación", "phrase": "¿Qué innovaciones existen actualmente en energía solar?"},
    {"category": "Desventajas", "phrase": "¿Cuáles son las desventajas de la energía solar?"},
    {"category": "Educación", "phrase": "¿Qué programas educativos promueven la energía solar?"},
    {"category": "Impacto en la factura", "phrase": "¿Cómo afecta la energía solar la factura eléctrica?"},
    {"category": "Residuos", "phrase": "¿Qué se hace con los paneles solares al final de su vida útil?"},
    {"category": "Zonas rurales", "phrase": "¿Cómo puede la energía solar beneficiar a zonas rurales?"},
    {"category": "Energía híbrida", "phrase": "¿Qué es un sistema híbrido de energía solar?"},
    {"category": "Durabilidad", "phrase": "¿Cómo se asegura la durabilidad de los paneles solares?"},
    {"category": "Microinversores", "phrase": "¿Qué son los microinversores en un sistema solar?"}
]

responses = {
    "Viabilidad por región": "Las zonas con mayor irradiación solar, como La Guajira, son ideales para paneles solares.",
    "Época de productividad": "Los meses con menos lluvias y más sol son los más productivos para los paneles solares.",
    "Incentivos": "Existen incentivos fiscales en Colombia para fomentar la instalación de paneles solares.",
    "Sector agrícola": "La energía solar es altamente útil en el sector agrícola para riego, iluminación y otros usos.",
    "Impacto ambiental": "Los paneles solares reducen la dependencia de combustibles fósiles, disminuyendo las emisiones de CO2.",
    "Vida útil": "La vida útil promedio de un panel solar es de 25 a 30 años.",
    "Mantenimiento": "El mantenimiento consiste principalmente en limpieza y revisiones anuales.",
    "Autonomía energética": "Es posible alcanzar la independencia energética combinando paneles solares y baterías.",
    "Energías renovables": "Además de la solar, existen energías eólica, geotérmica, hidráulica y biomasa.",
    "Almacenamiento": "Las baterías de litio son las más utilizadas por su eficiencia y durabilidad.",
    "Rentabilidad": "La inversión se recupera en promedio en 5 a 10 años dependiendo del consumo.",
    "Instalación": "La instalación requiere de profesionales certificados para garantizar un sistema seguro.",
    "Normatividad": "En Colombia, la Ley 1715 de 2014 regula el uso de energías renovables.",
    "Costo inicial": "El costo inicial varía según el tamaño del sistema, desde COP 10 millones en adelante.",
    "Potencia requerida": "Se calcula según el consumo energético promedio del hogar en kWh.",
    "Clima y producción": "Los días nublados reducen la producción, pero los sistemas siguen generando energía.",
    "Futuro": "La energía solar está en constante crecimiento gracias a los avances tecnológicos y mayor accesibilidad.",
    "Empresas": "Las empresas pueden reducir costos operativos y mejorar su imagen ambiental usando energía solar.",
    "Ahorro": "El ahorro depende del consumo y el costo de la energía, pero puede llegar al 50% o más.",
    "Innovación": "Los paneles solares bifaciales y los sistemas flotantes son algunas innovaciones recientes.",
    "Desventajas": "Las principales desventajas son el costo inicial alto y la dependencia de la luz solar.",
    "Educación": "Programas como el SENA promueven el aprendizaje en energías renovables.",
    "Impacto en la factura": "La energía solar puede reducir la factura eléctrica a cero en algunos casos.",
    "Residuos": "Los paneles solares son reciclables en un 80%, reduciendo el impacto ambiental.",
    "Zonas rurales": "En zonas rurales, los paneles solares pueden proveer energía en áreas sin conexión a la red.",
    "Energía híbrida": "Un sistema híbrido combina energía solar con otras fuentes, como la red eléctrica.",
    "Durabilidad": "La calidad de los materiales y el mantenimiento adecuado aseguran la durabilidad.",
    "Microinversores": "Los microinversores convierten la energía directamente en cada panel, aumentando la eficiencia."
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
