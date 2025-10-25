from fastapi import FastAPI
from app.routers.routes_classification import router as classroutes_classification_router

app = FastAPI(title="API de Classificação de Grãos", version="1.0")

app.include_router(classroutes_classification_router)

@app.get("/")
def root():
    return {"mensagem": "API de Classificação de Grãos está no ar"}
